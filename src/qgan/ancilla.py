# Copyright 2025 GIQ, Universitat Autònoma de Barcelona
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ancilla post-processing tools — PennyLane rewrite.

get_max_entangled_state_with_ancilla_if_needed(size) -> (gen_state, target_state)
project_ancilla_zero(state, renormalize)              -> (projected_state, prob)
trace_out_ancilla(state)                              -> sampled_state
get_final_gen_state_for_discriminator(state)           -> final_state
"""

import numpy as np
import torch
import pennylane as qml
import torch

from config import CFG


# -- MAXIMALLY ENTANGLED STATE PREPARATION -------------------
def get_max_entangled_state_with_ancilla_if_needed(size: int) -> np.ndarray:
    """Get the maximally entangled state for the system size (With Ancilla if needed).
    Args:
        size (int): the size of the system.

    Returns:
        tuple[np.ndarray]: the maximally entangled states, plus ancilla
        if needed for generation and target.
    """
    # Generate the maximally entangled state for the system size
    state = np.zeros(2 ** (2 * size), dtype=complex)
    dim_register = 2**size
    for i in range(dim_register):
        state[i * dim_register + i] = 1.0
    state /= np.sqrt(dim_register)

    # Add ancilla qubit at the end, if needed
    initial_state_with_ancilla = np.kron(state, np.array([1, 0], dtype=complex))

    # Different conditions for gen and target:
    initial_state_for_gen = initial_state_with_ancilla if CFG.extra_ancilla else state
    initial_state_for_target = initial_state_with_ancilla if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else state

    return np.asmatrix(initial_state_for_gen).T, np.asmatrix(initial_state_for_target).T


# -- HAAR RANDOM STATE PREPARATION -------------------
def haar_random_batch(dim: int, batch_size: int) -> list[torch.Tensor]:
    """Generate batch of Haar-random pure states, with ancilla if needed.
    In this case, we reuse the vector for the target (we multiply the target Unitary to the vector)
    and for the generator (we transform vector to quantum state using qml.StatePrep)

    Args:
        dim: Dimension of the system Hilbert space (2^system_size).
        batch_size: Number of states to generate.

    Returns:
        List of torch tensors, each of shape (dim,) or (2*dim,) with ancilla.
    """
    ancilla_zero = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    batch_raw = []
    batch_inputs = []
    for _ in range(batch_size):
        real = torch.randn(dim, dtype=torch.float32)
        imag = torch.randn(dim, dtype=torch.float32)
        v = torch.complex(real, imag)
        v = v / v.norm()
        batch_raw.append(v)
        batch_inputs.append(torch.kron(v, ancilla_zero) if CFG.extra_ancilla else v)
    return batch_raw, batch_inputs

# -- DETERMINISTIC ALL-PAIRS STATE SET ---------------------
# Cached module-level: built once on first use, reused across all iterations.
# A new build is triggered only when dim changes (e.g. between experiments).
_DET_STATES: list[torch.Tensor] | None = None
_DET_DIM: int | None = None
 
 
def _build_deterministic_state_set(dim: int) -> list[torch.Tensor]:
    """Build the all-pairs deterministic state set for unitary identification.

    Args:
        dim: Hilbert space dimension d = 2^n.
 
    Returns:
        List of d(d+1)/2 torch tensors of shape (dim,), dtype complex64.
        Order: |0>, |1>, ..., |d-1>, (|0>+|1>)/sqrt(2), (|0>+|2>)/sqrt(2), ...
    """
    states: list[torch.Tensor] = []
 
    # Basis states |j>
    for j in range(dim):
        v = torch.zeros(dim, dtype=torch.complex64)
        v[j] = 1.0
        states.append(v)
 
    # Pair superpositions 
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            v = torch.zeros(dim, dtype=torch.complex64)
            v[i] = inv_sqrt2
            v[j] = inv_sqrt2
            states.append(v)
 
    return states
 
def deterministic_batch_size(dim: int) -> int:
    """Return M = d(d+1)/2, the size of the deterministic all-pairs set."""
    return dim * (dim + 1) // 2
 
def comp_random_batch(dim: int, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return a random subset of the deterministic all-pairs state set.

    The full set has size M = d(d+1)/2. If `batch_size >= M` (or <= 0) the
    full set is returned; otherwise `batch_size` states are drawn uniformly
    at random without replacement from the cached set.

    Args:
        dim: Hilbert space dimension d = 2^n.
        batch_size: Number of states to draw, in [1, M]. Values >= M return
                    the full set.

    Returns:
        (batch_raw, batch_inputs):
            batch_raw    : list of dim-vectors (no ancilla), length = batch_size.
            batch_inputs : list of (dim,) or (2*dim,) vectors depending on
                           CFG.extra_ancilla; ancilla is kron'd with |0⟩.
    """
    global _DET_STATES, _DET_DIM
    # Build (or rebuild if dim changed) the cached state set.
    if _DET_STATES is None or _DET_DIM != dim:
        _DET_STATES = _build_deterministic_state_set(dim)
        _DET_DIM = dim

    max_bs = deterministic_batch_size(dim)
    if batch_size <= 0 or batch_size >= max_bs:
        selected = list(_DET_STATES)
    else:
        idx = torch.randperm(max_bs)[:batch_size].tolist()
        selected = [_DET_STATES[i] for i in idx]

    ancilla_zero = torch.tensor([1.0, 0.0], dtype=torch.complex64)
    batch_raw: list[torch.Tensor] = list(selected)
    batch_inputs: list[torch.Tensor] = [
        torch.kron(v, ancilla_zero) if CFG.extra_ancilla else v for v in selected
    ]
    return batch_raw, batch_inputs

def get_random_batch(dim: int, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Dispatch to the input-state sampler selected by CFG.batch_mode."""
    if CFG.batch_mode == "haar":
        return haar_random_batch(dim, batch_size)
    if CFG.batch_mode == "comp":
        return comp_random_batch(dim, batch_size)
    raise ValueError(f"Unknown batch_mode: {CFG.batch_mode}")

def prepare_batch_targets(
    batch_raw: list[torch.Tensor],
    batch_inputs: list[torch.Tensor],
    target_op: torch.Tensor,
) -> list[torch.Tensor]:
    """Give us the target state
    If ancilla_mode == "pass":
        target_i = (U_target \otimes I_2) |\psi_i> \otimes |0>
    Otherwise:
        target_i = U_target |\psi_i>

    Args:
        batch_raw               : States without ancilla (dim 2^n).
        batch_inputs            : States with ancilla
        target_op (torch tensor): Pre-computed target operator.

    Returns:
        List of target state vectors, each 1D torch tensor.
    """
    if CFG.extra_ancilla and CFG.ancilla_mode == "pass":
        return [(target_op @ psi).reshape(-1) for psi in batch_inputs]
    else:
        return [(target_op @ psi).reshape(-1) for psi in batch_raw]

# -- ANCILLA POST-PROCESSING -----------------------------
def _project_ancilla_zero(state: torch.Tensor, renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the last qubit onto |0> (torch version, differentiable).

    Keeps only the amplitudes where the ancilla is in |0> (even indices).

    Args:
        state: Statevector as a 1D torch tensor, shape (2^N,).
        renormalize: Whether to renormalize the projected state.

    Returns:
        (projected_state, prob_zero):
            projected_state: shape (2^(N-1), 1) column vector.
            prob_zero: scalar, probability of ancilla being |0>.
    """
    # Even indices = ancilla in |0>
    projected = state[::2]

    # Norm^2 = probability of |0>
    prob = torch.sum(projected.conj() * projected).real

    if prob.item() < 1e-15:
        n_system_qubits = CFG.system_size * (2 if CFG.use_choi else 1)  # choi + system register
        return (
            torch.zeros(2**n_system_qubits, 1, dtype=torch.complex64),
            torch.tensor(0.0, dtype=torch.float32),
        )

    if renormalize:
        if CFG.ancilla_project_norm == "re-norm":
            projected = projected / torch.sqrt(prob)
        elif CFG.ancilla_project_norm != "pass":
            raise ValueError(f"Unknown ancilla_project_norm: {CFG.ancilla_project_norm}")

    return projected.reshape(-1, 1), prob


def _trace_out_ancilla(state: torch.Tensor) -> torch.Tensor:
    """Trace out the last qubit and return a sampled pure state (torch version).

    This operation involves eigendecomposition + stochastic sampling,
    which breaks differentiability. Same limitation as the original numpy version.

    Args:
        state: Statevector as a 1D torch tensor, shape (2^N,).

    Returns:
        Sampled pure state, shape (2^(N-1), 1) column vector.
    """
    n_total = state.shape[0]
    n_qubits = int(np.log2(n_total))
    n_system = n_qubits - 1

    # Full density matrix |psi><psi|
    rho_full = torch.outer(state, state.conj())

    # Reshape to (2^n_system, 2, 2^n_system, 2) and trace over ancilla (last qubit)
    dim_sys = 2**n_system
    rho_reshaped = rho_full.reshape(dim_sys, 2, dim_sys, 2)
    # Trace over ancilla: sum over ancilla indices (axes 1 and 3)
    rho_reduced = torch.einsum("iaja->ij", rho_reshaped)

    # Sample a pure state from the reduced density matrix
    rho_np = rho_reduced.detach().numpy()
    eigvals, eigvecs = np.linalg.eigh(rho_np)
    eigvals = np.maximum(eigvals, 0)
    eigvals /= np.sum(eigvals)

    idx = np.random.choice(len(eigvals), p=eigvals)
    sampled = torch.tensor(eigvecs[:, idx], dtype=torch.complex64)

    return sampled.reshape(-1, 1)


def get_final_gen_state_torch(total_output_state: torch.Tensor) -> torch.Tensor:
    """Process the generator output state according to ancilla_mode.

    Routes to the appropriate post-processing:
        - "pass":    state goes directly to discriminator
        - "project": project ancilla onto |0>, remove it
        - "trace":   trace out ancilla, sample pure state

    Args:
        total_output_state: Generator output, 1D torch tensor of shape (2^N,).

    Returns:
        Column vector, shape (d, 1), as torch tensor.
    """
    total_output_state = total_output_state.to(torch.complex64)  # Ensure correct dtype
    if not CFG.extra_ancilla:
        return total_output_state.reshape(-1, 1)

    if CFG.ancilla_mode == "pass":
        return total_output_state.reshape(-1, 1)
    if CFG.ancilla_mode == "project":
        projected, _ = _project_ancilla_zero(total_output_state)
        return projected
    if CFG.ancilla_mode == "trace":
        return _trace_out_ancilla(total_output_state)

    raise ValueError(f"Unknown ancilla_mode: {CFG.ancilla_mode}")

def extract_gen_unitary(gen) -> torch.Tensor:
    """Extract U_gen as a (D, D) torch tensor using qml.matrix.

    D = 2^(system_size + 1) if ancilla is present and ancilla_mode == "pass",
    otherwise D = 2^system_size.

    Only meaningful when the post-ancilla map is linear (i.e. ancilla_mode == "pass"
    or no ancilla). For "project"/"trace" the generator is not a unitary.
    """
    from qgan.generator import Ansatz, _wire_layout

    _, gen_wires, ancilla_wire, total_wires = _wire_layout()

    def ansatz_only(params):
        Ansatz.apply(params, gen_wires, ancilla_wire)

    with torch.no_grad():
        U = qml.matrix(ansatz_only, wire_order=list(range(total_wires)))(
            gen._effective_params()
        )
    return U.to(torch.complex64)


def process_fidelity(U_gen: torch.Tensor, U_target: torch.Tensor) -> float:
    """F_proc = |Tr(U_target^† U_gen)|^2 / d^2.

    ⟨U_target | U_gen⟩_HS normalised. Equals the Choi-state fidelity.
    """
    d = U_target.shape[0]
    tr = torch.trace(U_target.conj().T @ U_gen)
    return (tr.abs() ** 2 / d ** 2).item()