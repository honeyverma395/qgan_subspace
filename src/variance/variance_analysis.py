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
"""Gradient variance analysis for the QGAN generator (BP diagnostic).

Instead of a full training loop, we sample many random parameter vectors
theta Uniform[0, 2pi) together with a fresh random discriminator
(alpha, beta Uniform[-1, 1)), compute the gradient of the Wasserstein
loss via autograd (same detach pattern as Generator.compute_loss, so gradient
flows through theta only), and estimate Var[dW/dtheta_k] for each angle
k.

Sweep configs:
    1. no_ancilla            : extra_ancilla = False
    2. ancilla_total         : extra_ancilla = True, ancilla_topology = "total"
    3. ancilla_bridge        : extra_ancilla = True, ancilla_topology = "bridge", ancilla_connect_to = None
    4. ancilla_shortBridge   : extra_ancilla = True, ancilla_topology = "bridge", ancilla_connect_to = 1
"""

import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Paths --------------------------------------------------------------
# Go up one level for src/ (needed for imports), two levels for the
# project root (where we put the output folder, next to generated_data/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import CFG
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.cost_functions import _calc_wasserstein
from qgan.ancilla import (
    get_final_gen_state_torch,
    get_max_entangled_state_with_ancilla_if_needed,
    haar_random_batch,
    prepare_batch_targets,
)
from qgan.target import get_target_operator
from qgan.generator import (
    _get_ansatz_terms,
    _layer_coupling,
    _count_ancilla_coupling_params,
    _1Q_GATES,
    _2Q_GATES,
)


# -- Script parameters -------------------------------------------------
N_SAMPLES = 20000
SEED = None  # For reproducibility


# -- Config management -------------------------------------------------
# Fields we overwrite during the sweep
_SWEEPED_FIELDS = (
    "extra_ancilla",
    "ancilla_topology",
    "ancilla_connect_to",
)

def _snapshot_cfg() -> dict:
    """Save values of the CFG fields we will modify."""
    return {f: getattr(CFG, f) for f in _SWEEPED_FIELDS}

def _restore_cfg(snapshot: dict) -> None:
    for f, v in snapshot.items():
        setattr(CFG, f, v)

def _apply_config(name: str) -> None:
    """Overwrite CFG with the requested sweep config."""
    if name == "no_ancilla":
        CFG.extra_ancilla = False
        # other are irrelevant when extra_ancilla is False
    elif name == "ancilla_total":
        CFG.extra_ancilla = True
        CFG.ancilla_topology = "total" 
        CFG.ancilla_connect_to = None
    elif name == "ancilla_bridge":
        CFG.extra_ancilla = True
        CFG.ancilla_topology = "bridge" 
        CFG.ancilla_connect_to = None
    elif name== "ancilla_shortBridge":
        CFG.extra_ancilla = True
        CFG.ancilla_topology = "bridge" 
        CFG.ancilla_connect_to = 1
    else:
        raise ValueError(f"Unknown config: {name}")


# -- Loss --------------------------------------------------------------
def _build_target_state() -> torch.Tensor:
    """Build |target> = (I_choi \otimes U_target [\otimes I_anc]) |Phi+>.
    """
    target_op = torch.tensor(get_target_operator(), dtype=torch.complex64)
    _, initial_state_final = get_max_entangled_state_with_ancilla_if_needed(
        CFG.system_size
    )
    initial_state = torch.tensor(
        np.asarray(initial_state_final), dtype=torch.complex64
    ).reshape(-1)
    return (target_op @ initial_state).reshape(-1)

def _wasserstein_loss(
    gen: Generator,
    dis: Discriminator,
    target_state: torch.Tensor,
) -> torch.Tensor:
    """Compute the Wasserstein loss function

    Gradient flows through theta only: discriminator matrices are detached.

    alpha, beta are kept random (per-sample), so the variance aggregates over
    both theta and a random discriminator, the gradient the generator would 
    see at a random point of the joint landscape at the start of training.
    """
    total_gen = gen.get_total_gen_state()
    final_gen = get_final_gen_state_torch(total_gen).reshape(-1)

    # Detach discriminator matrices (as Generator._get_detached_matrices)
    with torch.no_grad():
        A, B, psi, phi = dis.get_dis_matrices_rep()
    dis_matrices = (A.detach(), B.detach(), psi.detach(), phi.detach())

    return _calc_wasserstein(final_gen, target_state, dis_matrices)


# -- Gradient sampling -------------------------------------------------

# -- Per-config parameter layout --------------------------------------
def _get_layerwise_ancilla_indices(gen) -> list[dict]:
    """
    Maps out the indices of the ancilla (auxiliary) parameters within the flattened 
    parameter array of the generator.
    
    It iterates through each layer of the quantum circuit, skips the parameters 
    dedicated to the main system, and records the indices for 1-qubit and 2-qubit 
    ancilla gates.
    """
    terms = _get_ansatz_terms()
    n_1q_terms = sum(1 for t in terms if t in _1Q_GATES)
    n_2q_terms = sum(1 for t in terms if t in _2Q_GATES)
    n_sys = CFG.system_size

    layout: list[dict] = []
    idx = 0
    
    for layer_idx in range(CFG.gen_layers):
        # Skip the indices that belong to the core system qubits
        idx += n_1q_terms * n_sys
        idx += n_2q_terms * (n_sys - 1)

        idx_1q_anc, idx_2q_anc = [], []
        
        # If this layer utilizes ancilla coupling, track their parameter indices
        if CFG.extra_ancilla and _layer_coupling(layer_idx):
            if CFG.do_ancilla_1q_gates:
                for _ in range(n_1q_terms):
                    idx_1q_anc.append(idx)
                    idx += 1
                    
            n_anc_2q = _count_ancilla_coupling_params(n_2q_terms, n_sys)
            for _ in range(n_anc_2q):
                idx_2q_anc.append(idx)
                idx += 1

        layout.append({"1q_anc": idx_1q_anc, "2q_anc": idx_2q_anc})
    return layout


def _system_indices(gen) -> np.ndarray:
    """
    Retrieves the indices of the system-only parameters by finding the set 
    difference between all parameters and the ancilla parameters.
    """
    if not gen.ancilla:
        return np.arange(gen.n_params)
        
    anc = np.array(gen._get_ancilla_param_indices(), dtype=int)
    # Return all indices that are NOT in the ancilla list
    return np.setdiff1d(np.arange(gen.n_params), anc, assume_unique=True)

# -- Coupled gradient sampling ----------------------------------------
def sample_gradients_coupled(
    n_samples: int,
    configs: list[str],
) -> dict[str, np.ndarray]:
    """
    Samples gradients for multiple circuit configurations while sharing random seeds.
    
    By ensuring that all configurations in a given sample share the same system 
    parameters and Discriminator state, we isolate the effect of the configuration 
    changes. Ancilla pools are generated dynamically so that larger configs simply 
    extend the shared random sequences used by smaller configs.
    """
    snapshot = _snapshot_cfg()

    # Probe each configuration to understand its parameter layout
    probe: dict[str, dict] = {}
    n_sys_ref: int | None = None
    
    try:
        for name in configs:
            _apply_config(name)
            gen = Generator()
            sys_idx = _system_indices(gen)
            layout = _get_layerwise_ancilla_indices(gen)
            
            probe[name] = {
                "n_params": gen.n_params,
                "sys_idx": sys_idx,
                "layout": layout,
            }
            
            # Ensure all configs have the exact same number of core system parameters
            if n_sys_ref is None:
                n_sys_ref = len(sys_idx) 
            elif len(sys_idx) != n_sys_ref:
                raise RuntimeError(
                    f"System param count differs across configs "
                    f"({n_sys_ref} vs {len(sys_idx)} for {name}). "
                    f"Coupled sampling requires identical system layout."
                )
    finally:
        _restore_cfg(snapshot)

    assert n_sys_ref is not None
    n_layers = CFG.gen_layers

    # Allocate output arrays for the gradients
    grads: dict[str, np.ndarray] = {
        name: np.zeros((n_samples, probe[name]["n_params"]), dtype=np.float64)
        for name in configs
    }

    starttime = datetime.now()
    
    # -- Run the sampling loop
    for i in range(n_samples):
        # Draw ONE random set of system parameters shared across ALL configs for this sample
        sys_vals = np.random.uniform(0, 2 * np.pi, size=n_sys_ref)

        # Initialize empty pools for ancilla parameters. These will be filled on-demand.
        pool_1q: list[list[float]] = [[] for _ in range(n_layers)]
        pool_2q: list[list[float]] = [[] for _ in range(n_layers)]

        # Generate a shared seed to ensure the Discriminator is identical for all configs
        dis_seed = np.random.randint(0, 2**31 - 1)

        for name in configs:
            _apply_config(name)
            try:
                gen = Generator()

                # Build the target state based on the current config's dimension
                if CFG.use_choi:
                    target_state = _build_target_state()
                else:
                    target_op = torch.tensor(
                        get_target_operator(), dtype=torch.complex64
                    )
                    dim = 2 ** CFG.system_size
                    B = CFG.batch_size
                    batch_raw, batch_inputs = haar_random_batch(dim, B)
                    batch_targets = prepare_batch_targets(batch_raw, batch_inputs, target_op)

                # Initialize the Discriminator. 
                # We isolate the RNG state here so the shared seed doesn't permanently 
                # alter the global RNG, which would mess up our ancilla pooling.
                torch_state = torch.get_rng_state()
                np_state = np.random.get_state()
                try:
                    torch.manual_seed(dis_seed)
                    np.random.seed(dis_seed)
                    dis = Discriminator()
                finally:
                    torch.set_rng_state(torch_state)
                    np.random.set_state(np_state)

                # Construct the full parameter array (theta) for this specific config
                theta = np.zeros(probe[name]["n_params"], dtype=np.float64)
                # Even though theta is created from scratch each time, 
                # sys_vals is same for all
                theta[probe[name]["sys_idx"]] = sys_vals 

                # Fill in the ancilla parameters layer by layer
                for layer_idx, slots in enumerate(probe[name]["layout"]):
                    
                    # 1-Q Ancilla: Extend the shared pool if this config needs more parameters
                    need_1q = len(slots["1q_anc"])
                    while len(pool_1q[layer_idx]) < need_1q:
                        pool_1q[layer_idx].append(float(np.random.uniform(0, 2 * np.pi)))
                    
                    # Map the shared pool values into the config's specific parameter slots
                    for j, slot in enumerate(slots["1q_anc"]):
                        theta[slot] = pool_1q[layer_idx][j]

                    # 2-Qubit Ancilla: Extend the shared pool if needed
                    need_2q = len(slots["2q_anc"])
                    while len(pool_2q[layer_idx]) < need_2q:
                        pool_2q[layer_idx].append(float(np.random.uniform(0, 2 * np.pi)))
                    
                    # Map the shared pool values
                    for j, slot in enumerate(slots["2q_anc"]):
                        theta[slot] = pool_2q[layer_idx][j]

                # Load the assembled parameters into the generator
                with torch.no_grad():
                    gen.params.copy_(torch.tensor(theta, dtype=gen.params.dtype))

                if gen.params.grad is not None:
                    gen.params.grad.zero_()

                # Forward pass (calculate loss)
                if CFG.use_choi:
                    loss = _wasserstein_loss(gen, dis, target_state)
                else:
                    loss = _wasserstein_loss_batch(gen, dis, batch_inputs, batch_targets)

                # Backward pass (calculate gradients)
                loss.backward()
                
                # Store the resulting gradients
                grads[name][i] = gen.params.grad.detach().numpy().copy()
                
            finally:
                _restore_cfg(snapshot)

        # Progress tracker
        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"    sample {i + 1}/{n_samples}")

    endtime = datetime.now()
    print(f"\n Run took: {endtime - starttime} time.")
    return grads

# -- Helper for Batch mode ------------
def _wasserstein_loss_batch(
    gen: Generator,
    dis: Discriminator,
    batch_inputs: list[torch.Tensor],
    batch_targets: list[torch.Tensor],
) -> torch.Tensor:
    """Mean Wasserstein loss over a Haar batch. Gradient flows through theta only."""
    with torch.no_grad():
        A, B, psi, phi = dis.get_dis_matrices_rep()
    dis_matrices = (A.detach(), B.detach(), psi.detach(), phi.detach())

    total = torch.tensor(0.0, dtype=torch.float32)
    # We use different haar random states each time
    for inp, tgt in zip(batch_inputs, batch_targets):
        total_gen = gen.get_total_gen_state(inp)
        final_gen = get_final_gen_state_torch(total_gen).reshape(-1)
        total = total + _calc_wasserstein(final_gen, tgt.reshape(-1), dis_matrices)
    return total / len(batch_inputs)

# -- Alignment ---------------------------------------------------------
def _split_system_ancilla(grads: np.ndarray, gen: Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split a (N, n_params) grad matrix into (system cols, ancilla cols).
    """
    if not gen.ancilla:
        return grads, np.empty((grads.shape[0], 0), dtype=grads.dtype)
    
    anc_idx = np.array(gen._get_ancilla_param_indices(), dtype=int)
    all_idx = np.arange(gen.n_params)
    sys_idx = np.setdiff1d(all_idx, anc_idx, assume_unique=True)
    return grads[:, sys_idx], grads[:, anc_idx]


# -- Plotting ----------------------------------------------------------
def plot_variance_sweep(
        results: dict, 
        out_path: str,
        n_sys_params: int,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        ) -> None:
    """Scatter plot: one series per config, system + ancilla-only markers.

    Args:
        results: {config_name: {"var_sys": np.ndarray, "var_anc": np.ndarray}}
        out_path: file path for the PNG.
        n_sys_params: number of system-only parameters (shared x-axis head).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    color_map = {
        "no_ancilla":          "blue",  
        "ancilla_total":       "orange",  
        "ancilla_bridge":      "green",  
        "ancilla_shortBridge":  "peru",  
    }

    for name, data in results.items():
        color = color_map.get(name, None)
        var_sys = data["var_sys"]
        var_anc = data["var_anc"]

        # system params: indices 0 .. n_sys_params - 1
        xs_sys = np.arange(len(var_sys))
        ax.scatter(xs_sys, var_sys, color=color, marker="o", s=28,
                   alpha=0.75, label=f"{name} (system)")

        # ancilla-only params: appended after the system block
        if var_anc.size > 0:
            xs_anc = np.arange(n_sys_params, n_sys_params + len(var_anc))
            ax.scatter(xs_anc, var_anc, color=color, marker="^", s=42,
                       alpha=0.9, edgecolors="black", linewidths=0.5,
                       label=f"{name} (ancilla)")

    # Vertical divider between system and ancilla blocks
    ax.axvline(n_sys_params - 0.5, color="black", linestyle=":", linewidth=1.0,
               alpha=0.6, label=f"system / ancilla boundary (k={n_sys_params})")

    ax.set_yscale("log")
    ax.set_xlabel("parameter index k")
    ax.set_ylabel(r"Var$[\partial W / \partial \theta_k]$")
    h_str = _format_hamiltonian_str()
    if CFG.use_choi:
        ax.set_title(
            f"Variance per parameter: "
            f"(N={N_SAMPLES} samples, {CFG.system_size} qubits, "
            f"{CFG.gen_layers} layers, ansatz={CFG.gen_ansatz}, Choi Mode \n "
            f"{h_str}"
        )
    else:
        ax.set_title(
            f"Variance per parameter: "
            f"(N={N_SAMPLES} samples, {CFG.system_size} qubits, "
            f"{CFG.gen_layers} layers, ansatz={CFG.gen_ansatz}), "
            f"Batch Mode, Batch size={CFG.batch_size} \n"
            f"{h_str}"
        )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

# -- Hamiltonian in plot title
def _format_hamiltonian_str() -> str:
    """Return a short LaTeX-ish string describing the current target Hamiltonian."""
    if CFG.target_hamiltonian == "custom_h":
        parts = []
        for s, t in zip(CFG.custom_hamiltonian_strengths, CFG.custom_hamiltonian_terms):
            parts.append(f"{s:g}·{t}")
        body = " + ".join(parts)
        return f"H = {body}"
    return f"H = {CFG.target_hamiltonian}"

# -- Main --------------------------------------------------------------
def run_sweep() -> None:
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    # Sanity Check
    if CFG.use_choi:
        print("Mode: Choi")
    else:
        print(f"Mode: Haar batching (B={CFG.batch_size})")

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    out_dir = os.path.join(
        _PROJECT_ROOT,
        "variance_analysis",
        timestamp,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    snapshot = _snapshot_cfg()
    configs = ["no_ancilla", "ancilla_bridge", "ancilla_shortBridge", "ancilla_total"]

    # -- Config snapshot: write to file AND print before running -------
    config_summary = (
        f"N_SAMPLES = {N_SAMPLES}\n"
        f"SEED = {SEED}\n"
        f"configs = {configs}\n\n"
        + CFG.show_data()
    )
    with open(os.path.join(out_dir, "config.txt"), "w") as fh:
        fh.write(config_summary)
    print("\n" + config_summary)

    results: dict[str, dict[str, np.ndarray]] = {}
    n_sys_params_ref = None

    try:
        # Coupled sampling: shared theta_sys, shared discriminator, shared
        # ancilla random pools (per layer, extended on demand).
        grads_by_config = sample_gradients_coupled(N_SAMPLES, configs)

        for name in configs:
            _apply_config(name)
            try:
                probe_gen = Generator()
                grads = grads_by_config[name]
                grads_sys, grads_anc = _split_system_ancilla(grads, probe_gen)
            finally:
                _restore_cfg(snapshot)

            var_sys = np.var(grads_sys, axis=0)
            var_anc = np.var(grads_anc, axis=0) if grads_anc.size > 0 else np.array([])

            np.save(os.path.join(out_dir, f"grads_{name}.npy"), grads)
            np.save(os.path.join(out_dir, f"variance_{name}.npy"),
                    np.concatenate([var_sys, var_anc]))

            results[name] = {"var_sys": var_sys, "var_anc": var_anc}

            if n_sys_params_ref is None:
                n_sys_params_ref = len(var_sys)
            elif len(var_sys) != n_sys_params_ref:
                print(
                    f"  WARNING: system param count changed across configs "
                    f"({n_sys_params_ref} vs {len(var_sys)})."
                )

            print(f"  [{name}] n_params = {grads.shape[1]}  "
                  f"(system = {len(var_sys)}, ancilla-only = {len(var_anc)})")
            print(f"  [{name}] mean Var(system)     = {var_sys.mean():.3e}")
            if var_anc.size > 0:
                print(f"  [{name}] mean Var(ancilla)    = {var_anc.mean():.3e}")

    finally:
        _restore_cfg(snapshot)

    # -- Plot -----------------------------------------------------------
    plot_path = os.path.join(out_dir, "variance_plot.png")
    plot_variance_sweep(results, plot_path, n_sys_params_ref or 0)
    print(f"\nSaved plot: {plot_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    run_sweep()