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
"""Discriminator module — PyTorch-PennyLane version.

The discriminator is NOT a quantum circuit, it is a classical parametrisation
of two Hermitian operators (psi, phi) built from tensor products of Pauli matrices.

Changes from PennyLane/numpy version:
    - alpha and beta are now torch.nn.Parameter -> autograd handles ALL gradients.
    - Manual gradient methods (_compute_grad, _grad_alpha, _grad_beta,
      _grad_psi_or_phi) are REMOVED entirely.
    - Momentum SGD via torch.optim.SGD(momentum=...).
    - Loss computed via torch.vdot()
    - save/load preserved, adapted for torch state_dict.

The discriminator MAXIMISES the Wasserstein cost:
    Loss = psi_term − phi_term − reg_term
We put a minus sign and call optimizer.step() to minimise (−Loss).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from config import CFG
from tools.data_managers import print_and_log
from qgan.cost_functions import _calc_wasserstein

# -- PAULI MATRICES ---------------------------------
I = torch.eye(2, dtype=torch.complex64)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)


PAULIS = [I, X, Y, Z]

# -- DISCRIMINATOR ---------------------------------
# Wasserstein cost constants from config
cst1, cst2, cst3, lamb = CFG.cst1, CFG.cst2, CFG.cst3, CFG.lamb


class Discriminator(nn.Module):
    """Discriminator for the Quantum Wasserstein GAN

    Parameters alpha, beta are nn.Parameters. The full forward pass
    (alpha,beta -> psi, phi -> A,B -> loss) is differentiable via autograd.

    Usage:
        dis = Discriminator()
        loss = dis.compute_loss(final_target_state, final_gen_state)
        loss.backward()
        dis.optimizer.step()
        dis.optimizer.zero_grad()
    """

    def __init__(self):
        super().__init__()

        # Total number of qubits the discriminator acts on
        # Since we are applying the measurement to each qubit, the size changes
        # if we are using Choi method (2* system size) or Haar Batching (system size)
        self.size: int = (
            CFG.system_size * (2 if CFG.use_choi else 1)
            + (1 if CFG.extra_ancilla and CFG.ancilla_mode == "pass" else 0)
            )

        # alpha and beta as trainable parameters (real-valued)
        self.alpha = nn.Parameter(
            -1 + 2 * torch.rand(self.size, 4, dtype=torch.float32)
        )
        self.beta = nn.Parameter(
            -1 + 2 * torch.rand(self.size, 4, dtype=torch.float32)
        )

        # Optimizer: SGD with momentum, MAXIMISING
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=CFG.l_rate,
            momentum=CFG.momentum_coeff,
        )

        # Save/load compatibility
        self.ancilla: bool = CFG.extra_ancilla
        self.ancilla_mode: str = CFG.ancilla_mode
        self.target_size: int = CFG.system_size
        self.target_hamiltonian: str = CFG.target_hamiltonian
        # -- Gradient history (trajectory variance) --
        self.grad_history: list[np.ndarray] = [] 

    # -- matrix representations ---------------------------------
    def get_psi_and_phi(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build psi and phi matrices via Kronecker product of per-qubit Hermitians.

        For each qubit i:
            H_i = \sum_j coeff[i][j] · \sigma_j    (real coeffs, Hermitian result)

        Full operator: psi = H_0 \otimes H_1 \otimes ... \otimes H_{N-1}

        Returns:
            (\psi, \phi) each of shape (2^N, 2^N), on the autograd graph.
        """
        # Cast alpha, beta for matrix multiplication
        alpha_c = self.alpha.to(torch.complex64)
        beta_c = self.beta.to(torch.complex64)

        psi = torch.tensor([[1.0]], dtype=torch.complex64)  # scalar 1 as 1×1 matrix
        phi = torch.tensor([[1.0]], dtype=torch.complex64)

        for i in range(self.size):
            # Per-qubit Hermitian
            psi_i = sum(alpha_c[i, j] * PAULIS[j] for j in range(4))
            phi_i = sum(beta_c[i, j] * PAULIS[j] for j in range(4))
            psi = torch.kron(psi, psi_i)
            phi = torch.kron(phi, phi_i)

        return psi, phi

    def get_dis_matrices_rep(self) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor, torch.Tensor]:
        """Compute A = exp(−phi / lambda) and B = exp(+\psi / lambda).

        Returns:
            (A, B, psi, phi)
        """
        psi, phi = self.get_psi_and_phi()
        A = torch.linalg.matrix_exp((-1.0 / lamb) * phi)
        B = torch.linalg.matrix_exp((1.0 / lamb) * psi)
        return A, B, psi, phi

    # -- loss computation ---------------------------------
    def compute_loss(self, final_target_state: torch.Tensor,
                     final_gen_state: torch.Tensor) -> torch.Tensor:
        """Compute the Wasserstein loss as a differentiable scalar."""
        dis_matrices = self.get_dis_matrices_rep()

        g = final_gen_state.reshape(-1)
        t = final_target_state.reshape(-1)

        return -(_calc_wasserstein(g, t, dis_matrices))
        
    # -- save / load ---------------------------------
    def save_model(self, file_path: str):
        """Save discriminator state to disk.

        Saves both the nn.Module state_dict (alpha, beta) and metadata
        needed for compatibility checks.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_dict = {
            "state_dict": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "size": self.size,
            "ancilla": self.ancilla,
            "ancilla_mode": self.ancilla_mode,
            "target_size": self.target_size,
            "target_hamiltonian": self.target_hamiltonian,
        }
        torch.save(save_dict, file_path)

    def load_model_params(self, file_path: str) -> bool:
        """Load discriminator parameters from a saved model (torch format only).
        """
        if not os.path.exists(file_path):
            print_and_log("Discriminator model file not found\n", CFG.log_path)
            return False

        try:
            saved = torch.load(file_path, weights_only=False)
        except Exception as e:
            print_and_log(
                f"ERROR: Could not load discriminator model: {e}\n"
                "Old pickle/numpy formats are no longer supported.\n",
                CFG.log_path,
            )
            return False

        if not isinstance(saved, dict) or "state_dict" not in saved:
            print_and_log(
                "ERROR: Unrecognised format. Only torch dict format is supported.\n",
                CFG.log_path,
            )
            return False

        return self._load_from_torch_format(saved)
    
    def _load_from_torch_format(self, saved: dict) -> bool:
        """Load from new torch save format."""
        # Compatibility checks
        if saved.get("target_size") != self.target_size:
            print_and_log("ERROR: target size mismatch.\n", CFG.log_path)
            return False
        if saved.get("target_hamiltonian") != self.target_hamiltonian:
            print_and_log("ERROR: target hamiltonian mismatch.\n", CFG.log_path)
            return False

        if saved.get("size") == self.size:
            self.load_state_dict(saved["state_dict"])
            if "optimizer_state" in saved:
                self.optimizer.load_state_dict(saved["optimizer_state"])
            print_and_log("Discriminator parameters loaded (torch format).\n", CFG.log_path)
            return True

        # \pm1 qubit (ancilla difference)
        if abs(saved.get("size", 0) - self.size) == 1:
            saved_alpha = saved["state_dict"]["alpha"]
            saved_beta = saved["state_dict"]["beta"]
            min_size = min(saved_alpha.shape[0], self.size)
            with torch.no_grad():
                self.alpha[:min_size] = saved_alpha[:min_size].clone()
                self.beta[:min_size] = saved_beta[:min_size].clone()
            print_and_log(
                "Discriminator parameters partially loaded (\pm1 qubit, torch).\n",
                CFG.log_path,
            )
            return True

        print_and_log("ERROR: incompatible discriminator (size mismatch).\n", CFG.log_path)
        return False

