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
theta, Uniform[0, 2pi) together with a fresh random discriminator
(alpha, beta, Uniform[-1, 1)), compute the gradient of the Wasserstein
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
)
from qgan.target import get_target_operator


# -- Script parameters -------------------------------------------------
N_SAMPLES = 100000
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
def sample_gradients(n_samples: int) -> np.ndarray:
    """For the current CFG, draw n_samples random theta and compute grad L.

    Returns:
        np.ndarray of shape (n_samples, n_params).
    """
    # Fresh generator so count_params, wire layout, QNode all match CFG.
    gen = Generator()
    target_state = _build_target_state()
    n_params = gen.n_params

    grads = np.zeros((n_samples, n_params), dtype=np.float64)

    for i in range(n_samples):
        # Random theta [0, 2 \pi)
        with torch.no_grad():
            gen.params.copy_(
                torch.empty(n_params, dtype=torch.float32).uniform_(0, 2 * np.pi)
            )
        # Fresh random discriminator (alpha, beta [-1, 1) via its __init__).
        # We rebuild it each sample so (alpha, beta) are random per sample;
        # its optimizer is never used here.
        dis = Discriminator()

        if gen.params.grad is not None:
            gen.params.grad.zero_()
        loss = _wasserstein_loss(gen, dis, target_state)
        loss.backward()
        grads[i] = gen.params.grad.detach().numpy().copy()

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"    sample {i + 1}/{n_samples}")

    return grads


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
def plot_variance_sweep(results: dict, out_path: str, n_sys_params: int) -> None:
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
    ax.set_title(
        f"Wasserstein-gradient variance per parameter   "
        f"(N={N_SAMPLES} samples, {CFG.system_size} qubits, "
        f"{CFG.gen_layers} layers, ansatz={CFG.gen_ansatz})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# -- Main --------------------------------------------------------------
def run_sweep() -> None:
    if SEED is not None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    # Sanity Check
    if not CFG.use_choi:
        raise RuntimeError(
            "variance_analysis.py currently assumes CFG.use_choi = True. "
            "Set it in config.py before running."
        )

    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    out_dir = os.path.join(
        _PROJECT_ROOT,
        "variance_analysis",
        timestamp,
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    snapshot = _snapshot_cfg()
    configs = ["no_ancilla", "ancilla_total", "ancilla_bridge", "ancilla_shortBridge"]
    results: dict[str, dict[str, np.ndarray]] = {}
    n_sys_params_ref = None

    try:
        for name in configs:
            print(f"\n=== Config: {name} ===")
            _apply_config(name)

            # We need a generator just to access _get_ancilla_param_indices
            # and n_params for splitting. We also reuse it inside
            # sample_gradients, but that one is built fresh there.
            probe_gen = Generator()
            grads = sample_gradients(N_SAMPLES)

            # Split into system / ancilla-only columns
            grads_sys, grads_anc = _split_system_ancilla(grads, probe_gen)
            var_sys = np.var(grads_sys, axis=0)
            var_anc = np.var(grads_anc, axis=0) if grads_anc.size > 0 else np.array([])

            # Save
            np.save(os.path.join(out_dir, f"grads_{name}.npy"), grads)
            np.save(os.path.join(out_dir, f"variance_{name}.npy"),
                    np.concatenate([var_sys, var_anc]))

            results[name] = {"var_sys": var_sys, "var_anc": var_anc}

            # First config sets the reference for # of system params
            # Same across the three configs since
            # system_size and gen_layers don't change
            if n_sys_params_ref is None:
                n_sys_params_ref = len(var_sys)
            elif len(var_sys) != n_sys_params_ref:
                print(
                    f"  WARNING: system param count changed across configs "
                    f"({n_sys_params_ref} vs {len(var_sys)}). "
                    f"Plot alignment may be off."
                )

            print(f"  n_params = {grads.shape[1]}  "
                  f"(system = {len(var_sys)}, ancilla-only = {len(var_anc)})")
            print(f"  mean Var(system)     = {var_sys.mean():.3e}")
            if var_anc.size > 0:
                print(f"  mean Var(ancilla)    = {var_anc.mean():.3e}")

    finally:
        _restore_cfg(snapshot)

    # -- Plot -----------------------------------------------------------
    plot_path = os.path.join(out_dir, "variance_plot.png")
    plot_variance_sweep(results, plot_path, n_sys_params_ref or 0)
    print(f"\nSaved plot: {plot_path}")

    # -- Config snapshot -----------------------------------------------
    with open(os.path.join(out_dir, "config.txt"), "w") as fh:
        fh.write(f"N_SAMPLES = {N_SAMPLES}\n")
        fh.write(f"SEED = {SEED}\n")
        fh.write(f"configs = {configs}\n\n")
        fh.write(CFG.show_data())

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    run_sweep()