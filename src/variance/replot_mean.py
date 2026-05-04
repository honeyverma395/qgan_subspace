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
"""Plot mean Var[dW/dtheta_k] (averaged over ALL params: system + ancilla)
vs Mode and batch size, across multiple system sizes.
"""
import os
import re
import sys

# Same path setup as variance_analysis.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG  # not strictly used now, but kept for compatibility


# ---- EDIT HERE ------------------------------------------------------
# (n_qubits, batch_or_"choi").
RUNS: dict[tuple, str] = {
    (3, 1):      "3_1_ZZZ",
    (3, 5):      "3_5_ZZZ",
    (3, "choi"): "3_C_ZZZ",
    (4, 1):      "4_1_ZZZ",
    (4, 5):      "4_5_ZZZ",
    (4, "choi"): "4_C_ZZZ",
    (5, 1):      "5_1_ZZZ",
    (5, 5):      "5_5_ZZZ",
}

CONFIGS = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]

# Change the name to not overwrite the plot
OUT_NAME = "HamiltonianZZZ.png"     # saved at variance_analysis/

YLIM: tuple[float, float] | None = None     # (1e-4, 1e-1)
# ---------------------------------------------------------------------

COLOR_MAP = {
    "no_ancilla":         "blue",
    "ancilla_total":      "orange",
    "ancilla_bridge":     "green",
    "ancilla_shortBridge": "peru",
}


# -- Helpers ---------------------------------------------------------

def _mean_variance(grads_path: str) -> float:
    """Var per parameter (over samples), then mean over all parameters."""
    grads = np.load(grads_path)                  # (N_samples, n_params)
    var_per_param = np.var(grads, axis=0)        # (n_params,)
    return float(var_per_param.mean())


def _ordered_runs(runs: dict) -> list[tuple]:
    """Sort by n_qubits ascending; within each n, numeric batches ascending,
    then Choi at the end of that n group.
    """
    def sort_key(item):
        (n, b), _ = item
        # Numeric batches before Choi within the same n.
        if b == "choi":
            return (n, 1, 0)
        return (n, 0, b)
    return sorted(runs.items(), key=sort_key)


def _xlabel(key: tuple) -> str:
    """Two-line label: top = n=N, bottom = batch or Choi."""
    n, b = key
    bottom = "Choi" if b == "choi" else f"B={b}"
    return f"n={n}\n{bottom}" #Change n with L for layers comparison


def _parse_config_txt(path: str) -> dict:
    """Parse `key: value` and `key = value` lines from a saved config.txt."""
    info = {}
    if not os.path.exists(path):
        return info
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*([a-zA-Z_]\w*)\s*[:=]\s*(.+?),?\s*$", line)
            if m:
                info[m.group(1)] = m.group(2).strip()
    return info


def _format_hamiltonian_from_info(info: dict) -> str:
    """Short string for H using values parsed from config.txt."""
    target = info.get("target_hamiltonian", "?")
    n_samples = info.get("N_SAMPLES", "?")
    if target == "custom_h":
        terms = eval(info.get("custom_hamiltonian_terms", "[]"))
        strengths = eval(info.get("custom_hamiltonian_strengths", "[]"))
        body = " + ".join(f"{s:g}·{t_}" for s, t_ in zip(strengths, terms))
        return f"H = {body},  N = {n_samples}"
    return f"H = {target},  N = {n_samples}"


# -- Main --------------------------------------------------------------

def replot_mean_vs_batch() -> None:
    base_dir = os.path.join(_PROJECT_ROOT, "variance_analysis")

    # results[config] = list of (run_key, mean_var)
    results: dict[str, list[tuple]] = {c: [] for c in CONFIGS}
    # Per-run metadata parsed from config.txt
    run_info: dict = {}

    for run_key, timestamp in _ordered_runs(RUNS):
        run_dir = os.path.join(base_dir, timestamp)
        if not os.path.isdir(run_dir):
            print(f"[skip] {_xlabel(run_key)}: folder not found: {run_dir}")
            continue

        info = _parse_config_txt(os.path.join(run_dir, "config.txt"))
        run_info[run_key] = info

        for name in CONFIGS:
            grads_path = os.path.join(run_dir, f"grads_{name}.npy")
            if not os.path.exists(grads_path):
                print(f"[skip] {_xlabel(run_key)} / {name}: "
                      f"{grads_path} not found")
                continue

            mean_var = _mean_variance(grads_path)
            results[name].append((run_key, mean_var))
            label = _xlabel(run_key).replace("\n", " ")
            print(f"[ok]   {label:<12s}  {name:<22s}  "
                  f"mean Var = {mean_var:.3e}")

    # Drop configs with no data
    results = {k: v for k, v in results.items() if v}
    if not results:
        print("Nothing to plot.")
        return

    # -- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(run_info)), 5))

    run_keys = [k for k, _ in _ordered_runs(RUNS) if k in run_info]
    key_to_x = {k: i for i, k in enumerate(run_keys)}

    n_configs = len(results)
    spread = 0.5
    if n_configs > 1:
        offsets = np.linspace(-spread / 2, spread / 2, n_configs)
    else:
        offsets = np.array([0.0])
    config_offset = dict(zip(results.keys(), offsets))

    for name, pts in results.items():
        color = COLOR_MAP.get(name)
        dx = config_offset[name]
        xs = [key_to_x[k] + dx for k, _ in pts]
        ys = [v for _, v in pts]
        ax.scatter(xs, ys, color=color, s=55, label=name, zorder=3)

    # X-axis labels: "n=N\nB=K" or "n=N\nChoi"
    xtick_labels = [_xlabel(k) for k in run_keys]

    ax.set_yscale("log")
    ax.set_xticks(list(key_to_x.values()))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-0.5, len(run_keys) - 0.5)

    # Vertical dividers between different system sizes
    for i in range(1, len(run_keys)):
        if run_keys[i][0] != run_keys[i - 1][0]:
            ax.axvline(i - 0.5, color="black", linestyle=":",
                       linewidth=1.0, alpha=0.5)

    # -- Title: only show fields that are constant across runs ---------
    def _all_same(field: str) -> str | None:
        vals = {run_info[k].get(field) for k in run_keys}
        vals.discard(None)
        return next(iter(vals)) if len(vals) == 1 else None

    title_bits = ["Mean gradient variance"]
    extras = []
    layers = _all_same("gen_layers")
    ansatz = _all_same("gen_ansatz")
    if layers:
        extras.append(f"{layers} layers")
    if ansatz:
        extras.append(f"ansatz={ansatz}")
    if extras:
        title_bits.append("(" + ", ".join(extras) + ")")

    # H is constant only if target + terms + strengths + time + N_SAMPLES all match
    h_fields = ("target_hamiltonian", "custom_hamiltonian_terms",
                "custom_hamiltonian_strengths", "time_to_evolve", "N_SAMPLES")
    if all(_all_same(f) is not None for f in h_fields):
        ref_info = run_info[run_keys[0]]
        title_bits.append("\n" + _format_hamiltonian_from_info(ref_info))

    ax.set_title(" ".join(title_bits))
    ax.set_xlabel("run")
    ax.set_ylabel(r"mean Var$[\partial W / \partial \theta_k]$")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    if YLIM is not None:
        ax.set_ylim(YLIM)

    fig.tight_layout()
    out_path = os.path.join(base_dir, OUT_NAME)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    replot_mean_vs_batch()