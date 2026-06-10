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
"""Plot ROBUST gradient spread per parameter, averaged over all params,
vs Mode and batch size, across multiple system sizes.

Two estimators per config (same color, different marker):
  - square    : (1.4826 * MAD)^2          MAD-based variance estimate
  - triangle  : Trimmed Var (top/bottom TRIM_FRAC dropped)
  - circle    : Mean or Median 
  
Both are robust to outliers (heavy tails), unlike np.var.
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
from matplotlib.lines import Line2D
from config import CFG 
from qgan.generator import Generator
from variance.variance_analysis import (
    _apply_config,
    _snapshot_cfg,
    _restore_cfg,
    _split_system_ancilla,
)

# ---- EDIT HERE ------------------------------------------------------
# (n_qubits, batch_or_"choi") : timestamp folder
RUNS: dict[tuple, str] = {
    (3, 1):   "3_1_ZZZ",
    (3, 5):   "3_5_ZZZ",
    (3, "choi"):   "3_C_ZZZ",
    (4, 1):   "4_1_ZZZ",
    (4, 5):   "4_5_ZZZ",
    (4, "choi"):   "4_C_ZZZ",
    (5, 1):   "5_1_ZZZ",
    (5, 5):   "5_5_ZZZ",
}

FIRST_DIM = "n"   # "n" for system size, "L" for layers

CONFIGS = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]

# Change the name to not overwrite the plot
OUT_NAME = "HamiltonianZZZ.png" 
YLIM: tuple[float, float] | None = None    # (1e-4, 1e-1)
TRIM_FRAC = 0.0000000001     # trim top + bottom 1% of samples for trimmed Var

# Series toggles
SHOW_TOTAL  = False   # system+ancilla (pastel markers), if False, only system-only
SHOW_MAD    = True   # mean_k((1.4826·MAD_i)^2)  — square
SHOW_TVAR   = False   # mean_k(trimmed Var_i)     — triangle
SHOW_MEDIAN = False   # mean_k(|median_i(g_k)|)  — circle
SHOW_MEAN   = False   # mean_k(|mean_i(g_k)|)    — cross 
# ---------------------------------------------------------------------

COLOR_MAP = {
    "no_ancilla":         "blue",
    "ancilla_total":      "orange",
    "ancilla_bridge":     "green",
    "ancilla_shortBridge": "peru",
}

def _pastelize(color, factor: float = 0.5):
    """Lighten a matplotlib color toward white by the given factor (0..1)."""
    import matplotlib.colors as mcolors
    r, g, b, a = mcolors.to_rgba(color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b, a)

# Marker convention shared across all configs: square = MAD, triangle = trimmed
MARK_MAD = "s"
MARK_TVAR = "^"
MARK_MED = "o"
MARK_MEAN = "x"

# -- Robust estimators -----------------------------------------------
# -- Robust estimators (column-subset aware) -------------------------
def _mad_sigma2_mean(grads: np.ndarray) -> float:
    """Mean over params of (1.4826 * MAD_per_param)^2."""
    if grads.size == 0:
        return float("nan")
    med = np.median(grads, axis=0)
    mad = np.median(np.abs(grads - med), axis=0)
    sigma2 = (1.4826 * mad) ** 2
    return float(sigma2.mean())


def _trimmed_var_mean(grads: np.ndarray, frac: float) -> float:
    """Mean over params of trimmed variance (per param)."""
    if grads.size == 0:
        return float("nan")
    n = grads.shape[0]
    k = int(np.floor(n * frac))
    if k == 0:
        return float(np.var(grads, axis=0).mean())
    g_sorted = np.sort(grads, axis=0)
    g_trim = g_sorted[k : n - k]
    return float(np.var(g_trim, axis=0).mean())


def _abs_median_grad_mean(grads: np.ndarray) -> float:
    """Mean over params of |median_samples(∂W/∂θ_k)|."""
    if grads.size == 0:
        return float("nan")
    med_per_param = np.median(grads, axis=0)
    return float(np.abs(med_per_param).mean())

def _abs_mean_grad_mean(grads: np.ndarray) -> float:
    """Mean over params of |mean_samples(∂W/∂θ_k)|.

    Drift detector: if gradients are symmetric around 0 (expected under
    a random-circuit prior), this collapses as sqrt(Var/N) and should be
    much smaller than |median|. If it doesn't, that parameter has bias.
    """
    if grads.size == 0:
        return float("nan")
    mean_per_param = np.mean(grads, axis=0)
    return float(np.abs(mean_per_param).mean())

def _robust_estimates_subset(grads: np.ndarray) -> tuple[float, float, float]:
    """Return (mad_sigma2, trimmed_var, abs_median) for a given grads matrix."""
    return (
        _mad_sigma2_mean(grads),
        _trimmed_var_mean(grads, TRIM_FRAC),
        _abs_median_grad_mean(grads),
        _abs_mean_grad_mean(grads),
    )


def _robust_estimates_split(grads_path: str, config_name: str) -> dict:
    """Compute estimators on (a) total params and (b) system-only params.

    Returns:
        {
            "total":  (mad_s2, tvar, abs_med),
            "system": (mad_s2, tvar, abs_med),
        }
    """
    grads = np.load(grads_path)  # (N_samples, n_params)

    total = _robust_estimates_subset(grads)

    snapshot = _snapshot_cfg()
    try:
        _apply_config(config_name)
        gen = Generator()
        grads_sys, _ = _split_system_ancilla(grads, gen)
    finally:
        _restore_cfg(snapshot)

    system = _robust_estimates_subset(grads_sys)
    return {"total": total, "system": system}
def _normalize_key(key: tuple, info: dict | None = None) -> tuple:
    """Normalize a RUNS key to (n_qubits, batch_or_'choi', mode).

    - 3-tuple: returned as-is.
    - 2-tuple: mode is read from config.txt's batch_mode field if available,
      else defaults to "haar" for batch runs and "choi" for Choi runs.
    """
    if len(key) == 3:
        return key
    if len(key) != 2:
        raise ValueError(f"RUNS key must be a 2- or 3-tuple, got {key!r}")
    n, b = key
    if b == "choi":
        return (n, b, "choi")
    if info is not None and "batch_mode" in info:
        # info values are raw strings; strip surrounding quotes if literal_eval
        # wasn't applied (we only do raw regex parsing in this script).
        mode = info["batch_mode"].strip().strip("'\"")
    else:
        mode = "haar"
    return (n, b, mode)


# Sort priority: mode "comp_basis" before "haar" before "choi" within an (n,b).
_MODE_SORT = {"comp_basis": 0, "haar": 1, "choi": 2}


def _ordered_runs(runs: dict) -> list[tuple]:
    """Sort by n_qubits ascending; within each n, numeric batches ascending,
    then Choi at the end; ties on (n,b) broken by mode.

    Note: this sorts on the *raw* key (not the normalized key) since we don't
    have config.txt info here. 3-tuple keys participate in the mode tiebreak
    naturally; 2-tuple keys are treated as Haar for sort order.
    """
    def sort_key(item):
        key, _ = item
        if len(key) == 3:
            n, b, mode = key
        else:
            n, b = key
            mode = "choi" if b == "choi" else "haar"
        if b == "choi":
            return (n, 1, 0, _MODE_SORT.get(mode, 99))
        return (n, 0, b, _MODE_SORT.get(mode, 99))
    return sorted(runs.items(), key=sort_key)


def _xlabel(key: tuple, info: dict | None = None) -> str:
    """Two-line label: top = n=N, bottom = batch or Choi (with mode if known).

    For batch runs, mode is shown as a suffix only when it's not the default
    "haar" — keeps labels short for the common case.
    """
    n, b, mode = _normalize_key(key, info)
    if b == "choi":
        bottom = "Choi"
    elif mode == "haar":
        bottom = f"B={b}"
    else:
        # Surface non-Haar modes explicitly (e.g. comp_basis)
        bottom = f"B={b}\n{mode}"
    return f"{FIRST_DIM}={n}\n{bottom}"

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
def replot_mean_vs_batch_robust() -> None:
    base_dir = os.path.join(_PROJECT_ROOT, "variance_analysis")

    # results[config] = list of (run_key, mad_sigma2, trimmed_var)
    results: dict[str, list[tuple]] = {c: [] for c in CONFIGS}
    run_info: dict = {}

    for run_key, timestamp in _ordered_runs(RUNS):
        run_dir = os.path.join(base_dir, timestamp)
        if not os.path.isdir(run_dir):
            # info not yet read; label without mode hint is fine here
            print(f"[skip] {_xlabel(run_key)}: folder not found: {run_dir}")
            continue
        info = _parse_config_txt(os.path.join(run_dir, "config.txt"))
        run_info[run_key] = info

        for name in CONFIGS:
            grads_path = os.path.join(run_dir, f"grads_{name}.npy")
            if not os.path.exists(grads_path):
                print(f"[skip] {_xlabel(run_key, info)} / {name}: "
                      f"{grads_path} not found")
                continue

            ests = _robust_estimates_split(grads_path, name)
            results[name].append((run_key, ests))

            label = _xlabel(run_key, info).replace("\n", " ")
            mt = ests["total"]
            ms = ests["system"]
            print(f"[ok]   {label:<14s}  {name:<22s}  "
                f"[total]  MAD-σ²={mt[0]:.3e}  TVar={mt[1]:.3e}  "
                f"|med|={mt[2]:.3e}  |mean|={mt[3]:.3e}  "
                f"[sys]    MAD-σ²={ms[0]:.3e}  TVar={ms[1]:.3e}  "
                f"|med|={ms[2]:.3e}  |mean|={ms[3]:.3e}")

    results = {k: v for k, v in results.items() if v}
    if not results:
        print("Nothing to plot.")
        return

    # -- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(run_info) + 6), 5.5))

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
        color_total = _pastelize(color, factor=0.55)  # softer for "total"
        dx = config_offset[name]

        xs = [key_to_x[k] + dx for k, _ in pts]

        # System-only series
        ys_mad_sys  = [e["system"][0] for _, e in pts]
        ys_tvar_sys = [e["system"][1] for _, e in pts]
        ys_med_sys  = [e["system"][2] for _, e in pts]
        ys_mean_sys = [e["system"][3] for _, e in pts]

        # Total = system + ancilla
        ys_mad_tot  = [e["total"][0] for _, e in pts]
        ys_tvar_tot = [e["total"][1] for _, e in pts]
        ys_med_tot  = [e["total"][2] for _, e in pts]
        ys_mean_tot = [e["total"][3] for _, e in pts]

        # -- Total (pastel, drawn first so it sits "behind") --
        if SHOW_TOTAL:
            if SHOW_MAD:
                ax.scatter(xs, ys_mad_tot,  color=color_total, marker=MARK_MAD,  s=55,
                           zorder=2, label=f"{name} (MAD-σ², total)")
            if SHOW_TVAR:
                ax.scatter(xs, ys_tvar_tot, color=color_total, marker=MARK_TVAR, s=70,
                           zorder=2, edgecolors="gray", linewidths=0.4,
                           label=f"{name} (trimmed Var, total)")
                
            if SHOW_MEDIAN:
                ax.scatter(xs, ys_med_tot, color=color_total, marker=MARK_MED, s=55,
                           zorder=2, facecolors="none", edgecolors=color_total,
                           linewidths=1.5,
                           label=f"{name} (mean |median|, total)")
            if SHOW_MEAN:
                ax.scatter(xs, ys_mean_tot, color=color_total, marker=MARK_MEAN, s=60,
                           zorder=2, linewidths=1.8,
                           label=f"{name} (mean |mean|, total)")

        # -- System-only (saturated, on top) --
        if SHOW_MAD:
            ax.scatter(xs, ys_mad_sys,  color=color, marker=MARK_MAD,  s=55,
                       zorder=3, label=f"{name} (MAD-σ², system)")
        if SHOW_TVAR:
            ax.scatter(xs, ys_tvar_sys, color=color, marker=MARK_TVAR, s=70,
                       zorder=3, edgecolors="black", linewidths=0.4,
                       label=f"{name} (trimmed Var, system)")
        if SHOW_MEDIAN:
            ax.scatter(xs, ys_med_sys, color=color, marker=MARK_MED, s=55,
                       zorder=3, facecolors="none", edgecolors=color,
                       linewidths=1.5,
                       label=f"{name} (mean |median|, system)")
        if SHOW_MEAN:
            ax.scatter(xs, ys_mean_sys, color=color, marker=MARK_MEAN, s=60,
                       zorder=3, linewidths=1.8,
                       label=f"{name} (mean |mean|, system)")

    # X-axis labels
    xtick_labels = [_xlabel(k, run_info.get(k)) for k in run_keys]
    ax.set_yscale("log")
    ax.set_xticks(list(key_to_x.values()))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-0.5, len(run_keys) - 0.5)

    # Vertical dividers between different system sizes
    for i in range(1, len(run_keys)):
        if run_keys[i][0] != run_keys[i - 1][0]:
            ax.axvline(i - 0.5, color="black", linestyle=":",
                       linewidth=1.0, alpha=0.5)

    # -- Title: only show fields that are constant across runs --------
    def _all_same(field: str) -> str | None:
        vals = {run_info[k].get(field) for k in run_keys}
        vals.discard(None)
        return next(iter(vals)) if len(vals) == 1 else None

    title_bits = [r"Gradient spread"]
    extras = []
    layers = _all_same("gen_layers")
    ansatz = _all_same("gen_ansatz")
    if layers:
        extras.append(f"{layers} layers")
    if ansatz:
        extras.append(f"ansatz={ansatz}")
    if extras:
        title_bits.append("(" + ", ".join(extras) + ")")

    h_fields = ("target_hamiltonian", "custom_hamiltonian_terms",
                "custom_hamiltonian_strengths", "time_to_evolve", "N_SAMPLES")
    if all(_all_same(f) is not None for f in h_fields):
        ref_info = run_info[run_keys[0]]
        title_bits.append("\n" + _format_hamiltonian_from_info(ref_info))

    anc_train = _all_same("ancilla_training")
    if anc_train is not None:
        title_bits.append(
            f"  ancilla_training={anc_train}"
        )
    ax.set_title(" ".join(title_bits))

    ax.set_xlabel("run")
    ax.set_ylabel("mean robust spread / |median| per parameter")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    config_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_MAP[c],
               markersize=8, label=c)
        for c in results.keys()
    ]
    legend1 = ax.legend(handles=config_handles, title="Config",
                        loc="upper left", bbox_to_anchor=(1.05, 1.0),
                        fontsize=8, framealpha=0.9, borderaxespad=0.0)
    ax.add_artist(legend1)

    marker_handles = []
    if SHOW_MAD:
        marker_handles.append(
            Line2D([0], [0], marker=MARK_MAD, color="gray", linestyle="None",
                   markersize=8, label="MAD-σ²")
        )
    if SHOW_TVAR:
        marker_handles.append(
            Line2D([0], [0], marker=MARK_TVAR, color="gray", linestyle="None",
                   markersize=9, markeredgecolor="black", label="Trimmed Var")
        )
    if SHOW_MEDIAN:
        marker_handles.append(
            Line2D([0], [0], marker=MARK_MED, color="gray", linestyle="None",
                   markersize=8, markerfacecolor="none", markeredgecolor="gray",
                   label="mean |median|")
        )
    if SHOW_MEAN:
        marker_handles.append(
            Line2D([0], [0], marker=MARK_MEAN, color="gray", linestyle="None",
                   markersize=9, label="mean |mean|")
        )
    marker_handles.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=8, label="system only")
    )
    ax.legend(handles=marker_handles, title="Marker / shade",
              loc="upper left", bbox_to_anchor=(1.05, 0.55),
              fontsize=8, framealpha=0.9, borderaxespad=0.0)
    
    if YLIM is not None:
        ax.set_ylim(YLIM)

    fig.tight_layout(rect=(0, 0, 0.85, 1))    
    out_path = os.path.join(base_dir, OUT_NAME)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    replot_mean_vs_batch_robust() 