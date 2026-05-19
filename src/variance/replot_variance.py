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
"""Replot variance_analysis from saved grads_{config}.npy files, plus
gradient-distribution diagnostics.

Reads config.txt from the saved run, then for each config:
  - reproduces the variance sweep plot (plot_variance_sweep)
  - prints mean / median / MAD and estimator summaries on |grad|
  - saves a log-scale histogram of |grad|     -> hist_{TIMESTAMP}.png
  - saves Var vs Trimmed-Var vs (1.4826 MAD)^2 -> estimators_{TIMESTAMP}.png
  - saves the gradient median per parameter
  - saves the gradient norm per sample (boxplot, two panels: all params and
    system-only) -> gradient_norm_{TIMESTAMP}.png
  - saves the empirical GOP eigenvalue spectrum (raw and trace-normalized)
    both on the full parameter space and restricted to the system block
  - saves the GOP analysis stratified by ||grad||^2 deciles, both on the full
    parameter space (stratified by ||g||^2) and restricted to the system block
    (stratified by ||g_sys||^2)
  - runs a KS two-sample test on |grad| restricted to the system block
"""

import os
import re
import sys
import ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import CFG
from qgan.generator import Generator
from variance import variance_analysis
from variance.variance_analysis import (
    _apply_config,
    _snapshot_cfg,
    _restore_cfg,
    _split_system_ancilla,
    plot_variance_sweep,
)

# ---- EDIT HERE ------------------------------------------------------
TIMESTAMP = "3_1_XXXZZZ"
CONFIGS = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]
OUT_NAME = "variance_plot.png"
XLIM: tuple[float, float] | None = None
YLIM: tuple[float, float] | None = None  # (1e-4, 1e-1)

# Diagnostics knobs
RUN_DIAGNOSTICS = True
RUN_NORM_DIST = True
RUN_MEAN_PARAM = True
RUN_GOP = True
RUN_GOP_SYS = True        # GOP restricted to the system block
RUN_DECILES = True
RUN_DECILES_SYS = True    # Deciles on the system block 
TRIM_FRAC = 0.01                              # trim 1% top + 1% bottom
N_DECILES = 10
RUN_KS = True
KS_PAIRS = [
    ("no_ancilla",    "ancilla_total"),
    ("no_ancilla", "ancilla_bridge"),
    ("no_ancilla", "ancilla_shortBridge"),
    ("ancilla_total", "ancilla_shortBridge"),
    ("ancilla_total", "ancilla_bridge"),
    ("ancilla_bridge", "ancilla_shortBridge"),
]
# ---------------------------------------------------------------------

REQUIRED_CFG_FIELDS = (
    "N_SAMPLES",
    "system_size",
    "gen_layers",
    "gen_ansatz",
    "extra_ancilla",
    "ancilla_topology",
    "ancilla_connect_to",
    "do_ancilla_1q_gates",
    "use_choi",
    "batch_size",
    "target_hamiltonian",
    "custom_hamiltonian_strengths",
    "custom_hamiltonian_terms",
)

COLOR_MAP = {
    "no_ancilla":          "blue",
    "ancilla_total":       "orange",
    "ancilla_bridge":      "green",
    "ancilla_shortBridge": "peru",
}

# -- Duplicate stdout to a file ----------------------------------
class _write_and_print:
    """Write to multiple streams at once. Used to mirror stdout to a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# -- CFG loader ----------------------------------
def _load_cfg_from_file(out_dir: str) -> None:
    config_path = os.path.join(out_dir, "config.txt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.txt not found in {out_dir}")

    with open(config_path) as f:
        text = f.read()

    loaded = {}
    for line in text.splitlines():
        # Accept either "key: value" (CFG fields) or "key = value" (N_SAMPLES, SEED)
        m = re.match(r"^\s*(\w+)\s*[:=]\s*(.+)$", line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()

        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw  # treat as plain string

        if hasattr(CFG, key):
            setattr(CFG, key, value)
            loaded[key] = raw
        elif hasattr(variance_analysis, key):
            setattr(variance_analysis, key, value)
            loaded[key] = raw

    missing = [f for f in REQUIRED_CFG_FIELDS if f not in loaded]
    if missing:
        raise RuntimeError(
            f"config.txt is missing critical fields: {missing}\n"
            f"Cannot replot safely without them."
        )

    print("CFG loaded from config.txt:")
    for k, v in loaded.items():
        print(f"  {k} = {v}")


# -- Diagnostics helpers ----------------------------------
def _trimmed_var(x: np.ndarray, frac: float) -> np.ndarray:
    """Trimmed variance along axis 0: drop top and bottom `frac` of samples."""
    n = x.shape[0]
    k = int(np.floor(n * frac))
    if k == 0:
        return np.var(x, axis=0)
    x_sorted = np.sort(x, axis=0)
    x_trim = x_sorted[k : n - k]
    return np.var(x_trim, axis=0)


def _mad_sigma2(x: np.ndarray) -> np.ndarray:
    """MAD-based estimator of variance, per column.

    sigma_hat = 1.4826 * MAD; returns sigma_hat^2 to compare with Var.
    """
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return (1.4826 * mad) ** 2

def _ks_test_pairs(
    grads_by_config: dict[str, np.ndarray],
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    pairs: list[tuple[str, str]],
) -> None:
    """KS two-sample test on |grad| over system params only.

    With huge sample sizes the p-value collapses to 0 even for tiny
    differences, so the useful number is the KS statistic D (sup |F1 - F2|,
    in [0, 1]). p-value is reported as a sanity check.
    """
    print("\n--- KS two-sample test on |grad|  (system params only) ---")
    print(f"  {'pair':<45} {'D':<10} {'p-value':<12} {'n1':<8} {'n2':<8}")
    for a, b in pairs:
        if a not in grads_by_config or b not in grads_by_config:
            print(f"  [skip] {a} vs {b}: missing grads")
            continue
        sys_a = splits[a][0]
        sys_b = splits[b][0]
        # |grad| restricted to system params, flattened across samples
        x = np.abs(grads_by_config[a][:, sys_a]).ravel()
        y = np.abs(grads_by_config[b][:, sys_b]).ravel()
        stat = ks_2samp(x, y, alternative="two-sided", mode="auto")
        label = f"{a} vs {b}"
        print(f"  {label:<45} {stat.statistic:<10.4f} "
              f"{stat.pvalue:<12.3e} {x.size:<8d} {y.size:<8d}")

def _print_diagnostics(name: str, grads: np.ndarray) -> dict:
    """Print summary statistics and estimator means for one config."""
    abs_g = np.abs(grads).ravel()
    med = np.median(abs_g)
    mean = np.mean(abs_g)
    mad = np.median(np.abs(abs_g - med))

    print(f"\n--- {name} ---")
    print(f"  shape           = {grads.shape}  (n_samples, n_params)")
    print(f"  |grad|  mean    = {mean:.3e}")
    print(f"  |grad|  median  = {med:.3e}")
    print(f"  |grad|  MAD     = {mad:.3e}")

    var_est = np.var(grads, axis=0)
    tvar_est = _trimmed_var(grads, TRIM_FRAC)
    mad_est = _mad_sigma2(grads)

    print(f"  estimator means across params:")
    print(f"    Var(grad)              = {var_est.mean():.3e}")
    print(f"    Trimmed Var ({TRIM_FRAC:.0%}/side)  = {tvar_est.mean():.3e}")
    print(f"    (1.4826*MAD)^2         = {mad_est.mean():.3e}")
    print(f"  ratio Var / MAD-sigma^2  = "
          f"{var_est.mean() / max(mad_est.mean(), 1e-300):.3e}")

    return {
        "abs_g": abs_g,
        "var": var_est,
        "tvar": tvar_est,
        "mad_sigma2": mad_est,
    }


def _plot_histograms(
    diags: dict,
    grads_by_config: dict,
    splits: dict,
    out_path: str,
) -> None:
    """Log-scale histogram of |grad| per config, overlaid.

    Two panels:
      - left:  all parameters (system + ancilla)
      - right: system parameters only (ancilla block filtered out)
    Shared bin edges and y-axis range across panels so the two are
    directly comparable.
    """
    fig, (ax_all, ax_sys) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Build per-config arrays for each panel up-front so we can compute
    # shared log-spaced bins from the union.
    vals_all_by_name = {
        name: d["abs_g"][d["abs_g"] > 0]
        for name, d in diags.items()
    }
    vals_sys_by_name = {}
    for name in diags.keys():
        if name not in grads_by_config or name not in splits:
            continue
        sys_idx = splits[name][0]
        v = np.abs(grads_by_config[name][:, sys_idx]).ravel()
        v = v[v > 0]
        vals_sys_by_name[name] = v

    all_vals = (np.concatenate(list(vals_all_by_name.values()))
                if vals_all_by_name else np.array([]))
    if all_vals.size == 0:
        print("All gradients are zero - skipping histogram.")
        plt.close(fig)
        return
    lo, hi = all_vals.min(), all_vals.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)

    # Left: all params
    for name, vals in vals_all_by_name.items():
        color = COLOR_MAP.get(name)
        ax_all.hist(vals, bins=bins, histtype="step", linewidth=1.5,
                    color=color, label=name, log=True)

    # Right: system params only
    for name, vals in vals_sys_by_name.items():
        color = COLOR_MAP.get(name)
        ax_sys.hist(vals, bins=bins, histtype="step", linewidth=1.5,
                    color=color, label=name, log=True)

    for ax, xlabel in (
        (ax_all, r"$|\partial W / \partial \theta_k|$  (all params, all samples)"),
        (ax_sys, r"$|\partial W / \partial \theta_k|$  (system params only)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    ax_all.set_ylabel("count (log)")
    ax_all.set_title("All parameters")
    ax_sys.set_title("System parameters only")

    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"
    fig.suptitle(
        f"Gradient magnitude distribution  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str})",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved histogram: {out_path}")

def _plot_estimator_comparison(
    diags: dict,
    splits: dict,
    out_path: str,
    n_sys_params_ref: int,
) -> None:
    """Per config: scatter Var / Trimmed-Var / MAD-sigma^2 vs param index."""
    n = len(diags)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, diags.items()):
        sys_idx, anc_idx = splits[name]
        order = np.concatenate([sys_idx, anc_idx]).astype(int)
        var_o = d["var"][order]
        tvar_o = d["tvar"][order]
        mad_o = d["mad_sigma2"][order]
        x = np.arange(var_o.size)

        ax.scatter(x, var_o,  marker="o", s=22, alpha=0.7,
                   color="tab:red",   label="Var")
        ax.scatter(x, tvar_o, marker="s", s=22, alpha=0.7,
                   color="tab:blue",  label=f"Trimmed Var ({TRIM_FRAC:.0%})")
        ax.scatter(x, mad_o,  marker="^", s=22, alpha=0.7,
                   color="tab:green", label=r"$(1.4826\,\mathrm{MAD})^2$")

        if anc_idx.size > 0:
            ax.axvline(n_sys_params_ref - 0.5, color="black",
                       linestyle=":", linewidth=1.0, alpha=0.6)

        ax.set_yscale("log")
        ax.set_ylabel("estimate")
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=3)

    axes[-1].set_xlabel("parameter index k  (system block | ancilla block)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved estimator comparison: {out_path}")


# -- Norm-per-sample distribution ----------------------------------------
def _plot_norm_distribution(
    grads_by_config: dict,
    splits: dict,
    out_path: str,
) -> None:
    """Boxplot of ||grad W||_2 per sample, one boxplot per config (log y-axis).

    Two panels, mirroring the histogram plot:
      - left:  ||grad||_2 over all parameters (system + ancilla)
      - right: ||grad_sys||_2 over system parameters only

    Each sample i contributes ||grad W^(i)||_2 = sqrt(sum_k (dW^(i)/dtheta_k)^2).
    Shows the typical step magnitude SGD would take from a random point in
    the joint landscape. The system-only panel is the directly comparable
    quantity across configs: same dimension n_sys for all of them.
    """
    fig, (ax_all, ax_sys) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    names = list(grads_by_config.keys())

    # Build both norm series up-front so the y-axis (shared) is sane
    norms_all_per_config = []
    norms_sys_per_config = []
    for name in names:
        grads = grads_by_config[name]
        # All-params norm
        n_all = np.sqrt(np.sum(grads ** 2, axis=1))
        n_all = n_all[n_all > 0]
        norms_all_per_config.append(n_all)
        # System-only norm
        if name in splits:
            sys_idx = splits[name][0]
            n_sys = np.sqrt(np.sum(grads[:, sys_idx] ** 2, axis=1))
            n_sys = n_sys[n_sys > 0]
        else:
            n_sys = np.array([])
        norms_sys_per_config.append(n_sys)

    def _draw_panel(ax, norms_per_config, title, ylabel):
        parts = ax.boxplot(
            norms_per_config,
            showfliers=True,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.3},
            widths=0.6,
        )
        for patch, name in zip(parts["boxes"], names):
            c = COLOR_MAP.get(name, "gray")
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
            patch.set_edgecolor("black")
        # Annotate medians next to each box
        for i, nn in enumerate(norms_per_config):
            if nn.size == 0:
                continue
            med = np.median(nn)
            ax.text(i + 1.05, med, f"med={med:.2e}",
                    va="center", ha="left", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3, axis="y")

    _draw_panel(
        ax_all, norms_all_per_config,
        title="All parameters",
        ylabel=r"$\|\nabla W\|_2$  (per sample)",
    )
    _draw_panel(
        ax_sys, norms_sys_per_config,
        title="System parameters only",
        ylabel=r"$\|\nabla W_\mathrm{sys}\|_2$  (per sample)",
    )

    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"

    fig.suptitle(
        f"Gradient norm per sample  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str})",
        y=1.00,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved norm distribution: {out_path}")


# -- Median-per-parameter plot ------------------------------------------
def _plot_mean_per_param(
    results_mean: dict,
    out_path: str,
    n_sys_params: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Companion to variance_plot.png: |<dW/dtheta_k>| per parameter, per config.

    Args:
        results_mean: {config_name: {"median_sys": np.ndarray, "median_anc": np.ndarray}}
                      where each value is |median[dW/dtheta_k]| (already abs-valued).
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, data in results_mean.items():
        color = COLOR_MAP.get(name)
        m_sys = data["median_sys"]
        m_anc = data["median_anc"]

        xs_sys = np.arange(len(m_sys))
        ax.scatter(xs_sys, m_sys, color=color, marker="o", s=28,
                   alpha=0.75, label=f"{name} (system)")

        if m_anc.size > 0:
            xs_anc = np.arange(n_sys_params, n_sys_params + len(m_anc))
            ax.scatter(xs_anc, m_anc, color=color, marker="^", s=42,
                       alpha=0.9, edgecolors="black", linewidths=0.5,
                       label=f"{name} (ancilla)")

    ax.axvline(n_sys_params - 0.5, color="black", linestyle=":",
               linewidth=1.0, alpha=0.6,
               label=f"system / ancilla boundary (k={n_sys_params})")

    ax.set_yscale("log")
    ax.set_xlabel("parameter index k")
    ax.set_ylabel(r"$|\langle \partial W / \partial \theta_k \rangle|$")
    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"
    ax.set_title(
        f"Median gradient magnitude per parameter  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str}), "
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
    print(f"Saved mean-per-param: {out_path}")


# ---- Empirical GOP and its spectrum ----------------------------------
def _empirical_GOP(grads: np.ndarray) -> np.ndarray:
    """Empirical GOP: F = (1/N) G^T G.
    """
    n = grads.shape[0]
    return (grads.T @ grads) / n


def _spectrum(F: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """Eigenvalues of F sorted decreasing, filtering numerical zeros.
    """
    # eigvalsh is for Hermitian matrices; more stable than eig
    eigvals = np.linalg.eigvalsh(F)
    eigvals = eigvals[::-1]  # descending
    if eigvals.size == 0 or eigvals.max() <= 0:
        return np.array([])
    cutoff = rel_tol * eigvals.max()
    return eigvals[eigvals > cutoff]

def _pr_spectral(eigvals: np.ndarray) -> float:
    """Spectral participation ratio: (sum lambda)^2 / sum lambda^2.

    Effective number of directions with comparable weight in F.
    """
    if eigvals.size == 0:
        return 0.0
    s = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float(s * s / s2) if s2 > 0 else 0.0


def _stable_rank(eigvals: np.ndarray) -> float:
    """Stable rank: trace / lambda_max = sum lambda / lambda_max.

    How many eigenvalues 'fit' before reaching the max in total
    magnitude. Insensitive to the tail, sensitive to the peak.
    """
    if eigvals.size == 0:
        return 0.0
    return float(eigvals.sum() / eigvals.max())


def _compute_GOP_diagnostics(grads: np.ndarray) -> dict:
    """Compute the full GOP diagnostic package: F, spectrum, PR, trace, etc.

    Args:
        grads: (N_SAMPLES, n_params).

    Returns:
        dict with:
          - eigvals: filtered eigenvalues, decreasing
          - pr_spec: absolute spectral PR
          - pr_spec_norm: PR_spec / min(N, d)  (in (0, 1])
          - trace: sum of eigenvalues = E[||grad L||^2]
          - lambda_max: maximum eigenvalue
          - stable_rank: trace / lambda_max
          - rank_cap: min(N_SAMPLES, n_params), theoretical rank cap
          - n_nonzero: number of effectively non-null eigenvalues
    """
    n, d = grads.shape
    rank_cap = min(n, d)

    F = _empirical_GOP(grads)
    eigvals = _spectrum(F)

    pr = _pr_spectral(eigvals)
    pr_norm = pr / rank_cap if rank_cap > 0 else 0.0
    trace = float(eigvals.sum()) if eigvals.size > 0 else 0.0
    lam_max = float(eigvals.max()) if eigvals.size > 0 else 0.0
    sr = _stable_rank(eigvals)

    return {
        "eigvals": eigvals,
        "pr_spec": pr,
        "pr_spec_norm": pr_norm,
        "trace": trace,
        "lambda_max": lam_max,
        "stable_rank": sr,
        "rank_cap": rank_cap,
        "n_nonzero": int(eigvals.size),
        "n_samples": n,
        "n_params": d,
    }


def _print_GOP_diagnostics(name: str, GOP_data: dict, label: str = "empirical GOP spectrum") -> None:
    """Console/log summary of the spectral diagnostics."""
    print(f"\n--- {name}  ({label}) ---")
    print(f"  n_samples = {GOP_data['n_samples']}, "
          f"n_params = {GOP_data['n_params']}, "
          f"rank cap (theoretical) = {GOP_data['rank_cap']}")
    print(f"  n_nonzero eigenvalues (after filtering) = "
          f"{GOP_data['n_nonzero']}")
    print(f"  trace(F)         = {GOP_data['trace']:.3e}    "
          f"(= E[||grad L||^2])")
    print(f"  lambda_max       = {GOP_data['lambda_max']:.3e}")
    print(f"  stable rank      = {GOP_data['stable_rank']:.3f}   "
          f"(trace / lambda_max)")
    print(f"  PR_spec          = {GOP_data['pr_spec']:.3f}   "
          f"(effective # of directions)")
    print(f"  PR_spec / rank_cap = {GOP_data['pr_spec_norm']:.3f}   "
          f"(in (0, 1], comparable across configs)")


# ---- Plot: GOP eigenvalue spectrum on log scale ----------------------
def _plot_GOP_spectrum(
    GOP_by_config: dict,
    out_path: str,
    block_label: str = "full",
) -> None:
    """Canonical Abbas-style figure: sorted eigenvalues, log-y.

    One line per config, x = index (1..n_nonzero), y = lambda_a.
    Visually shows:
      - how steeply the spectrum falls (slope) -> concentration
      - how far the useful tail extends (where it hits numerical floor)
      - absolute separation between configs

    Args:
        block_label: "full" or "system block only" — appears in titles.
                     Does NOT change the math; the restriction must already
                     be reflected in GOP_by_config.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_raw, ax_norm = axes

    for name, d in GOP_by_config.items():
        color = COLOR_MAP.get(name, "gray")
        eig = d["eigvals"]
        if eig.size == 0:
            continue

        x = np.arange(1, eig.size + 1)
        ax_raw.plot(x, eig, marker="o", markersize=4, linewidth=1.4,
                    color=color, label=name, alpha=0.85)

        # Normalized by the trace: shows only the spectrum shape,
        # removing global scale differences
        eig_norm = eig / eig.sum()
        ax_norm.plot(x, eig_norm, marker="o", markersize=4, linewidth=1.4,
                     color=color, label=name, alpha=0.85)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue index $a$  (sorted decreasing)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    ax_raw.set_ylabel(r"$\lambda_a$  (raw)")
    ax_raw.set_title("Empirical GOP spectrum (raw)")
    ax_norm.set_ylabel(r"$\lambda_a / \mathrm{tr}(F)$  (normalized)")
    ax_norm.set_title("Empirical GOP spectrum (trace-normalized)")

    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"
    fig.suptitle(
        f"GOP eigenvalue spectrum [{block_label}]  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str})",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved GOP spectrum [{block_label}]: {out_path}")


# ---- GOP analysis stratified by ||grad||^2 deciles -------------------
def _compute_decile_analysis(grads: np.ndarray,
                             n_deciles: int = N_DECILES,
                             stratify_by: np.ndarray | None = None) -> dict:
    """GOP analysis stratified by ||grad||^2 deciles.

    Args:
        grads: (N_SAMPLES, n_params).
        n_deciles: number of strata (10 for deciles, 20 for ventiles).
        stratify_by: optional array of shape (N_SAMPLES,) used to define
                     the deciles. If None, uses ||grad||^2 over the columns
                     of `grads`. Pass ||g_sys||^2 when stratifying by the
                     system block while `grads` already is the system block.
                     (When grads is the system block AND stratify_by is None,
                     deciles are by ||g_sys||^2, which is the coherent choice.)

    Returns:
        dict with arrays of length n_deciles indexed by j = 0..n_deciles-1
        (j=0 is the decile with the lowest stratifier, j=n_deciles-1 the highest):
          - decile_bounds: stratifier bounds (n_deciles+1 values)
          - n_per_decile: number of seeds falling in each decile
          - trace_per_decile: local trace F_j (not normalized by total N)
          - trace_frac_per_decile: contribution to tr(F_total) (sums to 1)
          - pr_per_decile: spectral PR of F_j
          - pr_norm_per_decile: PR_j / min(N_j, d)
          - lambda_max_per_decile: maximum eigenvalue of F_j
          - stable_rank_per_decile: PR_j / lambda_max
          - rank_cap_per_decile: min(N_j, d)
    """
    n, d = grads.shape
    if stratify_by is None:
        stratifier = np.sum(grads ** 2, axis=1)
    else:
        if stratify_by.shape != (n,):
            raise ValueError(
                f"stratify_by must have shape ({n},), got {stratify_by.shape}"
            )
        stratifier = stratify_by

    # Sort by stratifier and split into deciles
    order = np.argsort(stratifier)
    bin_size = n // n_deciles

    decile_bounds = []
    n_per_decile = []
    trace_per_decile = []
    pr_per_decile = []
    pr_norm_per_decile = []
    lambda_max_per_decile = []
    stable_rank_per_decile = []
    rank_cap_per_decile = []

    # Lower bound of the first decile
    decile_bounds.append(float(stratifier[order[0]]))

    for j in range(n_deciles):
        lo_idx = j * bin_size
        # The last decile absorbs the remainder if N does not divide exactly
        hi_idx = (j + 1) * bin_size if j < n_deciles - 1 else n
        decile_idx = order[lo_idx:hi_idx]

        decile_bounds.append(float(stratifier[order[hi_idx - 1]]))
        n_j = decile_idx.size
        n_per_decile.append(n_j)

        # Local GOP of the decile
        g_j = grads[decile_idx]
        F_j = (g_j.T @ g_j) / n_j
        eig_j = _spectrum(F_j)

        trace_j = float(eig_j.sum()) if eig_j.size > 0 else 0.0
        lam_max_j = float(eig_j.max()) if eig_j.size > 0 else 0.0
        pr_j = _pr_spectral(eig_j)
        sr_j = _stable_rank(eig_j)
        rank_cap_j = min(n_j, d)

        trace_per_decile.append(trace_j)
        pr_per_decile.append(pr_j)
        pr_norm_per_decile.append(
            pr_j / rank_cap_j if rank_cap_j > 0 else 0.0
        )
        lambda_max_per_decile.append(lam_max_j)
        stable_rank_per_decile.append(sr_j)
        rank_cap_per_decile.append(rank_cap_j)

    # Fractional contribution to the global trace.
    # When the stratifier equals ||grad_block||^2 (the canonical case here,
    # whether block = full or block = sys), we have
    #   tr(F_total_block) = (1/N) sum_i stratifier_i
    #                     = sum_j (N_j/N) * trace(F_j)
    # so the fraction from decile j is (N_j/N) * trace(F_j) / tr(F_total_block).
    # If stratify_by is a different quantity, this identity does NOT hold and
    # the "fractions" no longer sum to 1 exactly — they remain interpretable
    # as relative contributions but only modulo that caveat.
    trace_total = float(stratifier.sum() / n)
    trace_frac_per_decile = [
        (n_per_decile[j] / n) * trace_per_decile[j] / trace_total
        if trace_total > 0 else 0.0
        for j in range(n_deciles)
    ]

    return {
        "n_deciles": n_deciles,
        "decile_bounds": np.array(decile_bounds),
        "n_per_decile": np.array(n_per_decile),
        "trace_per_decile": np.array(trace_per_decile),
        "trace_frac_per_decile": np.array(trace_frac_per_decile),
        "pr_per_decile": np.array(pr_per_decile),
        "pr_norm_per_decile": np.array(pr_norm_per_decile),
        "lambda_max_per_decile": np.array(lambda_max_per_decile),
        "stable_rank_per_decile": np.array(stable_rank_per_decile),
        "rank_cap_per_decile": np.array(rank_cap_per_decile),
        "trace_total": trace_total,
        "n_samples": n,
        "n_params": d,
    }


def _print_decile_analysis(name: str, dec_data: dict, label: str = "per-decile GOP analysis") -> None:
    """Per-decile summary in the log."""
    print(f"\n--- {name}  ({label}) ---")
    print(f"  n_samples = {dec_data['n_samples']}, "
          f"n_params = {dec_data['n_params']}, "
          f"n_deciles = {dec_data['n_deciles']}")
    print(f"  tr(F_total) = {dec_data['trace_total']:.3e}")
    print(f"  {'decile':<8} {'n':<6} {'|g|^2 hi':<12} "
          f"{'trace_j':<12} {'frac':<8} {'PR_j':<8} "
          f"{'PR_j/cap':<10} {'lam_max_j':<12}")
    for j in range(dec_data["n_deciles"]):
        bound_hi = dec_data["decile_bounds"][j + 1]
        n_j = dec_data["n_per_decile"][j]
        tr_j = dec_data["trace_per_decile"][j]
        frac = dec_data["trace_frac_per_decile"][j]
        pr = dec_data["pr_per_decile"][j]
        prn = dec_data["pr_norm_per_decile"][j]
        lam = dec_data["lambda_max_per_decile"][j]
        print(f"  D{j+1:<7} {n_j:<6} {bound_hi:<12.3e} "
              f"{tr_j:<12.3e} {frac:<8.3f} {pr:<8.3f} "
              f"{prn:<10.3f} {lam:<12.3e}")


def _plot_decile_analysis(
    decile_by_config: dict,
    out_path: str,
    block_label: str = "full",
    stratifier_label: str = r"$\|g\|^2$",
) -> None:
    """Four panels: trace contribution, PR_norm, stable rank, lambda_max
    as a function of decile, one line per config.

    Args:
        block_label: "full" or "system block only" — appears in titles.
        stratifier_label: LaTeX label for the stratifying quantity.
                          e.g. r"$\\|g\\|^2$" or r"$\\|g_\\mathrm{sys}\\|^2$".
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_frac, ax_pr, ax_sr, ax_lam = axes.flat

    for name, d in decile_by_config.items():
        color = COLOR_MAP.get(name, "gray")
        n_dec = d["n_deciles"]
        x = np.arange(1, n_dec + 1)

        ax_frac.plot(x, d["trace_frac_per_decile"], marker="o",
                     markersize=5, linewidth=1.6, color=color,
                     label=name, alpha=0.85)
        ax_pr.plot(x, d["pr_norm_per_decile"], marker="o",
                   markersize=5, linewidth=1.6, color=color,
                   label=name, alpha=0.85)

        sr_norm = (d["stable_rank_per_decile"]
                   / np.maximum(d["rank_cap_per_decile"], 1))
        ax_sr.plot(x, sr_norm, marker="o", markersize=5, linewidth=1.6,
                   color=color, label=name, alpha=0.85)

        ax_lam.plot(x, d["lambda_max_per_decile"], marker="o",
                    markersize=5, linewidth=1.6, color=color,
                    label=name, alpha=0.85)

    # Panel 1: fraction of the trace
    ax_frac.set_xlabel(f"decile of {stratifier_label}  (D1 = lowest)")
    ax_frac.set_ylabel("fraction of $\\mathrm{tr}(F_{\\mathrm{total}})$")
    ax_frac.set_title("Where the signal lives")
    ax_frac.axhline(1.0 / N_DECILES, color="black", linestyle=":",
                    linewidth=0.8, alpha=0.5,
                    label=f"uniform (1/{N_DECILES})")
    ax_frac.grid(True, alpha=0.3)
    ax_frac.legend(fontsize=8, loc="best")
    ax_frac.set_yscale("log")

    # Panel 2: normalized PR
    ax_pr.set_xlabel(f"decile of {stratifier_label}")
    ax_pr.set_ylabel(r"PR$_\mathrm{spec}$ / rank cap")
    ax_pr.set_title("Directional structure per decile")
    ax_pr.set_ylim(0, 1.1)
    ax_pr.axhline(1.0, color="black", linestyle=":", linewidth=0.8,
                  alpha=0.5)
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(fontsize=8, loc="best")

    # Panel 3: normalized stable rank
    ax_sr.set_xlabel(f"decile of {stratifier_label}")
    ax_sr.set_ylabel("stable rank / rank cap")
    ax_sr.set_title("Spectral concentration per decile")
    ax_sr.set_ylim(0, 1.1)
    ax_sr.grid(True, alpha=0.3)
    ax_sr.legend(fontsize=8, loc="best")

    # Panel 4: lambda max
    ax_lam.set_xlabel(f"decile of {stratifier_label}")
    ax_lam.set_ylabel(r"$\lambda_{\max}(F_j)$")
    ax_lam.set_title("Largest local eigenvalue per decile")
    ax_lam.set_yscale("log")
    ax_lam.grid(True, alpha=0.3)
    ax_lam.legend(fontsize=8, loc="best")

    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"
    fig.suptitle(
        f"GOP analysis stratified by {stratifier_label} deciles [{block_label}]  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str})",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved decile analysis [{block_label}]: {out_path}")

def _plot_abs_mean_per_param(
    results_mean: dict,
    out_path: str,
    n_sys_params: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Companion to variance_plot.png: |<dW/dtheta_k>| (mean) per parameter, per config.

    Args:
        results_mean: {config_name: {"mean_sys": np.ndarray, "mean_anc": np.ndarray}}
                      where each value is |mean[dW/dtheta_k]| (already abs-valued).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, data in results_mean.items():
        color = COLOR_MAP.get(name)
        m_sys = data["mean_sys"]
        m_anc = data["mean_anc"]
        xs_sys = np.arange(len(m_sys))
        ax.scatter(xs_sys, m_sys, color=color, marker="o", s=28,
                   alpha=0.75, label=f"{name} (system)")
        if m_anc.size > 0:
            xs_anc = np.arange(n_sys_params, n_sys_params + len(m_anc))
            ax.scatter(xs_anc, m_anc, color=color, marker="^", s=42,
                       alpha=0.9, edgecolors="black", linewidths=0.5,
                       label=f"{name} (ancilla)")
    ax.axvline(n_sys_params - 0.5, color="black", linestyle=":",
               linewidth=1.0, alpha=0.6,
               label=f"system / ancilla boundary (k={n_sys_params})")
    ax.set_yscale("log")
    ax.set_xlabel("parameter index k")
    ax.set_ylabel(r"$|\langle \partial W / \partial \theta_k \rangle|$")
    if CFG.use_choi:
        mode_str = "Choi"
    else:
        mode_str = f"{CFG.batch_mode} B={CFG.batch_size}"
    ax.set_title(
        f"Mean |gradient| per parameter  "
        f"({CFG.system_size} qubits, {CFG.gen_layers} layers, "
        f"ansatz={CFG.gen_ansatz}, {mode_str}), "
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
    print(f"Saved abs-mean-per-param: {out_path}")

# -- Main ----------------------------------
def replot():
    out_dir = os.path.join(_PROJECT_ROOT, "variance_analysis", TIMESTAMP)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Not found: {out_dir}")

    # Mirror everything we print to a .txt log in the same folder
    log_path = os.path.join(out_dir, f"diagnostics_{TIMESTAMP}.txt")
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _write_and_print(original_stdout, log_file)

    try:
        # Load CFG from the saved run - ignores whatever config.py currently has
        _load_cfg_from_file(out_dir)

        # One-line summary of the run's mode flags (helps when comparing logs)
        if CFG.use_choi:
            mode_summary = "Choi"
        else:
            mode_summary = f"{CFG.batch_mode} B={CFG.batch_size}"
        anc_summary = (
            "ancilla_training=ON" if CFG.ancilla_training else "ancilla_training=OFF"
        )
        print(f"\nRun mode: {mode_summary}  |  {anc_summary}\n")

        snapshot = _snapshot_cfg()
        results: dict[str, dict[str, np.ndarray]] = {}
        results_mean: dict[str, dict[str, np.ndarray]] = {}
        grads_by_config: dict[str, np.ndarray] = {}
        diags: dict[str, dict] = {}
        GOP_by_config: dict[str, dict] = {}
        GOP_sys_by_config: dict[str, dict] = {}
        decile_by_config: dict[str, dict] = {}
        decile_sys_by_config: dict[str, dict] = {}
        splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        n_sys_params_ref = None

        try:
            for name in CONFIGS:
                grads_path = os.path.join(out_dir, f"grads_{name}.npy")
                if not os.path.exists(grads_path):
                    print(f"[skip] {name}: {grads_path} not found")
                    continue

                _apply_config(name)
                probe_gen = Generator()

                grads = np.load(grads_path)

                # Sanity check: param count must match the reconstructed generator
                if grads.shape[1] != probe_gen.n_params:
                    raise RuntimeError(
                        f"{name}: saved grads have {grads.shape[1]} params "
                        f"but CFG gives {probe_gen.n_params}. "
                        f"config.txt may be incomplete."
                    )

                grads_sys, grads_anc = _split_system_ancilla(grads, probe_gen)
                var_sys = np.var(grads_sys, axis=0)
                var_anc = (np.var(grads_anc, axis=0)
                           if grads_anc.size > 0 else np.array([]))

                results[name] = {"var_sys": var_sys, "var_anc": var_anc}

                if n_sys_params_ref is None:
                    n_sys_params_ref = len(var_sys)

                print(f"[ok]   {name}: n_samples={grads.shape[0]}, "
                      f"n_params={grads.shape[1]} "
                      f"(sys={len(var_sys)}, anc={len(var_anc)})")

                median_sys = np.abs(np.median(grads_sys, axis=0))
                median_anc = (np.abs(np.median(grads_anc, axis=0))
                              if grads_anc.size > 0 else np.array([]))
                mean_sys = np.abs(np.mean(grads_sys, axis=0))
                mean_anc = (np.abs(np.mean(grads_anc, axis=0))
                              if grads_anc.size > 0 else np.array([]))
                results_mean[name] = {"median_sys": median_sys,
                                      "median_anc": median_anc,
                                      "mean_sys": mean_sys,
                                      "mean_anc": mean_anc}
                grads_by_config[name] = grads

                # Build sys/anc index lists once, used by several blocks below
                if probe_gen.ancilla:
                    anc_idx = np.array(
                        probe_gen._get_ancilla_param_indices(), dtype=int
                    )
                else:
                    anc_idx = np.array([], dtype=int)
                sys_idx = np.setdiff1d(
                    np.arange(grads.shape[1]), anc_idx, assume_unique=True
                )
                splits[name] = (sys_idx, anc_idx)

                if RUN_DIAGNOSTICS:
                    diags[name] = _print_diagnostics(name, grads)

                if RUN_GOP:
                    GOP_by_config[name] = _compute_GOP_diagnostics(grads)
                    _print_GOP_diagnostics(name, GOP_by_config[name],
                                           label="empirical GOP spectrum [full]")

                if RUN_GOP_SYS:
                    # GOP restricted to the system block: same math, different
                    # input. Dimension is n_sys for all configs, so trace,
                    # lambda_max, eigenvalue counts etc. are directly
                    # comparable across configs.
                    grads_sys_only = grads[:, sys_idx]
                    GOP_sys_by_config[name] = _compute_GOP_diagnostics(grads_sys_only)
                    _print_GOP_diagnostics(name, GOP_sys_by_config[name],
                                           label="empirical GOP spectrum [system block]")

                if RUN_DECILES:
                    decile_by_config[name] = _compute_decile_analysis(grads)
                    _print_decile_analysis(name, decile_by_config[name],
                                           label="per-decile GOP analysis [full]")

                if RUN_DECILES_SYS:
                    # Deciles on the system block, stratified by ||g_sys||^2.
                    # This is the coherent choice: high-||g_sys|| seeds are the
                    # ones where the system trains well, regardless of what
                    # the ancilla block is doing.
                    grads_sys_only = grads[:, sys_idx]
                    decile_sys_by_config[name] = _compute_decile_analysis(
                        grads_sys_only
                    )
                    _print_decile_analysis(name, decile_sys_by_config[name],
                                           label="per-decile GOP analysis [system block]")
        finally:
            _restore_cfg(snapshot)

        if not results:
            print("No configs loaded, nothing to plot.")
            return

        plot_path = os.path.join(out_dir, OUT_NAME)
        plot_variance_sweep(results, plot_path, n_sys_params_ref or 0,
                            xlim=XLIM, ylim=YLIM)
        print(f"\nSaved: {plot_path}")

        if RUN_DIAGNOSTICS and diags:
            hist_path = os.path.join(out_dir, f"hist_{TIMESTAMP}.png")
            _plot_histograms(diags, grads_by_config, splits, hist_path)

            est_path = os.path.join(out_dir, f"estimators_{TIMESTAMP}.png")
            _plot_estimator_comparison(
                diags, splits, est_path, n_sys_params_ref or 0
            )

        if RUN_NORM_DIST and grads_by_config:
            norm_path = os.path.join(out_dir, f"gradient_norm_{TIMESTAMP}.png")
            _plot_norm_distribution(grads_by_config, splits, norm_path)

        if RUN_GOP and GOP_by_config:
            spec_path = os.path.join(out_dir, f"GOP_spectrum_{TIMESTAMP}.png")
            _plot_GOP_spectrum(GOP_by_config, spec_path, block_label="full")

        if RUN_GOP_SYS and GOP_sys_by_config:
            spec_sys_path = os.path.join(
                out_dir, f"GOP_spectrum_sys_{TIMESTAMP}.png"
            )
            _plot_GOP_spectrum(GOP_sys_by_config, spec_sys_path,
                               block_label="system block only")

        if RUN_DECILES and decile_by_config:
            dec_path = os.path.join(out_dir, f"GOP_deciles_{TIMESTAMP}.png")
            _plot_decile_analysis(decile_by_config, dec_path,
                                  block_label="full",
                                  stratifier_label=r"$\|g\|^2$")

        if RUN_DECILES_SYS and decile_sys_by_config:
            dec_sys_path = os.path.join(
                out_dir, f"GOP_deciles_sys_{TIMESTAMP}.png"
            )
            _plot_decile_analysis(decile_sys_by_config, dec_sys_path,
                                  block_label="system block only",
                                  stratifier_label=r"$\|g_\mathrm{sys}\|^2$")
        
        if RUN_MEAN_PARAM and results_mean:
            mean_path = os.path.join(
                out_dir, f"gradient_median_per_param_{TIMESTAMP}.png"
            )
            _plot_mean_per_param(results_mean, mean_path,
                                n_sys_params_ref or 0,
                                xlim=XLIM, ylim=YLIM)
            abs_mean_path = os.path.join(
                out_dir, f"gradient_mean_per_param_{TIMESTAMP}.png"
            )
            _plot_abs_mean_per_param(results_mean, abs_mean_path,
                                    n_sys_params_ref or 0,
                                    xlim=XLIM, ylim=YLIM)
        if RUN_KS:
            _ks_test_pairs(grads_by_config, splits, KS_PAIRS)

        print(f"\nSaved diagnostics log: {log_path}")
    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == "__main__":
    replot()