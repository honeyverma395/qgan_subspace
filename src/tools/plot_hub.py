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
"""The plot tool"""

import os
import re

import matplotlib as mpl
import numpy as np

from tools.data_managers import print_and_log

mpl.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt


########################################################################
# MAIN PLOTTING FUNCTION
########################################################################
def generate_all_plots(
    base_path, log_path, n_runs, max_fidelity, common_initial_plateaus=None, run_names=None, x_label: str = "Run"
):
    """Generate all plots.

    Supports backward-compatible boolean `common_initial_plateaus`

    Optional `run_names` allows custom x-axis labels (list or dict mapping run index to string).
    """
    # 1) Histogram-style plots (unchanged)
    for run_idx in range(1, n_runs + 1):
        plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus)
    plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus)

    # 2) Scatter-style plots (refactored)
    if not common_initial_plateaus:
        # from_scratch (and change): single merged scatter
        scatter_plot(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
    else:
        # from_common_plateaus: now plots (clouds, avg_fidelity, success, combined, overall)
        scatter_plateau_clouds(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_avg_fidelity(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_success(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_avg_success_combined(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
        scatter_plateau_overall(base_path, log_path, n_runs, max_fidelity, run_names, x_label)
    # Gradient trajectory plots
    plot_grad_trajectory(base_path, log_path, n_runs,
                         common_initial_plateaus=common_initial_plateaus,
                         run_names=run_names)
    plot_grad_joined_all(base_path, log_path, n_runs,
                            common_initial_plateaus=common_initial_plateaus,
                            run_names=run_names)
    plot_grad_joined_mean(base_path, log_path, n_runs,
                             common_initial_plateaus=common_initial_plateaus,
                             run_names=run_names)

########################################################################
# REAL TIME RUN PLOTTING FUNCTION
########################################################################
def plt_fidelity_vs_iter(fidelities, losses, config, indx=0):
    fig, (axs1, axs2) = plt.subplots(1, 2)
    axs1.plot(range(len(fidelities)), fidelities)
    axs1.set_xlabel("Iteration")
    axs1.set_ylabel("Fidelity")
    axs1.set_title("Fidelity <target|gen> vs Iterations")
    axs2.plot(range(len(losses)), losses)
    axs2.set_xlabel("Iteration")
    axs2.set_ylabel("Loss")
    axs2.set_title("Wasserstein Loss vs Iterations")
    plt.tight_layout()

    # Save the figure
    fig_path = f"{config.figure_path}/{config.system_size}qubit_{config.gen_layers}_{indx}.png"
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


#########################################################################
# PLOT INDIVIDUAL RUNS HISTOGRAMS
#########################################################################
def plot_recurrence_vs_fid(base_path, log_path, run_idx, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors  # Consistent palette for control and runs
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    control_hist, _ = np.histogram(control_fids, bins=bins) if control_fids else (np.zeros(len(bins) - 1), bins)
    changed_hist, _ = np.histogram(changed_fids, bins=bins)
    # Renormalize histograms to show distributions
    control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
    changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(8, 6))
    width = (bins[1] - bins[0]) * 0.4
    bars = []
    if common_initial_plateaus and np.any(control_hist):
        bars.append(
            plt.bar(
                bin_centers - width / 2,
                control_hist,
                width=width,
                label=f"Control (no change) ({len(control_fids)} tries)",
                alpha=0.7,
                color=run_colors[0],
            )
        )
    if np.any(changed_hist):
        # Use the second color from the palette for the first run, or cycle if run_idx is given
        run_color = run_colors[run_idx % len(run_colors)] if run_idx else run_colors[1]
        run_label = (
            f"Run {run_idx} ({len(changed_fids)} tries)" if run_idx else f"Experiment Runs ({len(changed_fids)} tries)"
        )
        bars.append(
            plt.bar(
                bin_centers + width / 2,
                changed_hist,
                width=width,
                label=run_label,
                alpha=0.7,
                color=run_color,
            )
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Distribution vs Maximum Fidelity"
    if run_idx:
        title += f" (run {run_idx})"
    elif not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if bars:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(
        base_path,
        f"comparison_distribution_vs_fidelity_run{run_idx}.png" if run_idx else "distribution_vs_fidelity.png",
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


###########################################################################
# PLOT COMPARISON OF ALL HISTOGRAMS TOGETHER
###########################################################################
def plot_comparison_all_runs(base_path, log_path, n_runs, max_fidelity, common_initial_plateaus):
    run_colors = plt.cm.tab10.colors
    control_fids = (
        collect_max_fidelities_nested(base_path, r"repeated_control", None) if common_initial_plateaus else []
    )
    # Split last bin at max_fidelity
    bins = [*list(np.linspace(0, max_fidelity, 20)), max_fidelity, 1.0]
    bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
    plt.figure(figsize=(10, 7))
    all_hists = []
    all_labels = []
    all_colors = []
    # Collect control as first 'run' if present
    if common_initial_plateaus and len(control_fids) > 0:
        control_hist, _ = np.histogram(control_fids, bins=bins)
        control_hist = control_hist / control_hist.sum() if control_hist.sum() > 0 else control_hist
        all_hists.append(control_hist)
        all_labels.append(f"Control (no change) ({len(control_fids)} tries)")
        all_colors.append(run_colors[0])
    # Collect all runs
    for run_idx in range(1, n_runs + 1):
        if common_initial_plateaus:
            changed_fids = collect_latest_changed_fidelities_nested_run(base_path, run_idx)
        else:
            changed_fids = collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx)
        changed_hist, _ = np.histogram(changed_fids, bins=bins)
        changed_hist = changed_hist / changed_hist.sum() if changed_hist.sum() > 0 else changed_hist
        all_hists.append(changed_hist)
        all_labels.append(f"Run {run_idx} ({len(changed_fids)} tries)")
        all_colors.append(run_colors[run_idx % len(run_colors)])
    # Plot as grouped bars: each group is a run (control is group 0 if present)
    n_groups = len(all_hists)
    width = (bins[1] - bins[0]) * 0.7 / n_groups
    for i, (hist, label, color) in enumerate(zip(all_hists, all_labels, all_colors)):
        plt.bar(
            bin_centers + width * (i - n_groups / 2 + 0.5),
            hist,
            width=width,
            label=label,
            alpha=0.7,
            color=color,
        )
    plt.xlabel("Maximum Fidelity Reached")
    plt.ylabel("Distribution (Fraction)")
    title = "Comparison: Distribution vs Maximum Fidelity (All Runs)"
    if not common_initial_plateaus:
        title += " (Experiment Mode)"
    plt.title(title)
    if n_groups > 0:
        plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_path, "comparison_distribution_vs_fidelity_all.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


##########################################################################
# HELPER FUNCTIONS TO COLLECT MAX FIDELITIES
##########################################################################
def get_max_fidelity_from_file(fid_loss_path):
    if not os.path.exists(fid_loss_path):
        return None
    try:
        data = np.loadtxt(fid_loss_path)
        if data.ndim == 1:
            fidelities = data
        else:
            fidelities = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
        return np.max(fidelities)
    except (OSError, IOError, ValueError):
        return None


def collect_max_fidelities_nested(base_path, outer_pattern, inner_pattern):
    """
    Collect max fidelities from all outer_pattern/inner_pattern/fidelities/log_fidelity_loss.txt
    """
    max_fids = []
    for root, dirs, files in os.walk(base_path):
        if (
            re.search(outer_pattern, root)
            and (inner_pattern is None or re.search(inner_pattern, root))
            and os.path.basename(root) == "fidelities"
            and "log_fidelity_loss.txt" in files
        ):
            fid_loss_path = os.path.join(root, "log_fidelity_loss.txt")
            max_fid = get_max_fidelity_from_file(fid_loss_path)
            if max_fid is not None:
                max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested(base_path, common_initial_plateaus, run_idx=None):
    """
    Collect max fidelities for changed runs, supporting both folder structures.
    common_initial_plateaus: boolean, if True, uses the initial plateaus structure.
    If run_idx is not None, only collect for that run.
    """
    run_dirs = {}
    if common_initial_plateaus:
        pattern = (
            rf"initial_plateau_(\d+)/repeated_changed_run{run_idx}/(\d+)/fidelities$"
            if run_idx is not None
            else r"initial_plateau_(\d+)/repeated_changed_run(\d+)/(\d+)/fidelities$"
        )
    elif run_idx is not None:
        pattern = rf"experiment{run_idx}/(\d+)/fidelities$"
    else:
        pattern = r"experiment(\d+)/(\d+)/fidelities$"
    for root, dirs, files in os.walk(base_path):
        m = re.search(pattern, root)
        if m and "log_fidelity_loss.txt" in files:
            if common_initial_plateaus:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[2])
                else:
                    run_y = int(m[2])
                    x_num = int(m[3])
                exp_j = int(m[1])
                key = (exp_j, x_num)
            else:
                if run_idx is not None:
                    run_y = run_idx
                    x_num = int(m[1])
                else:
                    run_y = int(m[1])
                    x_num = int(m[2])
                key = (run_y, x_num)
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_latest_changed_fidelities_nested_run(base_path, run_idx):
    run_dirs = {}
    for root, dirs, files in os.walk(base_path):
        m = re.search(
            r"initial_plateau_(\d+)[/\\]repeated_changed_run" + str(run_idx) + r"[/\\](\d+)[/\\]fidelities$", root
        )
        if m and "log_fidelity_loss.txt" in files:
            exp_j = int(m[1])
            x_num = int(m[2])
            key = (exp_j, x_num)
            run_y = run_idx
            if key not in run_dirs or run_y > run_dirs[key][0]:
                run_dirs[key] = (run_y, os.path.join(root, "log_fidelity_loss.txt"))
    max_fids = []
    for run_y, fid_loss_path in run_dirs.values():
        max_fid = get_max_fidelity_from_file(fid_loss_path)
        if max_fid is not None:
            max_fids.append(max_fid)
    return max_fids


def collect_fidelities_by_plateau_for_run(base_path, run_idx):
    """
    Collect max fidelities grouped by plateau for a specific changed run (run_idx).

    For each repetition folder:
      initial_plateau_<P>/repeated_changed_run<run_idx>/<rep>/fidelities/log_fidelity_loss.txt

    We combine it with the initial plateau log:
      initial_plateau_<P>/log_fidelity_loss.txt

    and take max( max(rep_fids), max(initial_fids) ) so the value reflects the
    whole (initial + continued) trajectory. Each repetition contributes one
    combined max value to the list for that plateau.
    """
    plateau_fids: dict[int, list[float]] = {}

    for root, dirs, files in os.walk(base_path):
        # We only care about directories that end with 'fidelities' and have the fidelity log
        if not root.endswith(("fidelities", "fidelities/")):
            continue
        if "log_fidelity_loss.txt" not in files:
            continue
        # Precise filter: must contain the exact run folder
        run_marker = f"repeated_changed_run{run_idx}"
        if f"/{run_marker}/" not in root.replace("\\", "/"):
            continue
        # Must be inside an initial_plateau_<P>
        m_plateau = re.search(r"initial_plateau_(\d+)", root)
        if not m_plateau:
            continue
        plateau_num = int(m_plateau.group(1))

        rep_fid_path = os.path.join(root, "log_fidelity_loss.txt")
        rep_max = get_max_fidelity_from_file(rep_fid_path)

        initial_log = os.path.join(base_path, f"initial_plateau_{plateau_num}", "log_fidelity_loss.txt")
        init_max = get_max_fidelity_from_file(initial_log)

        candidates = [v for v in (rep_max, init_max) if v is not None]
        if not candidates:
            continue
        combined_max = max(candidates)
        plateau_fids.setdefault(plateau_num, []).append(combined_max)

    return plateau_fids


def collect_fidelities_by_plateau_control(base_path):
    """
    Collect max fidelities for control repetitions grouped by plateau.

    Control repetition folders:
      initial_plateau_<P>/repeated_control/<rep>/fidelities/log_fidelity_loss.txt

    Combine each control repetition with the initial plateau:
      initial_plateau_<P>/log_fidelity_loss.txt

    Store one combined max per repetition in the list for that plateau.
    """
    plateau_fids: dict[int, list[float]] = {}

    for root, dirs, files in os.walk(base_path):
        if not root.endswith(("fidelities", "fidelities/")):
            continue
        if "log_fidelity_loss.txt" not in files:
            continue
        if "repeated_control" not in root:
            continue
        m_plateau = re.search(r"initial_plateau_(\d+)", root)
        if not m_plateau:
            continue
        plateau_num = int(m_plateau.group(1))

        control_fid_path = os.path.join(root, "log_fidelity_loss.txt")
        control_max = get_max_fidelity_from_file(control_fid_path)

        initial_log = os.path.join(base_path, f"initial_plateau_{plateau_num}", "log_fidelity_loss.txt")
        init_max = get_max_fidelity_from_file(initial_log)

        candidates = [v for v in (control_max, init_max) if v is not None]
        if not candidates:
            continue
        combined_max = max(candidates)
        plateau_fids.setdefault(plateau_num, []).append(combined_max)

    return plateau_fids


# Helpers for labels
def _label_for_run(run_idx, tries, run_names=None):
    if isinstance(run_names, dict):
        base = run_names.get(run_idx, f"Run {run_idx}")
    elif isinstance(run_names, (list, tuple)) and 0 <= run_idx - 1 < len(run_names):
        base = str(run_names[run_idx - 1])
    else:
        base = f"Run {run_idx}"
    return f"{base}\n({tries} tries)" if tries and tries > 0 else base


def _base_label_for_run(run_idx, run_names=None):
    """Return the base label for a run without tries count."""
    if isinstance(run_names, dict):
        return run_names.get(run_idx, f"Run {run_idx}")
    if isinstance(run_names, (list, tuple)) and 0 <= run_idx - 1 < len(run_names):
        return str(run_names[run_idx - 1])
    return f"Run {run_idx}"


def _draw_tries_sublabels(ax, x_positions, tries_counts, fontsize=8, y_offset=-0.045, color="0.35"):
    """Draw smaller '(X tries)' sublabels below the x-axis tick labels.

    Uses the x-axis transform so x is in data coords and y is in axes coords.
    """
    for x, t in zip(x_positions, tries_counts):
        if t and t > 0:
            ax.text(
                x,
                y_offset,
                f"({t} tries)",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=fontsize,
                color=color,
            )


def _make_jittered_xs(base_x: float, count: int, jitter: float = 0.12):
    """Return x positions jittered around base_x to reduce overlap.

    Only generates the xs so callers can choose their own scatter styling.
    """
    if not count:
        return []
    return [base_x + np.random.uniform(-jitter, jitter) for _ in range(count)]


########################################################################
# FROM SCRATCH SCATTER PLOTS
########################################################################
def scatter_plot(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Right axis for percentages
    # Threshold line on left axis now in percent
    ax1.axhline(100 * max_fidelity, color="C0", linestyle="--", label=f"max_fidelity={max_fidelity}")

    # Plot clouds and compute averages
    avg_fid_percent = []
    avg_success_percent = []
    x_ticks = []
    base_labels = []
    tries_counts = []

    for run_idx in range(1, n_runs + 1):
        vals = collect_latest_changed_fidelities_nested(base_path, False, run_idx)
        tries = len(vals) if vals else 0
        # Cloud on left axis (percent)
        if vals:
            xs = _make_jittered_xs(run_idx, len(vals))
            ax1.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.4, s=18, zorder=3)
        avg_f = (np.nanmean(vals) * 100) if vals else 0.0
        avg_s = (100.0 * np.sum(np.array(vals) >= max_fidelity) / tries) if tries > 0 else 0.0
        std_f = (np.nanstd(vals) * 100) if vals and len(vals) > 1 else 0.0
        avg_fid_percent.append(avg_f)
        avg_success_percent.append(avg_s)
        ax2.errorbar(
            [run_idx], [avg_f], yerr=[std_f],
            fmt="o", color="green", ecolor="green",
            markeredgecolor="black", markersize=8,
            elinewidth=1.5, capsize=4, zorder=3,
        )        
        # Centered overlays on right axis (percent)
        ax2.scatter([run_idx], [avg_f], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=3)
        ax2.scatter([run_idx], [avg_s], color="red", marker="D", s=50, zorder=3)
        # Annotations for averages (percent)
        t1 = ax2.text(
            run_idx + 0.07,
            avg_f,
            f"{avg_f:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="green",
        )
        t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        t2 = ax2.text(
            run_idx + 0.07,
            avg_s,
            f"{avg_s:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="red",
        )
        t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # Axes labels
    ax1.set_ylabel("Best Fidelity Achieved in each Repetition (%)")
    ax2.set_ylabel("Average/Success Rate of each Run (%)")
    ax1.set_xlabel(x_label, labelpad=18)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(base_labels)
    _draw_tries_sublabels(ax1, x_ticks, tries_counts, fontsize=8)
    fig.subplots_adjust(bottom=0.24)

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="--", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            alpha=0.4,
            markersize=6,
            linestyle="None",
            label="Repetitions Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
    ]
    ax1.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plot.png")
    fig.tight_layout()
    fig.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


########################################################################
# PLATEAUS SCATTERS PLOTS
########################################################################
def _collect_all_plateau_ids(base_path):
    ids = set()
    # look for initial_plateau_X
    for root, dirs, files in os.walk(base_path):
        if m := re.search(r"initial_plateau_(\d+)(?=/|$)", root):
            ids.add(int(m[1]))
    return sorted(ids)


def scatter_plateau_clouds(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax2 = ax.twinx()  # mirror/right axis for success rate overlays
    ax.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")

    x_ticks, base_labels, tries_counts = [], [], []

    # Control run on the far left (x=0)
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        for vals in control_plateau_fids.values():
            if not vals:
                continue
            xs = _make_jittered_xs(0, len(vals))
            ax.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.7, s=18, label=None)
        # Control overall averages: blue squares centered at x=0 on both axes
        control_vals_flat = [v for lst in control_plateau_fids.values() for v in lst]
        if control_vals_flat:
            overall_fid = float(np.nanmean(control_vals_flat)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(control_vals_flat) >= max_fidelity) / len(control_vals_flat)
            std_fid_control = float(np.nanstd(control_vals_flat)) * 100.0 if len(control_vals_flat) > 1 else 0.0
            ax.scatter([0], [overall_fid], color="green", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=5)
            ax2.scatter([0], [overall_succ], color="red", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=5)
            ax.errorbar(
                [0], [overall_fid], yerr=[std_fid_control],
                fmt="s", color="green", ecolor="blue",
                markeredgecolor="blue", markersize=8,
                elinewidth=1.5, capsize=4, zorder=5,
            )
            # value tags
            t_f = ax.text(
                0 + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t_f.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            t_s = ax2.text(
                0 + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t_s.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        for vals in plateau_fids.values():
            if not vals:
                continue
            xs = _make_jittered_xs(run_idx, len(vals), jitter=0.15)
            ax.scatter(xs, [v * 100 for v in vals], color="gray", alpha=0.7, s=18, label=None)
        # Overlay run averages (when data exists): Avg Fidelity on left axis, Success on right axis
        all_vals = [v for lst in plateau_fids.values() for v in lst]
        if all_vals:
            overall_fid = float(np.nanmean(all_vals)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(all_vals) >= max_fidelity) / len(all_vals)
            std_fid = float(np.nanstd(all_vals)) * 100.0 if len(all_vals) > 1 else 0.0
            # Avg fidelity (green circle) + tag on left axis
            ax.scatter([run_idx], [overall_fid], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=5)
            ax.errorbar(
                [run_idx], [overall_fid], yerr=[std_fid],
                fmt="o", color="green", ecolor="green",
                markeredgecolor="black", markersize=8,
                elinewidth=1.5, capsize=4, zorder=5,
            )            
            t = ax.text(
                run_idx + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            # Success rate (red diamond) + tag on right axis
            ax2.scatter([run_idx], [overall_succ], color="red", marker="D", s=55, zorder=5)
            t2 = ax2.text(
                run_idx + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    ax.set_ylabel("Best Fidelity Achieved in each Repetition (%)")
    ax2.set_ylabel("Average/Success Rate of each Run (%)")
    ax.set_xlabel(x_label, labelpad=22)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.26)

    ax.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend: threshold + example repetition marker + run averages overlays + control
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="-", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, linestyle="None", label="Repetition"
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=None,
            markeredgecolor="blue",
            markersize=7,
            linestyle="None",
            label="Control (both)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_clouds.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def _pastelize(color, factor: float = 0.6):
    """Lighten a matplotlib color toward white by the given factor (0..1)."""
    r, g, b, a = mpl.colors.to_rgba(color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b, a)


def scatter_plateau_avg_fidelity(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")
    plateau_ids = _collect_all_plateau_ids(base_path)
    cmap = plt.cm.get_cmap("tab20", max(1, len(plateau_ids)))
    plateau_colors = {pid: cmap(i % cmap.N) for i, pid in enumerate(plateau_ids)}
    line_colors = {pid: _pastelize(plateau_colors[pid], 0.5) for pid in plateau_ids}

    x_ticks, base_labels, tries_counts = [], [], []
    plateau_series = {pid: {} for pid in plateau_ids}

    # Control averages at x=0 (collect and plot jittered gray points)
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        control_avgs = []
        for pid in plateau_ids:
            if vals := control_plateau_fids.get(pid, []):
                avg = float(np.nanmean(vals))
                plateau_series[pid][0] = avg
                control_avgs.append(avg)
        xs = _make_jittered_xs(0, len(control_avgs))
        ax.scatter(xs, [v * 100 for v in control_avgs], color="gray", s=45, zorder=3, alpha=0.7)
        if control_vals_flat := [v for lst in control_plateau_fids.values() for v in lst]:
            overall_fid = float(np.nanmean(control_vals_flat))
            ax.scatter(
                [0], [overall_fid * 100], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                0 + 0.1,
                overall_fid * 100,
                f"{overall_fid * 100:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        run_avgs = []
        for pid in plateau_ids:
            if vals := plateau_fids.get(pid, []):
                avg = float(np.nanmean(vals))
                plateau_series[pid][run_idx] = avg
                run_avgs.append(avg)
        xs = _make_jittered_xs(run_idx, len(run_avgs))
        ax.scatter(xs, [v * 100 for v in run_avgs], color="gray", s=45, zorder=3, alpha=0.7)
        if all_vals := [v for lst in plateau_fids.values() for v in lst]:
            overall_fid = float(np.nanmean(all_vals))
            ax.scatter(
                [run_idx], [overall_fid * 100], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                run_idx + 0.1,
                overall_fid * 100,
                f"{overall_fid * 100:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # connect lines per plateau in percent
    for pid, series in plateau_series.items():
        if len(series) > 1:
            xs = sorted(series.keys())
            ys = [series[i] * 100 for i in xs]
            plt.plot(xs, ys, "--", color=line_colors[pid], alpha=0.5, linewidth=1, zorder=1)

    ax.set_ylabel("Average Best Fidelity (%)")
    ax.set_xlabel(x_label, labelpad=22)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.26)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend remains the same
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="-", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Same plateau"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            linestyle="None",
            label="Plateau Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control Avg Best Fidelity (%)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_avg_fidelity.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def scatter_plateau_success(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plateau_ids = _collect_all_plateau_ids(base_path)
    cmap = plt.cm.get_cmap("tab20", max(1, len(plateau_ids)))
    plateau_colors = {pid: cmap(i % cmap.N) for i, pid in enumerate(plateau_ids)}
    line_colors = {pid: _pastelize(plateau_colors[pid], 0.5) for pid in plateau_ids}

    x_ticks, base_labels, tries_counts = [], [], []
    plateau_series = {pid: {} for pid in plateau_ids}

    # Control success at x=0
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        control_succ = []
        for pid in plateau_ids:
            if vals := control_plateau_fids.get(pid, []):
                succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
                plateau_series[pid][0] = succ
                control_succ.append(succ)
        xs = _make_jittered_xs(0, len(control_succ))
        ax.scatter(xs, control_succ, color="gray", s=45, zorder=3, alpha=0.7)
        if control_vals_flat := [v for lst in control_plateau_fids.values() for v in lst]:
            overall_succ = 100.0 * np.sum(np.array(control_vals_flat) >= max_fidelity) / len(control_vals_flat)
            ax.scatter(
                [0], [overall_succ], color="blue", marker="s", edgecolors="black", linewidths=0.5, s=60, zorder=4
            )
            t = ax.text(
                0 + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        run_succ = []
        for pid in plateau_ids:
            if vals := plateau_fids.get(pid, []):
                succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
                plateau_series[pid][run_idx] = succ
                run_succ.append(succ)
        xs = _make_jittered_xs(run_idx, len(run_succ))
        ax.scatter(xs, run_succ, color="gray", s=45, zorder=3, alpha=0.7)
        if all_vals := [v for lst in plateau_fids.values() for v in lst]:
            overall_succ = 100.0 * np.sum(np.array(all_vals) >= max_fidelity) / len(all_vals)
            ax.scatter([run_idx], [overall_succ], color="red", marker="D", s=55, zorder=4)
            t = ax.text(
                run_idx + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # connect lines per plateau
    for pid, series in plateau_series.items():
        if len(series) > 1:
            xs = sorted(series.keys())
            ys = [series[i] for i in xs]
            plt.plot(xs, ys, "--", color=line_colors[pid], alpha=0.5, linewidth=1, zorder=1)

    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel(x_label)
    plt.xticks(x_ticks, base_labels)
    _draw_tries_sublabels(ax, x_ticks, tries_counts, fontsize=8)
    ax.figure.subplots_adjust(bottom=0.24)

    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Same plateau"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=6,
            linestyle="None",
            label="Plateau Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Control Success Rate (%)",
        ),
    ]
    plt.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_success.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close()


def scatter_plateau_overall(base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    # Removed threshold line for the overall plot
    # ax1.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")

    x_ticks, base_labels, tries_counts = [], [], []

    # Control overall at x=0
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    if control_vals := [v for lst in control_plateau_fids.values() for v in lst]:
        control_tries = len(control_vals)
        avg_fid = float(np.nanmean(control_vals)) * 100.0
        avg_succ = 100.0 * np.sum(np.array(control_vals) >= max_fidelity) / control_tries if control_tries > 0 else 0.0
        std_fid_control = float(np.nanstd(control_vals)) * 100.0 if len(control_vals) > 1 else 0.0
        # Control as blue squares centered at x=0 on both axes
        ax1.errorbar(
            [0], [avg_fid], yerr=[std_fid_control],
            fmt="s", color="green", ecolor="blue",
            markeredgecolor="blue", markersize=8,
            elinewidth=1.5, capsize=4, zorder=4,
        )
        ax1.scatter([0], [avg_fid], color="green", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=4)
        ax2.scatter([0], [avg_succ], color="red", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=4)
        # value tags
        t1 = ax1.text(
            0 + 0.1,
            avg_fid,
            f"{avg_fid:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="blue",
            zorder=100,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )
        t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        t2 = ax2.text(
            0 + 0.1,
            avg_succ,
            f"{avg_succ:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
            color="blue",
            zorder=100,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
        )
        t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    # Runs
    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        vals = [v for lst in plateau_fids.values() for v in lst]
        tries = len(vals)
        avg_fid = float(np.nanmean(vals) * 100.0) if vals else 0.0
        avg_succ = (100.0 * np.sum(np.array(vals) >= max_fidelity) / tries) if tries > 0 else 0.0
        std_fid = float(np.nanstd(vals) * 100.0) if vals and len(vals) > 1 else 0.0

        # Only if there is data for this run
        if tries > 0:
            # Avg fidelity on left (green circle)
            ax1.errorbar(
                [run_idx], [avg_fid], yerr=[std_fid],
                fmt="o", color="green", ecolor="green",
                markeredgecolor="black", markersize=8,
                elinewidth=1.5, capsize=4, zorder=3,
            )
            ax1.scatter([run_idx], [avg_fid], color="green", edgecolors="black", linewidths=0.5, s=60)
            # Avg success on right (red diamond)
            ax2.scatter([run_idx], [avg_succ], color="red", marker="D", s=55)

            # tags
            t1 = ax1.text(
                run_idx + 0.1,
                avg_fid,
                f"{avg_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

            t2 = ax2.text(
                run_idx + 0.1,
                avg_succ,
                f"{avg_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    ax1.set_ylabel("Average Best Fidelity (%)")
    ax2.set_ylabel("Success Rate (%)")
    ax1.set_xlabel(x_label, labelpad=22)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(base_labels)
    _draw_tries_sublabels(ax1, x_ticks, tries_counts, fontsize=8)
    fig.subplots_adjust(bottom=0.26)

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Updated legend without threshold entry
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Avg Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=None,
            markeredgecolor="blue",
            markersize=7,
            linestyle="None",
            label="Control (both)",
        ),
    ]
    ax1.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_overall.png")
    fig.tight_layout()
    fig.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


def scatter_plateau_avg_success_combined(
    base_path, log_path, n_runs, max_fidelity, run_names=None, x_label: str = "Run"
):
    """Combined plot: plateau avg fidelity (light green) + plateau success (light red),
    plus overall run markers and control markers with value tags."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Fidelity threshold on left axis
    ax1.axhline(100 * max_fidelity, color="C0", linestyle="-", label=f"max_fidelity={max_fidelity}")

    # Colors
    light_green = _pastelize("green", 0.7)
    light_red = _pastelize("red", 0.7)

    x_ticks, base_labels, tries_counts = [], [], []

    # Prepare series to draw plateau lines across runs
    plateau_ids = _collect_all_plateau_ids(base_path)
    series_fid = {pid: {} for pid in plateau_ids}
    series_succ = {pid: {} for pid in plateau_ids}

    # Control at x=0
    control_plateau_fids = collect_fidelities_by_plateau_control(base_path)
    control_tries = sum(len(v) for v in control_plateau_fids.values()) if control_plateau_fids else 0
    if control_plateau_fids:
        fid_points = []
        succ_points = []
        for pid in plateau_ids:
            vals = control_plateau_fids.get(pid, [])
            if not vals:
                continue
            avg_f = float(np.nanmean(vals)) * 100.0
            succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
            series_fid[pid][0] = (
                avg_f / 100.0
            )  # store back in 0..1 for consistency with other funcs (then *100 on draw)
            series_succ[pid][0] = succ
            fid_points.append(avg_f)
            succ_points.append(succ)
        if fid_points:
            xs = _make_jittered_xs(0, len(fid_points))
            ax1.scatter(xs, fid_points, color=light_green, s=28, alpha=0.8, zorder=4)
        if succ_points:
            xs = _make_jittered_xs(0, len(succ_points))
            ax2.scatter(xs, succ_points, color=light_red, s=28, alpha=0.8, zorder=4)
        # Control overall markers
        control_vals_flat = [v for lst in control_plateau_fids.values() for v in lst]
        if control_vals_flat:
            overall_fid = float(np.nanmean(control_vals_flat)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(control_vals_flat) >= max_fidelity) / len(control_vals_flat)
            ax1.scatter(
                [0], [overall_fid], color="green", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=5
            )
            ax2.scatter([0], [overall_succ], color="red", marker="s", edgecolors="blue", linewidths=0.5, s=60, zorder=5)
            t1 = ax1.text(
                0 + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            t2 = ax2.text(
                0 + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="blue",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(0)
        base_labels.append("Control")
        tries_counts.append(control_tries)

    # Runs
    for run_idx in range(1, n_runs + 1):
        plateau_fids = collect_fidelities_by_plateau_for_run(base_path, run_idx)
        tries = sum(len(v) for v in plateau_fids.values()) if plateau_fids else 0
        fid_points = []
        succ_points = []
        for pid in plateau_ids:
            vals = plateau_fids.get(pid, [])
            if not vals:
                continue
            avg_f = float(np.nanmean(vals)) * 100.0
            succ = 100.0 * np.sum(np.array(vals) >= max_fidelity) / len(vals)
            series_fid[pid][run_idx] = avg_f / 100.0
            series_succ[pid][run_idx] = succ
            fid_points.append(avg_f)
            succ_points.append(succ)
        if fid_points:
            xs = _make_jittered_xs(run_idx, len(fid_points))
            ax1.scatter(xs, fid_points, color=light_green, s=28, alpha=0.8, zorder=4)
        if succ_points:
            xs = _make_jittered_xs(run_idx, len(succ_points))
            ax2.scatter(xs, succ_points, color=light_red, s=28, alpha=0.8, zorder=4)
        # Overall run markers + tags
        all_vals = [v for lst in plateau_fids.values() for v in lst]
        if all_vals:
            overall_fid = float(np.nanmean(all_vals)) * 100.0
            overall_succ = 100.0 * np.sum(np.array(all_vals) >= max_fidelity) / len(all_vals)
            ax1.scatter([run_idx], [overall_fid], color="green", edgecolors="black", linewidths=0.5, s=60, zorder=5)
            ax2.scatter([run_idx], [overall_succ], color="red", marker="D", s=55, zorder=5)
            t1 = ax1.text(
                run_idx + 0.1,
                overall_fid,
                f"{overall_fid:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="green",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t1.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
            t2 = ax2.text(
                run_idx + 0.1,
                overall_succ,
                f"{overall_succ:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
                zorder=100,
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.6},
            )
            t2.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
        x_ticks.append(run_idx)
        base_labels.append(_base_label_for_run(run_idx, run_names))
        tries_counts.append(tries)

    # Connect plateau lines (same light colors)
    for pid, s in series_fid.items():
        if len(s) > 1:
            xs = sorted(s.keys())
            ys = [s[i] * 100.0 for i in xs]
            ax1.plot(xs, ys, "--", color=light_green, alpha=0.5, linewidth=1, zorder=1)
    for pid, s in series_succ.items():
        if len(s) > 1:
            xs = sorted(s.keys())
            ys = [s[i] for i in xs]
            ax2.plot(xs, ys, "--", color=light_red, alpha=0.5, linewidth=1, zorder=1)

    # Labels and layout
    ax1.set_ylabel("Average Best Fidelity (%)")
    ax2.set_ylabel("Success Rate (%)")
    ax1.set_xlabel(x_label, labelpad=22)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(base_labels)
    _draw_tries_sublabels(ax1, x_ticks, tries_counts, fontsize=8)
    fig.subplots_adjust(bottom=0.26)

    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color="C0", linestyle="-", label=f"Max fidelity ({int(max_fidelity * 100)}%)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Same plateau"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=light_green,
            markersize=6,
            linestyle="None",
            label="Plateau Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=light_red,
            markersize=6,
            linestyle="None",
            label="Plateau Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
            linestyle="None",
            label="Run Avg Best Fidelity (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="red",
            markersize=7,
            linestyle="None",
            label="Run Success Rate (%)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=None,
            markeredgecolor="blue",
            markersize=7,
            linestyle="None",
            label="Control (both)",
        ),
    ]
    ax1.legend(handles=handles, loc="best")

    save_path = os.path.join(base_path, "scatter_plateau_avg_success_combined.png")
    fig.tight_layout()
    fig.savefig(save_path)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


def find_if_common_initial_plateaus(base_path):
    """
    Check if the folder structure indicates common initial plateaus.
    Returns True if 'initial_plateau_X' folders are found, False otherwise.
    """
    # Auto-detect whether this run used common initial plateaus
    try:
        plateau_dirs = [
            d
            for d in os.listdir(base_path)
            if d.startswith("initial_plateau_") and os.path.isdir(os.path.join(base_path, d))
        ]
        return len(plateau_dirs) > 0
    except FileNotFoundError:
        return False
    
# ------- GRADIENT TRAJECTORY PLOTS -----------------------------------------------------
def _find_grad_runs(base_path: str, n_runs: int, common_initial_plateaus: bool):
    """Walk folder structure, return {run_idx: [(label, grad_path, run_dir), ...]}."""
    # Initialize a dictionary with empty lists from 1 to n_runs
    runs: dict[int, list[tuple[str, str, str]]] = {i: [] for i in range(1, n_runs + 1)}

    if common_initial_plateaus:
      # Match paths ending in: initial_plateau_X/repeated_changed_runY/Z
      # X (plateau), Y (run_idx), and Z (rep)
        pat = re.compile(
            r"initial_plateau_(\d+)[/\\]repeated_changed_run(\d+)[/\\](\d+)$"
        )
        for root, _, files in os.walk(base_path):
            # Skip this directory if it doesn't contain the gradient history file
            if "grad_history.npy" not in files:
                continue
            # Check if the current folder path matches our pattern
            m = pat.search(root)
            if not m:
                continue
            # Extract the captured groups and convert them to integers
            plateau, run_idx, rep = int(m[1]), int(m[2]), int(m[3])
            # If the extracted run_idx is within our expected range, save its details
            if run_idx in runs:
                runs[run_idx].append(
                    (f"plateau{plateau}_rep{rep}",                # Custom label for the plot
                     os.path.join(root, "grad_history.npy"),      # Full path to the file
                     root))                                       # The directory containing the file
    else:
        #  Match paths ending in: experimentX/Y
        #  X (run_idx) and Y (rep)  
        pat = re.compile(r"experiment(\d+)[/\\](\d+)$")
        # Same as before
        for root, _, files in os.walk(base_path):
            if "grad_history.npy" not in files:
                continue
            m = pat.search(root)
            if not m:
                continue
            run_idx, rep = int(m[1]), int(m[2])
            if run_idx in runs:
                runs[run_idx].append(
                    (f"rep{rep}", os.path.join(root, "grad_history.npy"), root))

    return runs

def plot_grad_trajectory(base_path, log_path, n_runs, common_initial_plateaus=False,
                          run_names=None):
    """Per-run detail plot: one figure per run, all repetitions overlaid.

    Each curve is Var_theta[dL/dtheta] across ALL params at each iteration.
    """
    # Locate all the gradient history files grouped by their run index
    runs = _find_grad_runs(base_path, n_runs, common_initial_plateaus)
    
    # Iterate over each distinct run (experiment configuration)
    for run_idx, entries in runs.items():
        if not entries:
            # Skip if there are no gradient files found for this run
            continue
        # Initialize a new matplotlib figure and axis for this specific run
        fig, ax = plt.subplots(figsize=(8, 5))
        ax_f = ax.twinx()  # right axis for fidelity
        cmap = plt.cm.tab10

        for rep_i, (label, grad_path, run_dir) in enumerate(entries):
            G = np.load(grad_path)
            var_t = np.var(G, axis=1)
            color = cmap(rep_i % 10)
            ax.plot(np.arange(G.shape[0]), var_t,
                    color=color, linewidth=1.3, alpha=0.85, label=label)
            # Fidelity overlay on right axis (thinner, dashed, same color)
            fids = _load_fidelity_curve(run_dir)
            if fids is not None:
                ax_f.plot(np.arange(fids.size), fids,
                          color=color, linewidth=0.8, alpha=0.6, 
                          linestyle="--", label=f"{label} (Fid)")

        ax.set_yscale("log")
        ax.set_xlabel("training iteration")
        ax.set_ylabel(r"Var$_{\theta}[\partial L/\partial \theta]$")
        ax_f.set_ylabel("Fidelity")
        ax_f.set_ylim(0, 1.02)
        run_label = _base_label_for_run(run_idx, run_names)
        ax.set_title(f"Gradient variance & fidelity — {run_label}")
        ax.grid(alpha=0.3)
        # Combine legends: rep entries + one fidelity proxy
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_f.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc='best')

        fig.tight_layout()
        save_path = os.path.join(base_path, f"grad_trajectory_run{run_idx}.png")
        fig.savefig(save_path, dpi=120)
        print_and_log(f"Saved plot to {save_path}", log_path)
        plt.close(fig)

# Joined TRAJECTORIES: initial plateau + changed runs

def _find_initial_plateau_grads(base_path: str) -> dict[int, str]:
    """Return {plateau_id: grad_history.npy path} for initial plateau runs."""
    out: dict[int, str] = {}
    pat = re.compile(r"initial_plateau_(\d+)$")
    for root, _, files in os.walk(base_path):
        if "grad_history.npy" not in files:
            continue
        m = pat.search(root)
        if m:
            out[int(m.group(1))] = os.path.join(root, "grad_history.npy")
    return out


def _find_control_grads_by_plateau(base_path: str) -> dict[int, list[str]]:
    """Return {plateau_id: [grad_history paths]} for repeated_control runs."""
    out: dict[int, list[str]] = {}
    pat = re.compile(r"initial_plateau_(\d+)[/\\]repeated_control")
    for root, _, files in os.walk(base_path):
        if "grad_history.npy" not in files:
            continue
        m = pat.search(root)
        if m:
            out.setdefault(int(m.group(1)), []).append(
                os.path.join(root, "grad_history.npy"))
    return out


def _find_changed_grads_by_plateau(base_path: str, run_idx: int) -> dict[int, list[str]]:
    """Return {plateau_id: [grad_history paths]} for repeated_changed_runN."""
    out: dict[int, list[str]] = {}
    pat = re.compile(
        rf"initial_plateau_(\d+)[/\\]repeated_changed_run{run_idx}[/\\](\d+)$"
    )
    for root, _, files in os.walk(base_path):
        if "grad_history.npy" not in files:
            continue
        m = pat.search(root)
        if m:
            out.setdefault(int(m.group(1)), []).append(
                os.path.join(root, "grad_history.npy"))
    return out


def _build_Joined_trajectory(phase1_path: str, phase2_path: str) -> np.ndarray:
    """Concatenate Initial Plateau and Changed Run gradient histories into one variance curve.

    Changed Run may have a different number of params than Initial Plateau (when ancilla is
    added). We compute var-across-params per iteration independently for each
    phase, which is well-defined regardless of param count.

    Returns a 1D array of shape (T1 + T2,).
    """
    G1 = np.load(phase1_path)          # (T1, n1)
    G2 = np.load(phase2_path)          # (T2, n2), n2 may differ from n1
    v1 = np.var(G1, axis=1)            # (T1,)
    v2 = np.var(G2, axis=1)            # (T2,)
    return np.concatenate([v1, v2])


def _collect_Joined_per_config(base_path: str, n_runs: int):
    """Build Joined trajectories per (config_label, plateau_id, rep).

    Returns:
        Joined: dict[str, list[np.ndarray]] — label -> list of trajectories
        insert_iter: int or None — iteration index where Changed Run starts.
                     Assumes Initial Plateau length is constant across plateaus.
    """
    plateau_grads = _find_initial_plateau_grads(base_path)
    if not plateau_grads:
        return {}, None

    # -- control ---------------
    Joined: dict[str, list[np.ndarray]] = {}
    ctrl = _find_control_grads_by_plateau(base_path)
    ctrl_curves = []
    for pid, paths in ctrl.items():
        if pid not in plateau_grads:
            continue
        for p2 in paths:
            ctrl_curves.append(_build_Joined_trajectory(plateau_grads[pid], p2))
    if ctrl_curves:
        Joined["Control"] = ctrl_curves

    # -- each changed run ------------------
    for run_idx in range(1, n_runs + 1):
        run_paths = _find_changed_grads_by_plateau(base_path, run_idx)
        run_curves = []
        for pid, paths in run_paths.items():
            if pid not in plateau_grads:
                continue
            for p2 in paths:
                run_curves.append(_build_Joined_trajectory(plateau_grads[pid], p2))
        if run_curves:
            Joined[f"Run {run_idx}"] = run_curves

    # -- insertion iteration (length of Initial Plateau) -----------------
    # Pick any initial plateau path to get T1
    any_p1 = next(iter(plateau_grads.values()))
    insert_iter = np.load(any_p1).shape[0]
    return Joined, insert_iter


def plot_grad_joined_all(base_path, log_path, n_runs,
                            common_initial_plateaus=False, run_names=None):
    """Plot A: every Joined trajectory as a thin transparent line."""
    if not common_initial_plateaus:
        return
    Joined, insert_iter = _collect_Joined_per_config(base_path, n_runs)
    if not Joined:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.tab10
    colors: dict[str, tuple] = {}
    color_idx = 0
    for label in Joined.keys():
        # Use run_names for display if available
        if label.startswith("Run "):
            run_idx = int(label.split()[1])
            display = _base_label_for_run(run_idx, run_names)
        else:
            display = label
        colors[label] = (cmap(color_idx % 10), display)
        color_idx += 1

    for label, curves in Joined.items():
        color, display = colors[label]
        for i, curve in enumerate(curves):
            ax.plot(np.arange(curve.size), curve, color=color,
                    linewidth=0.6, alpha=0.3,
                    label=display if i == 0 else None)

    if insert_iter is not None:
        ax.axvline(insert_iter, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"Ancilla insertion (iter {insert_iter})")

    ax.set_yscale("log")
    ax.set_xlabel("training iteration (Initial Plateau + Changed Run)")
    ax.set_ylabel(r"Var$_{\theta}[\partial L/\partial \theta]$")
    ax.set_title("Gradient variance — all Joined trajectories")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_path = os.path.join(base_path, "grad_Joined_all.png")
    fig.savefig(save_path, dpi=120)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)


def plot_grad_joined_mean(base_path, log_path, n_runs,
                             common_initial_plateaus=False, run_names=None):
    """Plot B: mean Joined trajectory per config (one line each).
    + a zoomed version around the insertion point.
    """
    if not common_initial_plateaus:
        return
    Joined, insert_iter = _collect_Joined_per_config(base_path, n_runs)
    if not Joined:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cmap = plt.cm.tab10
    color_idx = 0
    # Cache mean curves for the zoom plot
    mean_curves: dict[str, tuple[np.ndarray, tuple, str]] = {}

    
    for label, curves in Joined.items():
        # Pad shorter curves with NaN so early-exit runs don't truncate the mean.
        # at each iteration.
        max_T = max(c.size for c in curves)
        V = np.full((len(curves), max_T), np.nan)
        for i, c in enumerate(curves):
            V[i, :c.size] = c

        mean_v = np.nanmean(V, axis=0)
        n_active = np.sum(~np.isnan(V), axis=0)  # reps alive per iter

        if label.startswith("Run "):
            run_idx = int(label.split()[1])
            display = _base_label_for_run(run_idx, run_names)
        else:
            display = label
        color = cmap(color_idx % 10)
        color_idx += 1

        ax.plot(np.arange(max_T), mean_v, color=color, linewidth=2.0,
                label=f"{display} — mean (n={V.shape[0]})")
        mean_curves[label] = (mean_v, n_active, color, display)

    if insert_iter is not None:
        ax.axvline(insert_iter, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"Ancilla insertion (iter {insert_iter})")

    ax.set_yscale("log")
    ax.set_xlabel("training iteration (Initial Plateau + Changed Run)")
    ax.set_ylabel(r"Var$_{\theta}[\partial L/\partial \theta]$  (mean over reps)")
    ax.set_title("Gradient variance — mean Joined trajectory per config")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    save_path = os.path.join(base_path, "grad_Joined_mean.png")
    fig.savefig(save_path, dpi=120)
    print_and_log(f"Saved plot to {save_path}", log_path)
    plt.close(fig)

    # -- Zoomed view around insertion --------------------------------------
    if insert_iter is None:
        return
    ZOOM = 100  # iterations after insertion
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    zoom_end = insert_iter + ZOOM
    for label, (mean_v, n_active, color, display) in mean_curves.items():
        end = min(zoom_end, mean_v.size)
        start = max(0, insert_iter - 10)
        ax2.plot(np.arange(start, end), mean_v[start:end],
                 color=color, linewidth=2.0, label=f"{display} — mean")
    ax2.axvline(insert_iter, color="black", linestyle=":", linewidth=1.2,
                alpha=0.7, label=f"Ancilla insertion (iter {insert_iter})")
    ax2.set_yscale("log")
    ax2.set_xlabel("training iteration")
    ax2.set_ylabel(r"Var$_{\theta}[\partial L/\partial \theta]$")
    ax2.set_title(f"Zoom: {ZOOM} iterations after ancilla insertion")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="best", ncol=2)
    fig2.tight_layout()
    save_path2 = os.path.join(base_path, "grad_Joined_mean_zoom.png")
    fig2.savefig(save_path2, dpi=120)
    print_and_log(f"Saved plot to {save_path2}", log_path)
    plt.close(fig2)

    # -- n_active subplot: how many reps still training at each iter ------
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    for label, (mean_v, n_active, color, display) in mean_curves.items():
        ax3.plot(np.arange(n_active.size), n_active,
                 color=color, linewidth=1.8, label=display)
    if insert_iter is not None:
        ax3.axvline(insert_iter, color="black", linestyle=":", linewidth=1.2,
                    alpha=0.7)
    ax3.set_xlabel("training iteration")
    ax3.set_ylabel("reps still running")
    ax3.set_title("Number of reps contributing to the mean (reps exit when they converge)")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=9, loc="best")
    fig3.tight_layout()
    save_path3 = os.path.join(base_path, "grad_Joined_n_active.png")
    fig3.savefig(save_path3, dpi=120)
    print_and_log(f"Saved plot to {save_path3}", log_path)
    plt.close(fig3)

def _load_fidelity_curve(run_dir: str) -> np.ndarray | None:
    """Load the fidelity curve from <run_dir>/fidelities/log_fidelity_loss.txt.

    The file is saved as fidelities stacked on top of losses (single column,
    length 2*T). Returns only the fidelities (first half).
    """
    path = os.path.join(run_dir, "fidelities", "log_fidelity_loss.txt")
    if not os.path.exists(path):
        return None
    try:
        data = np.loadtxt(path)
        if data.ndim != 1 or data.size % 2 != 0:
            return None
        return data[: data.size // 2]
    except (OSError, IOError, ValueError):
        return None
    

# -- Create a specific BP plot
def plot_grad_trajectory_by_plateau(base_path, log_path, n_runs, plateau_ids,
                                     run_names=None, include_control=True,
                                     include_initial=True):
    """Per-plateau detail plot: one figure per (plateau, run), all reps overlaid.

    For each plateau in `plateau_ids`, generates one figure per run (and optionally
    control) containing: gradient variance (log, left axis) + fidelity (right axis)
    for every repetition, plus optionally the initial plateau trajectory prepended.

    Args:
        plateau_ids: int or list[int]. Which initial_plateau_<X> to include.
        include_control: also generate a figure for the control (no change) reps.
        include_initial: prepend the initial_plateau_<X> trajectory to each rep
            (concatenated), with a vertical line marking the insertion iter.
    """
    if isinstance(plateau_ids, int):
        plateau_ids = [plateau_ids]

    plateau_grads = _find_initial_plateau_grads(base_path)  # {pid: grad_path}

    for pid in plateau_ids:
        if pid not in plateau_grads:
            print_and_log(f"[grad_by_plateau] plateau {pid} not found, skipping", log_path)
            continue

        init_grad_path = plateau_grads[pid]
        init_run_dir = os.path.dirname(init_grad_path)

        # Build list of (config_label, rep_dirs) to plot, one figure each
        configs: list[tuple[str, list[str]]] = []
        if include_control:
            ctrl = _find_control_grads_by_plateau(base_path).get(pid, [])
            ctrl_dirs = [os.path.dirname(p) for p in ctrl]
            if ctrl_dirs:
                configs.append(("Control", ctrl_dirs))
        for run_idx in range(1, n_runs + 1):
            run_paths = _find_changed_grads_by_plateau(base_path, run_idx).get(pid, [])
            run_dirs = [os.path.dirname(p) for p in run_paths]
            if run_dirs:
                label = _base_label_for_run(run_idx, run_names)
                configs.append((label, run_dirs))

        if not configs:
            continue

        for cfg_label, rep_dirs in configs:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax_f = ax.twinx()
            cmap = plt.cm.tab10

            for rep_i, rep_dir in enumerate(rep_dirs):
                color = cmap(rep_i % 10)
                # --- variance ---
                v_rep = np.var(np.load(os.path.join(rep_dir, "grad_history.npy")), axis=1)
                if include_initial:
                    v_init = np.var(np.load(init_grad_path), axis=1)
                    var_curve = np.concatenate([v_init, v_rep])
                else:
                    var_curve = v_rep
                ax.plot(np.arange(var_curve.size), var_curve,
                        color=color, linewidth=1.3, alpha=0.85,
                        label=f"rep{rep_i}")

                # --- fidelity ---
                f_rep = _load_fidelity_curve(rep_dir)
                if include_initial:
                    f_init = _load_fidelity_curve(init_run_dir)
                    parts = [x for x in (f_init, f_rep) if x is not None]
                    fid_curve = np.concatenate(parts) if parts else None
                else:
                    fid_curve = f_rep
                if fid_curve is not None:
                    ax_f.plot(np.arange(fid_curve.size), fid_curve,
                              color=color, linewidth=0.8, alpha=0.6,
                              linestyle="--", label=f"rep{rep_i} (Fid)")

            # insertion line
            if include_initial:
                insert_iter = np.load(init_grad_path).shape[0]
                ax.axvline(insert_iter, color="black", linestyle=":",
                           linewidth=1.2, alpha=0.7,
                           label=f"Ancilla insertion (iter {insert_iter})")

            ax.set_yscale("log")
            ax.set_xlabel("training iteration")
            ax.set_ylabel(r"Var$_{\theta}[\partial L/\partial \theta]$")
            ax_f.set_ylabel("Fidelity")
            ax_f.set_ylim(0, 1.02)
            ax.set_title(f"Plateau {pid} — {cfg_label}")
            ax.grid(alpha=0.3)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_f.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="best")
            fig.tight_layout()

            # sanitize label for filename
            safe_label = re.sub(r"[^A-Za-z0-9_\-]+", "_", cfg_label)
            save_path = os.path.join(
                base_path, f"grad_trajectory_plateau{pid}_{safe_label}.png"
            )
            fig.savefig(save_path, dpi=120)
            print_and_log(f"Saved plot to {save_path}", log_path)
            plt.close(fig)
