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
"""Decile migration tables and heatmaps for CRN training output.

Outputs, next to the manifest:
  - `decile_migration.txt`             : tables + per-architecture % summary.
  - `decile_migration_counts.png`      : heatmaps (one per architecture) of
                                         seed counts, rows = D_ref, cols = D_actual.
  - `decile_migration_fractions.png`   : same but row-normalized (each row sums
                                         to 1) so architectures are directly
                                         comparable.

`no_ancilla` is included as a sanity check: its heatmap should be exactly
diagonal
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- EDIT HERE -----------------------------------------------------
VARIANCE_TIMESTAMP = "Try_3_C_ZZZ"
MANIFEST_PATH = os.path.join(
    "generated_data", "decile_seeds", VARIANCE_TIMESTAMP, "manifest.csv"
)

# Column order in the output tables (left to right).
CONFIGS_ORDER = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]

# Total number of deciles. Keep this in sync with N_DECILES in
N_DECILES = 10
# --------------------------------------------------------------------


def _resolve_manifest_path(p: str) -> str:
    """Resolve MANIFEST_PATH against either CWD or the project root.

    Lets the script be launched from anywhere (the project root, or from
    `src/`, etc.) without breaking. We try CWD first; if not found, fall
    back to the project root inferred from this file's location.
    """
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    # Fallback: relative to project root (two levels up from this file
    # if it lives in src/variance/, one level up if in src/, ...).
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        cand = os.path.join(up, p)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not locate manifest at: {p}")


def load_manifest(path: str) -> list[dict]:
    """Load manifest.csv as a list of row-dicts, skipping failed rows.

    We only count seeds with status == 'ok'. Rows missing decile fields
    or with non-integer values are dropped with a warning.
    """
    rows: list[dict] = []
    dropped = 0
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                # We could choose to count failed runs too, but their
                # decile_actual is still a valid measurement; flip this
                # if you want them in the tables.
                continue
            try:
                r["decile_ref"] = int(r["decile_ref"])
                r["decile_actual"] = int(r["decile_actual"])
                r["seed_id"] = int(r["seed_id"])
            except (KeyError, ValueError, TypeError):
                dropped += 1
                continue
            rows.append(r)
    if dropped:
        print(f"[warn] dropped {dropped} rows with bad/missing decile fields")
    return rows


def build_counts(rows: list[dict]) -> dict:
    """Build counts[ref_decile][config][actual_decile] = N.

    Uses nested defaultdicts so missing combinations naturally read as 0.
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        counts[r["decile_ref"]][r["config"]][r["decile_actual"]] += 1
    return counts


def _format_table(
    ref_decile: int,
    counts_for_ref: dict,
    configs_present: list[str],
    n_deciles: int,
) -> str:
    """Build the count table for one ref decile as a plain-text block.
    """
    # Pick a config-column width that fits all labels.
    cfg_w = max(len("config"), max(len(c) for c in configs_present))
    cell_w = 5  # fits 3-digit counts plus padding
    sep = "  "

    header = f"{'config':<{cfg_w}}{sep}"
    header += sep.join(f"{'D'+str(d):>{cell_w}}" for d in range(1, n_deciles + 1))
    header += f"{sep}|{sep}{'Total':>{cell_w}}"

    lines = [
        f"=== Reference decile D{ref_decile} "
        f"(seeds chosen from no_ancilla's D{ref_decile}) ===",
        header,
        "-" * len(header),
    ]
    for cfg in configs_present:
        row = f"{cfg:<{cfg_w}}{sep}"
        per_d = counts_for_ref.get(cfg, {})
        total = sum(per_d.values())
        row += sep.join(f"{per_d.get(d, 0):>{cell_w}d}" for d in range(1, n_deciles + 1))
        row += f"{sep}|{sep}{total:>{cell_w}d}"
        lines.append(row)
    return "\n".join(lines)


def _format_summary(
    ref_decile: int,
    counts_for_ref: dict,
    configs_present: list[str],
    n_deciles: int,
) -> str:
    """Per-architecture % summary for one ref decile.

    For each architecture, list the decils where seeds landed with their
    percentages, sorted by descending count. Skips deciles with 0 seeds.
    """
    lines = [f"--- Summary D{ref_decile} (as % of seeds per architecture) ---"]
    for cfg in configs_present:
        per_d = counts_for_ref.get(cfg, {})
        total = sum(per_d.values())
        if total == 0:
            lines.append(f"  {cfg}: (no seeds)")
            continue
        # Sort by count desc, then by decile asc as a tie-breaker.
        items = sorted(per_d.items(), key=lambda kv: (-kv[1], kv[0]))
        parts = [
            f"D{d}: {n} ({100.0 * n / total:.1f}%)"
            for d, n in items
            if n > 0
        ]
        lines.append(f"  {cfg} (n={total}): " + ", ".join(parts))
    return "\n".join(lines)


def write_report(rows: list[dict], counts: dict, out_path: str) -> None:
    """Write the full multi-table report to out_path.

    Discovers reference deciles from the data (not hardcoded) so changing
    DECILES in decile_training_crn.py doesn't require editing here.
    """
    # Reference deciles actually present in the manifest, in ascending order.
    ref_deciles = sorted(counts.keys())
    if not ref_deciles:
        raise RuntimeError("No reference deciles found in manifest.")

    # Configs in the user-specified order, but only those actually present.
    seen_configs = {r["config"] for r in rows}
    configs_present = [c for c in CONFIGS_ORDER if c in seen_configs]
    # Append any configs found in the CSV that weren't in CONFIGS_ORDER,
    # so we don't silently hide them.
    for c in sorted(seen_configs):
        if c not in configs_present:
            configs_present.append(c)

    n_rows = len(rows)
    blocks = [
        "Decile migration tables",
        f"Manifest:           {os.path.basename(out_path).replace('decile_migration.txt', 'manifest.csv')}",
        f"Total seeds (ok):   {n_rows}",
        f"Reference deciles:  {ref_deciles}",
        f"Architectures:      {configs_present}",
        f"N_DECILES (target): {N_DECILES}",
        "",
    ]
    for d in ref_deciles:
        blocks.append(_format_table(d, counts[d], configs_present, N_DECILES))
        blocks.append("")
        blocks.append(_format_summary(d, counts[d], configs_present, N_DECILES))
        blocks.append("")

    text = "\n".join(blocks).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)


# -- Migration heatmaps ------------------------------------------------
def _configs_present_from(rows: list[dict]) -> list[str]:
    """List of configs actually in `rows`, ordered per CONFIGS_ORDER first."""
    seen = {r["config"] for r in rows}
    out = [c for c in CONFIGS_ORDER if c in seen]
    for c in sorted(seen):
        if c not in out:
            out.append(c)
    return out


def _build_migration_matrices(
    counts: dict,
    configs_present: list[str],
    n_deciles: int,
) -> dict[str, np.ndarray]:
    """For each config, build a (n_ref, n_deciles) matrix of seed counts.

    Rows index reference deciles (ascending), columns index actual deciles
    (1..n_deciles). Reference deciles are taken from `counts.keys()` so we
    only plot rows that actually have data.
    """
    ref_deciles = sorted(counts.keys())
    matrices: dict[str, np.ndarray] = {}
    for cfg in configs_present:
        M = np.zeros((len(ref_deciles), n_deciles), dtype=int)
        for i, d_ref in enumerate(ref_deciles):
            per_d = counts[d_ref].get(cfg, {})
            for d_actual, n in per_d.items():
                if 1 <= d_actual <= n_deciles:
                    M[i, d_actual - 1] = n
        matrices[cfg] = M
    return matrices


def _annotate_cells(
    ax,
    M: np.ndarray,
    mode: str,
    color_threshold: float,
) -> None:
    """Write cell values inside the heatmap.

    For counts: integer; for fractions: percentage with one decimal.
    Cells with zero value are left blank to reduce visual noise.
    `color_threshold` is the value above which we flip the text to white
    so it stays legible on dark cells.
    """
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            if v <= 0:
                continue
            txt = f"{int(v):d}" if mode == "counts" else f"{100.0 * v:.1f}"
            color = "white" if v >= color_threshold else "black"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8, color=color,
            )


def _plot_migration_grid(
    matrices: dict[str, np.ndarray],
    ref_deciles: list[int],
    n_deciles: int,
    out_path: str,
    mode: str,
    title_suffix: str,
) -> None:
    """One PNG, one heatmap per architecture in a grid.

    Args:
        matrices: {config: M of shape (n_ref, n_deciles)} as built by
                  `_build_migration_matrices`.
        ref_deciles: list of reference decils corresponding to M's rows.
        n_deciles: width of each heatmap (1..n_deciles in columns).
        out_path: PNG path.
        mode: "counts" or "fractions".
        title_suffix: e.g. "(seed counts)" or "(row-normalized fractions)".
    """
    configs = list(matrices.keys())
    n = len(configs)

    # Grid layout: try 2 columns if we have >2 configs, otherwise 1 row.
    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = 2
        nrows = (n + ncols - 1) // ncols

    # Reserve space on the right for a single shared colorbar via gridspec,
    # otherwise tight_layout fights with `fig.colorbar(ax=...)` and the bar
    # ends up sitting on top of the heatmaps.
    fig = plt.figure(figsize=(6.5 * ncols + 1.2, 5.0 * nrows))
    gs = fig.add_gridspec(
        nrows, ncols + 1,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.30, hspace=0.35,
    )
    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(nrows) for c in range(ncols)]
    cax = fig.add_subplot(gs[:, ncols])

    # Decide global vmax so all panels share the color scale (essential to
    # compare architectures visually).
    if mode == "counts":
        vmax = max((M.max() for M in matrices.values()), default=1)
        vmin = 0
    else:  # fractions: always 0..1
        vmax = 1.0
        vmin = 0.0

    # Threshold for switching annotation color to white. 60% of vmax is a
    # robust default for both 'Blues' and similar sequential maps.
    color_threshold = 0.6 * vmax if vmax > 0 else 0.0

    cmap = plt.get_cmap("Blues")

    im = None
    for ax, cfg in zip(axes_flat, configs):
        M_raw = matrices[cfg]
        if mode == "fractions":
            row_sums = M_raw.sum(axis=1, keepdims=True)
            # Avoid divide-by-zero for empty ref rows
            M = np.where(row_sums > 0, M_raw / np.maximum(row_sums, 1), 0.0)
        else:
            M = M_raw.astype(float)

        im = ax.imshow(
            M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            origin="upper",
        )

        # Diagonal guide: D_actual == D_ref. Only meaningful for cells
        # whose row index corresponds to a ref decile present in the data.
        # Draw the line as the set of (j, i) with the same decile number.
        # In image coords, x = col index (D_actual - 1), y = row index in
        # ref_deciles. The diagonal passes through (d-1, ref_deciles.index(d))
        # for d in ref_deciles ∩ [1..n_deciles].
        diag_x, diag_y = [], []
        for i, d_ref in enumerate(ref_deciles):
            if 1 <= d_ref <= n_deciles:
                diag_x.append(d_ref - 1)
                diag_y.append(i)
        if diag_x:
            ax.plot(
                diag_x, diag_y,
                linestyle=":", linewidth=1.2, color="black", alpha=0.6,
            )

        _annotate_cells(ax, M, mode, color_threshold)

        ax.set_xticks(range(n_deciles))
        ax.set_xticklabels([f"D{d}" for d in range(1, n_deciles + 1)],
                           fontsize=8)
        ax.set_yticks(range(len(ref_deciles)))
        ax.set_yticklabels([f"D{d}" for d in ref_deciles], fontsize=8)
        ax.set_xlabel(r"$D_\mathrm{actual}$  (decile in this architecture)")
        ax.set_ylabel(r"$D_\mathrm{ref}$  (decile in no_ancilla)")
        ax.set_title(cfg, fontsize=11)

    # Hide leftover panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Single shared colorbar on the dedicated axis
    if im is not None:
        cbar = fig.colorbar(
            im, cax=cax,
            label=("seed count" if mode == "counts"
                   else "fraction of seeds per row"),
        )
        if mode == "fractions":
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        f"Decile migration {title_suffix}\n"
        f"rows = decile in no_ancilla, cols = decile in this architecture",
        y=0.995,
    )
    # Leave room for the suptitle; no tight_layout because the gridspec
    # already manages spacing between panels and the colorbar axis.
    fig.subplots_adjust(top=0.92, left=0.06, right=0.94, bottom=0.06)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def write_heatmaps(rows: list[dict], counts: dict, out_dir: str) -> None:
    """Build matrices and write the two heatmap PNGs next to the report."""
    configs_present = _configs_present_from(rows)
    matrices = _build_migration_matrices(counts, configs_present, N_DECILES)
    ref_deciles = sorted(counts.keys())

    counts_path = os.path.join(out_dir, "decile_migration_counts.png")
    _plot_migration_grid(
        matrices, ref_deciles, N_DECILES, counts_path,
        mode="counts",
        title_suffix="(seed counts)",
    )

    frac_path = os.path.join(out_dir, "decile_migration_fractions.png")
    _plot_migration_grid(
        matrices, ref_deciles, N_DECILES, frac_path,
        mode="fractions",
        title_suffix="(row-normalized fractions)",
    )


def main() -> None:
    manifest_path = _resolve_manifest_path(MANIFEST_PATH)
    print(f"[load] manifest: {manifest_path}")
    rows = load_manifest(manifest_path)
    if not rows:
        print("[error] manifest has no usable rows (status='ok' with valid deciles).")
        sys.exit(1)

    counts = build_counts(rows)

    # Output next to the manifest, as requested.
    out_dir = os.path.dirname(manifest_path)
    out_path = os.path.join(out_dir, "decile_migration.txt")
    write_report(rows, counts, out_path)
    print(f"[done] wrote {out_path}")

    # Companion heatmaps
    write_heatmaps(rows, counts, out_dir)


if __name__ == "__main__":
    main()