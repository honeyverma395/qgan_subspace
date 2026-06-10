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
  - `decile_migration.txt`                       : per-D_ref tables of seed
                                                   migration counts, %-summary,
                                                   mean/median ||g||^2 per
                                                   architecture, and per-seed
                                                   fidelity changes vs
                                                   no_ancilla (grouped into
                                                   WORSE / BETTER).
  - `decile_migration_counts.png`                : migration heatmaps in seed
                                                   counts (one panel per arch).
  - `decile_migration_fidelities_counts.png`     : fidelity-bin heatmaps in
                                                   seed counts. Rows = D_ref
                                                   (no_ancilla decile), cols =
                                                   final fidelity bin reached

`no_ancilla` is included as a sanity check in the migration heatmaps: its
heatmap should be exactly diagonal. In the fidelity heatmaps it shows how the
no_ancilla baseline trains for each starting decile.
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

# Where training output for each cell lives:
TRAINING_ROOT = os.path.join("generated_data", "decile_training")

# Column order in the output tables (left to right).
CONFIGS_ORDER = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]

# Total number of deciles
N_DECILES = 10

# Write the detailed decile_migration.txt report
WRITE_TXT_REPORT = False

# Fidelity bin edges (in [0, 1]). Bins are right-open except the last which is
# closed at 1.0 to include perfect fidelity. Keep edges strictly increasing.
FIDELITY_BIN_EDGES = [0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0]

def _fidelity_bin_labels(edges: list[float]) -> list[str]:
    """Human-readable labels like '0-50%', '50-60%', ..., '99-100%'."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(round(lo * 100))}-{int(round(hi * 100))}%")
    return labels

FIDELITY_BIN_LABELS = _fidelity_bin_labels(FIDELITY_BIN_EDGES)
# --------------------------------------------------------------------


# -- PATH HELPERS ------------------------------------------------------
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
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        cand = os.path.join(up, p)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not locate manifest at: {p}")


def _resolve_training_root(manifest_path: str) -> str:
    """Find the absolute path of the training root for this variance timestamp.

    The manifest sits at <project>/generated_data/decile_seeds/<VTS>/manifest.csv
    and the training outputs at <project>/generated_data/decile_training/<VTS>/.
    We anchor to the project root inferred from the manifest path (3 levels up
    from manifest.csv -> generated_data/decile_seeds/<VTS>/manifest.csv).
    """
    manifest_dir = os.path.dirname(manifest_path)               # .../decile_seeds/<VTS>
    decile_seeds_dir = os.path.dirname(manifest_dir)            # .../decile_seeds
    generated_data_dir = os.path.dirname(decile_seeds_dir)      # .../generated_data
    project_root = os.path.dirname(generated_data_dir)          # .../

    cand = os.path.join(project_root, TRAINING_ROOT, VARIANCE_TIMESTAMP)
    if os.path.isdir(cand):
        return cand
    # Fallback: relative to CWD
    cand2 = os.path.join(TRAINING_ROOT, VARIANCE_TIMESTAMP)
    if os.path.isdir(cand2):
        return os.path.abspath(cand2)
    return cand  # return the most-likely path even if missing; we'll warn later


# -- MANIFEST + FIDELITY LOADERS ---------------------------------
def load_manifest(path: str) -> list[dict]:
    """Load manifest.csv as a list of row-dicts, preserving row order.

    We keep ALL rows (status='ok' and otherwise) but tag each with its
    1-based row index, since `cell_idx = row_idx + 1` is how
    decile_training_crn.py maps manifest rows to experiment<N> folders.
    Rows with non-integer decile fields are dropped with a warning;
    these can't be placed on the heatmap at all.
    """
    rows: list[dict] = []
    dropped = 0
    with open(path, "r", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            r["_row_idx"] = i  # 0-based, used to derive cell_idx = i+1
            try:
                r["decile_ref"] = int(r["decile_ref"])
                r["decile_actual"] = int(r["decile_actual"])
                r["seed_id"] = int(r["seed_id"])
                r["rep"] = int(r["rep"])
                r["norm_sq"] = float(r["norm_sq"])
            except (KeyError, ValueError, TypeError):
                dropped += 1
                continue
            rows.append(r)
    if dropped:
        print(f"[warn] dropped {dropped} rows with bad/missing fields")
    return rows


def _max_fidelity_from_file(fid_loss_path: str) -> float | None:
    """Read the maximum fidelity reached across the whole training run.

    Mirrors get_max_fidelity_from_file in plot_hub.py. save_fidelity_loss
    writes fidelities followed by losses via np.savetxt (single column,
    length 2*T). We take the max of the first half.
    """
    if not os.path.exists(fid_loss_path):
        return None
    try:
        data = np.loadtxt(fid_loss_path)
        if data.ndim == 1:
            # Could be the stacked (fids, losses) format (even length) or just
            # fidelities. The plot_hub helper just maxes the whole thing when
            # 1D
            if data.size % 2 == 0 and data.size >= 2:
                fids = data[: data.size // 2]
            else:
                fids = data
        else:
            fids = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
        return float(np.max(fids))
    except (OSError, IOError, ValueError):
        return None


def attach_fidelities(rows: list[dict], training_root: str) -> int:
    """Attach `max_fidelity` to each manifest row (None if unreadable).

    Returns the number of rows for which a fidelity was successfully loaded.
    Mapping: manifest row index i (0-based) -> experiment<i+1>/<rep>/fidelities/
    log_fidelity_loss.txt. This matches the deterministic CELL_IDX in
    decile_training_crn.py since rows are written in the same enumeration
    order as CELL_ORDER.
    """
    n_ok = 0
    n_missing = 0
    for r in rows:
        cell_idx = r["_row_idx"] + 1
        rep = r["rep"]
        fid_path = os.path.join(
            training_root, f"experiment{cell_idx}", str(rep),
            "fidelities", "log_fidelity_loss.txt",
        )
        v = _max_fidelity_from_file(fid_path)
        r["max_fidelity"] = v
        if v is None:
            n_missing += 1
        else:
            n_ok += 1
    if n_missing:
        print(f"[warn] {n_missing} rows have no readable fidelity file "
              f"(searched under {training_root})")
    return n_ok


def fidelity_to_bin(fid: float, edges: list[float]) -> int | None:
    """Return the 0-based bin index of `fid` given the edges.

    Bins are [edges[i], edges[i+1]). The last bin includes the right edge
    (so fidelity = 1.0 lands in the final bin). Returns None if out of range.
    """
    if fid is None or not np.isfinite(fid):
        return None
    if fid < edges[0] or fid > edges[-1]:
        return None
    # np.searchsorted with side='right' gives the bin index such that
    # edges[idx-1] <= fid < edges[idx]; clamp the right edge case.
    idx = int(np.searchsorted(edges, fid, side="right")) - 1
    if idx == len(edges) - 1:
        idx -= 1  # fid == edges[-1] falls into the last bin
    return max(0, min(idx, len(edges) - 2))


# -- COUNT BUILDERS --------------------------------------------------------
def build_counts(rows: list[dict]) -> dict:
    """Build counts[ref_decile][config][actual_decile] = N.

    Only rows with status == 'ok' contribute (a failed training has no
    meaningful "post-training" measurement; the gradient-based decile_actual
    is still valid but mixing them would be confusing for the report).
    Uses nested defaultdicts so missing combinations naturally read as 0.
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        if r.get("status") != "ok":
            continue
        counts[r["decile_ref"]][r["config"]][r["decile_actual"]] += 1
    return counts


def build_fidelity_counts(rows: list[dict]) -> dict:
    """Build counts[ref_decile][config][fidelity_bin_idx] = N.

    Same scoping as build_counts: only status == 'ok' rows with a readable
    fidelity contribute. Rows whose fidelity falls outside FIDELITY_BIN_EDGES
    are skipped (shouldn't happen for [0, 1] fidelities with our edges).
    """
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        if r.get("status") != "ok":
            continue
        v = r.get("max_fidelity")
        if v is None:
            continue
        b = fidelity_to_bin(v, FIDELITY_BIN_EDGES)
        if b is None:
            continue
        counts[r["decile_ref"]][r["config"]][b] += 1
    return counts


def build_norm_stats(rows: list[dict]) -> dict:
    """Build stats[ref_decile][config] = {'mean': float, 'median': float, 'n': int}.

    Uses ||g||^2 (the manifest's `norm_sq` column). Only status == 'ok' rows
    contribute, to stay consistent with the count tables.
    """
    bucket: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("status") != "ok":
            continue
        bucket[r["decile_ref"]][r["config"]].append(r["norm_sq"])

    stats: dict = defaultdict(dict)
    for d_ref, cfg_map in bucket.items():
        for cfg, vals in cfg_map.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            stats[d_ref][cfg] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "n": int(arr.size),
            }
    return stats


# -- TEXT REPORT ---------------------------------------------------------
def _format_migration_table(
    ref_decile: int,
    counts_for_ref: dict,
    configs_present: list[str],
    n_deciles: int,
) -> str:
    """Build the count table for one ref decile as a plain-text block."""
    cfg_w = max(len("config"), max(len(c) for c in configs_present))
    cell_w = 5
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
    """Per-architecture % summary for one ref decile."""
    lines = [f"--- Summary D{ref_decile} (as % of seeds per architecture) ---"]
    for cfg in configs_present:
        per_d = counts_for_ref.get(cfg, {})
        total = sum(per_d.values())
        if total == 0:
            lines.append(f"  {cfg}: (no seeds)")
            continue
        items = sorted(per_d.items(), key=lambda kv: (-kv[1], kv[0]))
        parts = [
            f"D{d}: {n} ({100.0 * n / total:.1f}%)"
            for d, n in items
            if n > 0
        ]
        lines.append(f"  {cfg} (n={total}): " + ", ".join(parts))
    return "\n".join(lines)


def _format_norm_stats_table(
    ref_decile: int,
    stats_for_ref: dict,
    configs_present: list[str],
) -> str:
    """Per-architecture mean & median ||g||^2 for one ref decile.

    Plain text, lines up with the count table aesthetics. Scientific notation
    because ||g||^2 spans many orders of magnitude in BP-prone regions.
    """
    cfg_w = max(len("config"), max(len(c) for c in configs_present))
    cell_w = 14
    sep = "  "
    header = (
        f"{'config':<{cfg_w}}{sep}"
        f"{'n':>5}{sep}"
        f"{'mean(||g||^2)':>{cell_w}}{sep}"
        f"{'median(||g||^2)':>{cell_w + 1}}"
    )
    lines = [
        f"--- Mean / median ||g||^2 at D{ref_decile} (per architecture) ---",
        header,
        "-" * len(header),
    ]
    for cfg in configs_present:
        s = stats_for_ref.get(cfg)
        if not s:
            lines.append(f"{cfg:<{cfg_w}}{sep}{'-':>5}{sep}"
                         f"{'(no seeds)':>{cell_w}}{sep}"
                         f"{'(no seeds)':>{cell_w + 1}}")
            continue
        lines.append(
            f"{cfg:<{cfg_w}}{sep}"
            f"{s['n']:>5d}{sep}"
            f"{s['mean']:>{cell_w}.4e}{sep}"
            f"{s['median']:>{cell_w + 1}.4e}"
        )
    return "\n".join(lines)


# -- Fidelity changes (per-seed, ancilla vs no_ancilla) ----------------
def _fidelity_bin_label_for(fid: float | None) -> str:
    """Return the bin label '0-50%' / '50-60%' / ... for a fidelity value.

    Returns '(missing)' when `fid` is None and '(out-of-range)' when it falls
    outside the bin edges (shouldn't happen for valid fidelities in [0, 1],
    but we guard against numerical surprises).
    """
    if fid is None:
        return "(missing)"
    b = fidelity_to_bin(fid, FIDELITY_BIN_EDGES)
    if b is None:
        return "(out-of-range)"
    return FIDELITY_BIN_LABELS[b]


def _format_fidelity_changes(
    rows: list[dict],
    ref_decile: int,
    configs_present: list[str],
    baseline_config: str = "no_ancilla",
) -> str:
    """Per-seed fidelity changes for one D_ref, grouped by config.

    For each seed at this D_ref, compares its post-training fidelity bin in
    each ancilla config against the same seed's bin in `baseline_config`.
    Seeds whose bin is unchanged are silent (no line written). Seeds with a
    changed bin are listed and grouped into WORSE / BETTER blocks.

    Notes on the displayed deciles:
      - The baseline (`no_ancilla`) decile_actual is always equal to D_ref by
        construction (seeds were selected from no_ancilla's D_ref). We don't
        repeat that information per seed; it's stated in the section header.
      - For each ancilla config we DO print decile_actual, since seeds
        migrate across deciles when re-ranked by ||g||^2 of that
        architecture, and the migration is informative.
    """
    # Index baseline rows by seed_id for fast lookup. We keep only baseline
    # rows belonging to THIS D_ref (defensive — a seed_id is a global index
    # so in principle the baseline shouldn't contain that seed under a
    # different D_ref, but checking is cheap and avoids surprises).
    baseline_by_seed: dict[int, dict] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r["config"] != baseline_config:
            continue
        if r["decile_ref"] != ref_decile:
            continue
        baseline_by_seed[r["seed_id"]] = r

    # Configs to compare against baseline: everything in configs_present
    # except the baseline itself, in the user-specified order.
    other_configs = [c for c in configs_present if c != baseline_config]

    lines: list[str] = [
        f"--- Fidelity changes at D{ref_decile}  "
        f"(seeds from {baseline_config}'s D{ref_decile}) ---",
        f"  Baseline:   {baseline_config}, D_actual = D{ref_decile} by construction.",
        f"  Comparison: each ancilla config below shows seeds whose fidelity",
        f"              bin DIFFERS from the baseline. Same-bin seeds are",
        f"              omitted. Within each config block, seeds are grouped",
        f"              into WORSE (lower bin in ancilla) and BETTER (higher",
        f"              bin in ancilla).",
        "",
    ]

    for cfg in other_configs:
        # Build the diff list for this config
        worse: list[dict] = []
        better: list[dict] = []
        n_same = 0
        n_unmatched = 0

        for r in rows:
            if r.get("status") != "ok":
                continue
            if r["config"] != cfg:
                continue
            if r["decile_ref"] != ref_decile:
                continue
            base = baseline_by_seed.get(r["seed_id"])
            if base is None:
                # Seed has no baseline counterpart (baseline run missing
                # or failed). Can't compare, skip it.
                n_unmatched += 1
                continue

            f_base = base.get("max_fidelity")
            f_anc = r.get("max_fidelity")
            b_base = fidelity_to_bin(f_base, FIDELITY_BIN_EDGES) if f_base is not None else None
            b_anc = fidelity_to_bin(f_anc, FIDELITY_BIN_EDGES) if f_anc is not None else None

            # Skip seeds where either side has no readable fidelity: we can't
            # tell whether the bin changed. (We could classify these as
            # "(missing)" vs a real bin, but that conflates "no data" with
            # "real change" and inflates the WORSE list.)
            if b_base is None or b_anc is None:
                n_unmatched += 1
                continue

            if b_base == b_anc:
                n_same += 1
                continue

            entry = {
                "seed_id": r["seed_id"],
                "f_base": f_base,
                "b_base": b_base,
                "f_anc": f_anc,
                "b_anc": b_anc,
                "d_actual_anc": r["decile_actual"],
            }
            if b_anc < b_base:
                worse.append(entry)
            else:
                better.append(entry)

        # Sort within each group: by seed_id ascending (deterministic, easy
        # to cross-reference back to the manifest).
        worse.sort(key=lambda e: e["seed_id"])
        better.sort(key=lambda e: e["seed_id"])

        n_changed = len(worse) + len(better)
        n_total = n_same + n_changed + n_unmatched
        header = (
            f"=== {cfg} vs {baseline_config} ===  "
            f"(changed: {n_changed}, same-bin: {n_same}"
            f"{', no-match: ' + str(n_unmatched) if n_unmatched else ''}, "
            f"total: {n_total})"
        )
        lines.append(header)

        if n_changed == 0:
            lines.append("  (no seeds changed bin)")
            lines.append("")
            continue

        # WORSE block
        lines.append(f"  WORSE  ({len(worse)} seeds):")
        if worse:
            for e in worse:
                lines.append(_format_change_entry(e, baseline_config, cfg))
        else:
            lines.append("    (none)")

        # BETTER block
        lines.append(f"  BETTER ({len(better)} seeds):")
        if better:
            for e in better:
                lines.append(_format_change_entry(e, baseline_config, cfg))
        else:
            lines.append("    (none)")

        lines.append("")

    return "\n".join(lines).rstrip()


def _format_change_entry(entry: dict, baseline_cfg: str, ancilla_cfg: str) -> str:
    """One-line summary of a seed whose fidelity bin changed."""
    seed = entry["seed_id"]
    f_b = entry["f_base"]
    f_a = entry["f_anc"]
    bin_b = FIDELITY_BIN_LABELS[entry["b_base"]]
    bin_a = FIDELITY_BIN_LABELS[entry["b_anc"]]
    d_a = entry["d_actual_anc"]
    return (
        f"    seed {seed:>5d}:  "
        f"{baseline_cfg} F={f_b:.4f} ({bin_b})  ->  "
        f"{ancilla_cfg} F={f_a:.4f} ({bin_a}, D_actual=D{d_a})"
    )


def write_report(
    rows: list[dict],
    counts: dict,
    norm_stats: dict,
    out_path: str,
) -> None:
    """Write the full multi-table report to out_path."""
    ref_deciles = sorted(counts.keys())
    if not ref_deciles:
        raise RuntimeError("No reference deciles found in manifest.")

    seen_configs = {r["config"] for r in rows if r.get("status") == "ok"}
    configs_present = [c for c in CONFIGS_ORDER if c in seen_configs]
    for c in sorted(seen_configs):
        if c not in configs_present:
            configs_present.append(c)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_with_fid = sum(
        1 for r in rows
        if r.get("status") == "ok" and r.get("max_fidelity") is not None
    )
    blocks = [
        "Decile migration tables",
        f"Manifest:             {os.path.basename(out_path).replace('decile_migration.txt', 'manifest.csv')}",
        f"Total seeds (ok):     {n_ok}",
        f"Seeds w/ fidelity:    {n_with_fid}",
        f"Reference deciles:    {ref_deciles}",
        f"Architectures:        {configs_present}",
        f"N_DECILES (target):   {N_DECILES}",
        f"Fidelity bin edges:   {FIDELITY_BIN_EDGES}",
        f"Baseline for fidelity changes: no_ancilla",
        "",
    ]
    for d in ref_deciles:
        blocks.append(_format_migration_table(d, counts[d], configs_present, N_DECILES))
        blocks.append("")
        blocks.append(_format_summary(d, counts[d], configs_present, N_DECILES))
        blocks.append("")
        blocks.append(_format_norm_stats_table(d, norm_stats.get(d, {}), configs_present))
        blocks.append("")
        blocks.append(_format_fidelity_changes(rows, d, configs_present))
        blocks.append("")

    text = "\n".join(blocks).rstrip() + "\n"
    with open(out_path, "w") as f:
        f.write(text)


# -- HEATMAP HELPERS (shared between migration and fidelity heatmaps) ---------
def _configs_present_from(rows: list[dict]) -> list[str]:
    """List of configs actually in `rows` (status='ok'), ordered per CONFIGS_ORDER first."""
    seen = {r["config"] for r in rows if r.get("status") == "ok"}
    out = [c for c in CONFIGS_ORDER if c in seen]
    for c in sorted(seen):
        if c not in out:
            out.append(c)
    return out


def _annotate_cells(
    ax,
    M: np.ndarray,
    mode: str,
    color_threshold: float,
) -> None:
    """Write cell values inside the heatmap.

    For counts: integer; for fractions: percentage with one decimal.
    Cells with zero value are left blank to reduce visual noise.
    """
    n_rows, n_cols = M.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            if v <= 0:
                continue
            txt = f"{int(v):d}"
            color = "white" if v >= color_threshold else "black"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=8, color=color,
            )



#-- MIGRATION HEATMAPS  (D_ref vs D_actual) ------------------------------
def _build_migration_matrices(
    counts: dict,
    configs_present: list[str],
    n_deciles: int,
) -> dict[str, np.ndarray]:
    """For each config, build a (n_ref, n_deciles) matrix of seed counts."""
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
        matrices: {config: M of shape (n_ref, n_deciles)}.
        ref_deciles: list of reference decils corresponding to M's rows.
        n_deciles: width of each heatmap (1..n_deciles in columns).
        out_path: PNG path.
        mode: "counts" or "fractions".
        title_suffix: e.g. "(seed counts)" or "(row-normalized fractions)".
    """
    configs = list(matrices.keys())
    n = len(configs)
    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = 2
        nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6.5 * ncols + 1.2, 5.0 * nrows))
    gs = fig.add_gridspec(
        nrows, ncols + 1,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.30, hspace=0.35,
    )
    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(nrows) for c in range(ncols)]
    cax = fig.add_subplot(gs[:, ncols])

    vmax = max((M.max() for M in matrices.values()), default=1)
    vmin = 0

    color_threshold = 0.6 * vmax if vmax > 0 else 0.0
    cmap = plt.get_cmap("Blues")

    im = None
    for ax, cfg in zip(axes_flat, configs):
        M_raw = matrices[cfg]
        if mode == "fractions":
            row_sums = M_raw.sum(axis=1, keepdims=True)
            M = np.where(row_sums > 0, M_raw / np.maximum(row_sums, 1), 0.0)
        else:
            M = M_raw.astype(float)

        im = ax.imshow(
            M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
        )

        # Diagonal guide: D_actual == D_ref
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

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(
            im, cax=cax,
            label="seed count",
        )
        if mode == "fractions":
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        f"Decile migration {title_suffix}\n"
        f"rows = decile in no_ancilla, cols = decile in this architecture",
        y=0.995,
    )
    fig.subplots_adjust(top=0.92, left=0.06, right=0.94, bottom=0.06)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def write_heatmaps(rows: list[dict], counts: dict, out_dir: str) -> None:
    """Build matrices and write the two migration heatmap PNGs."""
    configs_present = _configs_present_from(rows)
    matrices = _build_migration_matrices(counts, configs_present, N_DECILES)
    ref_deciles = sorted(counts.keys())

    counts_path = os.path.join(out_dir, "decile_migration_counts.png")
    _plot_migration_grid(
        matrices, ref_deciles, N_DECILES, counts_path,
        title_suffix="(seed counts)",
    )


# -- FIDELITY HEATMAPS  (D_ref vs fidelity bin) -------------------------------------------
def _build_fidelity_matrices(
    counts: dict,
    configs_present: list[str],
    n_bins: int,
) -> dict[str, np.ndarray]:
    """For each config, build a (n_ref, n_bins) matrix of seed counts."""
    ref_deciles = sorted(counts.keys())
    matrices: dict[str, np.ndarray] = {}
    for cfg in configs_present:
        M = np.zeros((len(ref_deciles), n_bins), dtype=int)
        for i, d_ref in enumerate(ref_deciles):
            per_b = counts[d_ref].get(cfg, {})
            for b_idx, n in per_b.items():
                if 0 <= b_idx < n_bins:
                    M[i, b_idx] = n
        matrices[cfg] = M
    return matrices


def _plot_fidelity_grid(
    matrices: dict[str, np.ndarray],
    ref_deciles: list[int],
    bin_labels: list[str],
    out_path: str,
    mode: str,
    title_suffix: str,
) -> None:
    """One PNG, one heatmap per architecture in a grid.

    Layout mirrors the migration heatmap so the eye can compare directly.
    No diagonal guide: rows and columns are different quantities here.
    """
    configs = list(matrices.keys())
    n = len(configs)
    if n <= 2:
        nrows, ncols = 1, n
    else:
        ncols = 2
        nrows = (n + ncols - 1) // ncols

    n_bins = len(bin_labels)
    fig = plt.figure(figsize=(6.5 * ncols + 1.2, 5.0 * nrows))
    gs = fig.add_gridspec(
        nrows, ncols + 1,
        width_ratios=[1.0] * ncols + [0.06],
        wspace=0.30, hspace=0.45,
    )
    axes_flat = [fig.add_subplot(gs[r, c])
                 for r in range(nrows) for c in range(ncols)]
    cax = fig.add_subplot(gs[:, ncols])

    vmax = max((M.max() for M in matrices.values()), default=1)
    vmin = 0

    color_threshold = 0.6 * vmax if vmax > 0 else 0.0
    cmap = plt.get_cmap("Blues")

    im = None
    for ax, cfg in zip(axes_flat, configs):
        M_raw = matrices[cfg]
        if mode == "fractions":
            row_sums = M_raw.sum(axis=1, keepdims=True)
            M = np.where(row_sums > 0, M_raw / np.maximum(row_sums, 1), 0.0)
        else:
            M = M_raw.astype(float)

        im = ax.imshow(
            M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
        )

        _annotate_cells(ax, M, color_threshold)

        ax.set_xticks(range(n_bins))
        # Rotate labels: fidelity bin labels are wider than "Dk".
        ax.set_xticklabels(bin_labels, fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(ref_deciles)))
        ax.set_yticklabels([f"D{d}" for d in ref_deciles], fontsize=8)
        ax.set_xlabel("max fidelity reached during training")
        ax.set_ylabel(r"$D_\mathrm{ref}$  (decile in no_ancilla)")
        ax.set_title(cfg, fontsize=11)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(
            im, cax=cax,
            label="seed count",
        )
        if mode == "fractions":
            cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    fig.suptitle(
        f"Fidelity after training {title_suffix}\n"
        f"rows = decile in no_ancilla ||g||², cols = max fidelity reached",
        y=0.995,
    )
    fig.subplots_adjust(top=0.92, left=0.06, right=0.94, bottom=0.10)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[done] wrote {out_path}")


def write_fidelity_heatmaps(rows: list[dict], fid_counts: dict, out_dir: str) -> None:
    """Build matrices and write the two fidelity-heatmap PNGs."""
    configs_present = _configs_present_from(rows)
    n_bins = len(FIDELITY_BIN_LABELS)
    matrices = _build_fidelity_matrices(fid_counts, configs_present, n_bins)
    ref_deciles = sorted(fid_counts.keys())

    counts_path = os.path.join(out_dir, "decile_migration_fidelities_counts.png")
    _plot_fidelity_grid(
        matrices, ref_deciles, FIDELITY_BIN_LABELS, counts_path,
        title_suffix="(seed counts)",
    )

#-- MAIN --------------------------------------------------------------------
def run(variance_timestamp: str | None = None) -> None:
    """Generate all migration tables and heatmaps for a variance run.

    Importable entry point so other scripts (e.g. decile_training_crn.py)
    can trigger the analysis automatically once training finishes. When
    `variance_timestamp` is given it overrides the module-level
    VARIANCE_TIMESTAMP (and the derived manifest path) for this call.
    """
    global VARIANCE_TIMESTAMP, MANIFEST_PATH
    if variance_timestamp is not None:
        VARIANCE_TIMESTAMP = variance_timestamp
        MANIFEST_PATH = os.path.join(
            "generated_data", "decile_seeds", VARIANCE_TIMESTAMP, "manifest.csv"
        )

    manifest_path = _resolve_manifest_path(MANIFEST_PATH)
    print(f"[load] manifest: {manifest_path}")
    rows = load_manifest(manifest_path)
    if not rows:
        print("[error] manifest has no usable rows.")
        sys.exit(1)

    # Attach max-fidelity per row by reading each training output.
    training_root = _resolve_training_root(manifest_path)
    print(f"[load] training root: {training_root}")
    if not os.path.isdir(training_root):
        print(f"[warn] training root not found; fidelity heatmaps will be empty.")
    n_fid_ok = attach_fidelities(rows, training_root)
    print(f"[load] attached fidelity for {n_fid_ok} / {len(rows)} rows")

    # Migration counts + norm stats + fidelity counts (all from the same rows).
    counts = build_counts(rows)
    norm_stats = build_norm_stats(rows)
    fid_counts = build_fidelity_counts(rows)

    # Outputs go next to the manifest.
    if WRITE_TXT_REPORT:
        out_dir = os.path.dirname(manifest_path)
        out_path = os.path.join(out_dir, "decile_migration.txt")
        write_report(rows, counts, norm_stats, out_path)
        print(f"[done] wrote {out_path}")
    else:
        print("[skip] WRITE_TXT_REPORT is False — text report not generated")

    # Migration heatmaps (D_ref vs D_actual)
    write_heatmaps(rows, counts, out_dir)

    # Fidelity heatmaps (D_ref vs fidelity bin) — only meaningful if we read
    # any fidelities at all.
    if n_fid_ok > 0:
        write_fidelity_heatmaps(rows, fid_counts, out_dir)
    else:
        print("[skip] no fidelities loaded — fidelity heatmaps not generated")


def main() -> None:
    run()


if __name__ == "__main__":
    main()