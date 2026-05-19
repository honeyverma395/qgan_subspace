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
"""Common Random Numbers (CRN) version of the decile-seed training.

Pipeline:
  1. Read the variance run's config.txt and reapply its CIRCUIT fields to
     CFG (system_size, gen_layers, gen_ansatz, custom_ansatz_terms,
     ancilla_*, target_hamiltonian, etc.). This guarantees Generator
     produces the SAME number of params as the .npy files on disk, even
     if config.py has been edited since the variance run.
  2. Load (grads, thetas) for ALL configs from the variance run.
  3. Pick seed_ids using ONLY `no_ancilla` as the reference distribution:
     for each decile in DECILES, sample N_PER_CELL seeds from that decile
     of ||g||^2_{no_ancilla}.
  4. For each chosen seed_id, train ALL FOUR configs with THE SAME
     seed_id (so they share theta_sys, ancilla pools, etc. - exactly the
     coupling that variance_analysis.py already produced).
  5. For each (config, seed_id), report the seed's REAL decile inside
     that config's own ||g||^2 distribution (`decile_actual`), alongside
     the reference decile from no_ancilla (`decile_ref`).
  6. Manifest is grouped BY SEED: four consecutive rows per seed, in the
     order [no_ancilla, ancilla_total, ancilla_bridge, ancilla_shortBridge].

Design choices kept from the previous script:
  - Shared Discriminator PER SIZE-FAMILY (no_anc vs with_anc).
  - type_of_warm_start = "none" so the loaded theta is used as-is.
  - Manifest flushed after every cell, atomic rename, so partial
    progress survives interruptions.
"""
import os
import sys
import csv
import ast
import traceback
import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import CFG
from qgan.generator import Generator
from qgan.discriminator import Discriminator
from qgan.training import Training
from variance.variance_analysis import _apply_config, _snapshot_cfg, _restore_cfg
from tools.data_managers import print_and_log_with_headers, print_and_log


# ---- EDIT HERE -----------------------------------------------------
VARIANCE_TIMESTAMP = "Try_3_C_ZZZ"

# Selection is driven by no_ancilla; the other three reuse the same seed_ids.
REF_CONFIG = "no_ancilla"
CONFIGS = [
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]
# Deciles taken from the REF_CONFIG distribution.
DECILES = [1, 5, 10]

N_PER_CELL  = 50
N_DECILES   = 10
SEED_PICKER = 0           # for reproducible decile-pool sampling
SEEDS_ROOT  = "decile_seeds"
TRAIN_ROOT  = "decile_training"

# Fields to re-apply from config.txt onto CFG before building anything.
# These are the ones that determine Generator's n_params, the circuit
# topology, and the target. NOT training hyperparameters (epochs, l_rate,
# steps_*, etc.), those come from your live config.py because the
_CIRCUIT_FIELDS = (
    "use_choi",
    "batch_size",
    "batch_mode",
    "system_size",
    "extra_ancilla",
    "ancilla_mode",
    "ancilla_project_norm",
    "ancilla_topology",
    "ancilla_connect_to",
    "do_ancilla_1q_gates",
    "start_ancilla_gates_randomly",
    "ancilla_coupling_layers",
    "ancilla_training",
    "gen_layers",
    "gen_ansatz",
    "custom_ansatz_terms",
    "target_hamiltonian",
    "custom_hamiltonian_terms",
    "custom_hamiltonian_strengths",
    "time_to_evolve",
)
# --------------------------------------------------------------------


# Deterministic mapping (decile_ref, rep, config) -> experiment index.
# Order: outer loop over DECILES, then rep within decile, then CONFIGS.
# This keeps the on-disk layout grouped by seed, matching the CSV.
CELL_ORDER = [
    (d, r, c)
    for d in DECILES
    for r in range(1, N_PER_CELL + 1)
    for c in CONFIGS
]
CELL_IDX = {cell: i + 1 for i, cell in enumerate(CELL_ORDER)}


# -- Config-file parsing ---------------------------------------------
def parse_variance_config(config_txt_path: str) -> dict:
    """Parse the `key: value` lines of variance's config.txt.

    The file has a free-form header (N_SAMPLES, SEED, configs), then a
    box-drawing-separated block with `key: value` pairs - one per line.
    We ignore the header, ignore separator lines, and parse anything of
    the form `name: literal` using ast.literal_eval to recover proper
    Python types (bool, int, float, list, None, ...).

    Quirks handled:
      - Some lines end with a trailing comma (e.g. `time_to_evolve: 1.0,`).
        We strip it before parsing.
      - `ancilla_coupling_layers` can be the bare word `all` (no quotes).
      - The bare word `None` is mapped to Python None.
      - Enum-like strings written without quotes (e.g. `batch_mode:haar`,
        `gen_ansatz: ZZ_Z_X`) fall back to plain str.
    """
    parsed: dict = {}
    with open(config_txt_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("=") or line.startswith("─"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().rstrip(",")  # strip trailing comma if any
            if val == "":
                parsed[key] = None
                continue
            if val == "all":
                parsed[key] = "all"
                continue
            if val == "None":
                parsed[key] = None
                continue
            try:
                parsed[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Fall back to raw string (e.g. enum-like values such as
                # "haar", "pass", "ZZ_Z_X" written without quotes).
                parsed[key] = val
    return parsed


def apply_variance_config_to_cfg(parsed: dict, fields: tuple = _CIRCUIT_FIELDS) -> dict:
    """Overwrite the listed CFG fields with values from `parsed`.

    Returns the dict of {field: (old, new)} for fields that were actually
    changed, for logging only. Missing fields in `parsed` are skipped:
    we don't error, the variance run may simply not have logged them.
    """
    changed: dict = {}
    for f in fields:
        if f not in parsed:
            continue
        old = getattr(CFG, f, "<missing>")
        new = parsed[f]
        if old != new:
            changed[f] = (old, new)
        setattr(CFG, f, new)
    return changed


# -- Helpers ---------------------------------------------------------
def pick_indices(norms_sq: np.ndarray, decile: int, n_pick: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Pick n_pick seed indices from the requested decile (1..N_DECILES)
    of the ||g||^2 distribution, sampled without replacement.
    """
    n = norms_sq.size
    bin_size = n // N_DECILES
    order = np.argsort(norms_sq)
    lo = (decile - 1) * bin_size
    hi = decile * bin_size if decile < N_DECILES else n
    pool = order[lo:hi]
    if n_pick > pool.size:
        raise ValueError(f"decile {decile}: pool={pool.size}, need={n_pick}")
    return rng.choice(pool, size=n_pick, replace=False)


def compute_decile_lookup(norms_sq: np.ndarray) -> np.ndarray:
    """Return an array `decile_of[i]` giving the 1..N_DECILES decile of
    sample i inside this `norms_sq` distribution.

    Same binning convention as `pick_indices`: sort by ||g||^2 ascending,
    split into N_DECILES equal-size bins (the last one absorbing the
    remainder). Decile 1 = smallest ||g||^2, decile 10 = largest.
    """
    n = norms_sq.size
    bin_size = n // N_DECILES
    order = np.argsort(norms_sq)                  # indices, ascending norm_sq
    decile_of = np.empty(n, dtype=np.int64)
    for d in range(1, N_DECILES + 1):
        lo = (d - 1) * bin_size
        hi = d * bin_size if d < N_DECILES else n
        decile_of[order[lo:hi]] = d
    return decile_of


def materialize_checkpoint(
    gen_probe: Generator,
    the_dis: Discriminator,
    theta: np.ndarray,
    ckpt_dir: str,
) -> None:
    """Write {gen, dis} into <ckpt_dir>/saved_model/ using the project's
    save_model() so load_models_if_specified can read it back.
    """
    saved_dir = os.path.join(ckpt_dir, "saved_model")
    os.makedirs(saved_dir, exist_ok=True)

    with torch.no_grad():
        gen_probe.params.copy_(
            torch.tensor(theta, dtype=gen_probe.params.dtype)
        )
    if hasattr(gen_probe, "_refresh_state"):
        gen_probe._refresh_state()

    # Temporarily redirect CFG paths so save_model writes into ckpt_dir.
    saved_base = CFG.base_data_path
    saved_load = CFG.load_timestamp
    try:
        CFG.base_data_path = ckpt_dir
        CFG.load_timestamp = None  # writing, not loading
        CFG.set_results_paths()
        gen_probe.save_model(CFG.model_gen_path)
        the_dis.save_model(CFG.model_dis_path)
    finally:
        CFG.base_data_path = saved_base
        CFG.load_timestamp = saved_load


def train_cell(
    config_name: str,
    decile_ref: int,
    rep: int,
    load_ts_rel: str,
) -> None:
    """Run Training().run() for one cell, with CFG pointed correctly.

    The experiment index is determined by (decile_ref, rep, config) so the
    on-disk layout follows the seed-grouped order of the manifest.
    """
    cell_idx = CELL_IDX[(decile_ref, rep, config_name)]
    out_rel = os.path.join(
        TRAIN_ROOT, VARIANCE_TIMESTAMP,
        f"experiment{cell_idx}", str(rep),
    )
    CFG.base_data_path = os.path.join(_PROJECT_ROOT, "generated_data", out_rel)
    CFG.load_timestamp = load_ts_rel
    CFG.set_results_paths()

    print_and_log_with_headers(
        f"\nTRAINING: {config_name} | D_ref{decile_ref} | rep {rep}\n"
        f"  load_timestamp = {load_ts_rel}\n"
        f"  output         = {out_rel}",
        CFG.log_path,
    )
    Training().run()


def write_manifest(manifest_path: str, rows: list[dict]) -> None:
    """Overwrite the manifest CSV with the current rows.

    Called after every cell so partial progress survives interruptions
    (Ctrl+C, training crashes, ...). Writes to a tmp file then renames,
    so a kill mid-write cannot leave a half-written manifest.

    Columns:
      - decile_ref : the no_ancilla decile that drove the selection of
                     this seed (same value for all four rows of a seed).
      - decile_actual : the decile of this same seed_id inside THIS
                        config's own ||g||^2 distribution. May differ
                        from decile_ref for the ancilla configs.
      - norm_sq : ||g||^2 for this seed in THIS config.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "config", "decile_ref", "decile_actual", "rep",
            "seed_id", "norm_sq",
            "load_timestamp", "status", "error_msg",
        ])
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_path, manifest_path)


# -- Main ------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(SEED_PICKER)
    src_dir = os.path.join(
        _PROJECT_ROOT, "variance_analysis", VARIANCE_TIMESTAMP
    )
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(src_dir)

    # -- (NEW) Replay the circuit config from the variance run -------
    # This is the key step: we read the config.txt that variance_analysis
    # wrote into src_dir and overwrite the relevant CFG fields. Generator
    # then sees the SAME settings that produced the .npy on disk, so its
    # n_params matches and we don't fight the live config.py.
    config_txt = os.path.join(src_dir, "config.txt")
    if not os.path.exists(config_txt):
        raise FileNotFoundError(
            f"Cannot replay variance config: missing {config_txt}. "
            f"This script needs it to align CFG with the .npy files."
        )
    parsed = parse_variance_config(config_txt)
    changed = apply_variance_config_to_cfg(parsed, _CIRCUIT_FIELDS)
    if changed:
        print("[replay] Overwrote CFG fields from variance config.txt:")
        for k, (old, new) in changed.items():
            print(f"    {k}: {old!r} -> {new!r}")
    else:
        print("[replay] No CFG fields needed updating (already aligned).")

    # The Training class must not perturb the loaded theta.
    CFG.type_of_warm_start = "none"

    # -- Load grads/thetas for ALL configs up front ------------------
    # We need this for two reasons:
    #   (a) compute each config's per-sample decile (decile_actual);
    #   (b) feed the right theta into each config's generator at train time.
    # Because variance_analysis.sample_gradients_coupled wrote the SAME
    # seed_id rows in the SAME order across all configs, seed_id is a
    # global identifier - row i in grads_no_ancilla.npy corresponds to
    # row i in grads_ancilla_total.npy, etc.
    grads_by_cfg:  dict[str, np.ndarray] = {}
    thetas_by_cfg: dict[str, np.ndarray] = {}
    norms_sq_by_cfg: dict[str, np.ndarray] = {}
    decile_of_by_cfg: dict[str, np.ndarray] = {}

    for cname in CONFIGS:
        gp = os.path.join(src_dir, f"grads_{cname}.npy")
        tp = os.path.join(src_dir, f"thetas_{cname}.npy")
        if not (os.path.exists(gp) and os.path.exists(tp)):
            raise FileNotFoundError(
                f"Missing grads/thetas for config '{cname}' in {src_dir}"
            )
        grads_by_cfg[cname]  = np.load(gp)
        thetas_by_cfg[cname] = np.load(tp)
        norms_sq_by_cfg[cname] = np.sum(grads_by_cfg[cname] ** 2, axis=1)
        decile_of_by_cfg[cname] = compute_decile_lookup(norms_sq_by_cfg[cname])

    # Sanity: all configs must have the same number of samples (CRN).
    n_samples_ref = grads_by_cfg[REF_CONFIG].shape[0]
    for cname in CONFIGS:
        if grads_by_cfg[cname].shape[0] != n_samples_ref:
            raise RuntimeError(
                f"Sample count mismatch: {cname} has "
                f"{grads_by_cfg[cname].shape[0]} samples, "
                f"REF_CONFIG '{REF_CONFIG}' has {n_samples_ref}. "
                f"CRN seed reuse across configs is not safe."
            )

    # -- Shared discriminators per size-family ------------------------
    # CFG.ancilla_mode == "pass" makes Discriminator.size depend on
    # extra_ancilla, so there are two size-families:
    #   - no_ancilla  -> size = N  (or 2N with Choi)
    #   - any ancilla -> size = N+1  (or 2N+1 with Choi)
    # We build ONE Discriminator per family, shared across all cells of
    # that family. This keeps the discriminator out of the variance budget
    # WITHIN a family, while the cross-family comparison already has the
    # ancilla qubit as a non-comparable axis.
    snapshot = _snapshot_cfg()
    dis_by_config: dict[str, Discriminator] = {}
    _built_for_no_anc = False
    _built_for_with_anc = False
    try:
        for cname in CONFIGS:
            _apply_config(cname)
            if cname == "no_ancilla":
                if not _built_for_no_anc:
                    dis_no_anc = Discriminator()
                    _built_for_no_anc = True
                    print(f"Built shared Discriminator for family 'no_ancilla' "
                          f"(size = {dis_no_anc.size}).")
                dis_by_config[cname] = dis_no_anc
            else:
                if not _built_for_with_anc:
                    dis_with_anc = Discriminator()
                    _built_for_with_anc = True
                    print(f"Built shared Discriminator for family 'with_ancilla' "
                          f"(size = {dis_with_anc.size}).")
                dis_by_config[cname] = dis_with_anc
    finally:
        _restore_cfg(snapshot)

    # -- Build a generator-probe per config (reused across all cells) --
    # We don't recreate Generator() for every seed; we just overwrite
    # its params via materialize_checkpoint.
    #
    # Diagnostic: print the CFG state that each Generator sees, plus the
    # resulting n_params and the on-disk thetas width. They MUST match.
    # If they don't even now (after replaying config.txt), something
    # downstream still differs: check the parser's coverage of the file.
    gen_probe_by_cfg: dict[str, Generator] = {}
    snapshot_gp = _snapshot_cfg()
    try:
        for cname in CONFIGS:
            _apply_config(cname)
            print(
                f"[probe {cname}] CFG: extra_ancilla={CFG.extra_ancilla} "
                f"topology={CFG.ancilla_topology} "
                f"connect_to={CFG.ancilla_connect_to} "
                f"layers={CFG.gen_layers} ansatz={CFG.gen_ansatz}"
            )
            gen_probe = Generator()
            disk_width = thetas_by_cfg[cname].shape[1]
            print(
                f"[probe {cname}] Generator.n_params={gen_probe.n_params}, "
                f"thetas on disk width={disk_width}"
            )
            if disk_width != gen_probe.n_params:
                raise RuntimeError(
                    f"{cname}: param mismatch on disk={disk_width} "
                    f"vs Generator.n_params={gen_probe.n_params} "
                    f"AFTER replaying config.txt. "
                    f"Check that all relevant fields are in "
                    f"_CIRCUIT_FIELDS and that parse_variance_config "
                    f"recovered them with the right types."
                )
            gen_probe_by_cfg[cname] = gen_probe
    finally:
        _restore_cfg(snapshot_gp)

    # -- Pick the seed_ids from REF_CONFIG only -----------------------
    # chosen_per_decile[decile] = list of seed_ids of length N_PER_CELL.
    # These seed_ids are GLOBAL row indices (consistent across configs).
    norms_sq_ref = norms_sq_by_cfg[REF_CONFIG]
    chosen_per_decile: dict[int, np.ndarray] = {}
    for decile in DECILES:
        chosen_per_decile[decile] = pick_indices(
            norms_sq_ref, decile, N_PER_CELL, rng
        )
        print(f"[select] decile {decile} on '{REF_CONFIG}': "
              f"picked {len(chosen_per_decile[decile])} seeds")

    # -- Run training, grouped by seed -------------------------------
    manifest_rows: list[dict] = []
    manifest_path = os.path.join(
        _PROJECT_ROOT, "generated_data",
        SEEDS_ROOT, VARIANCE_TIMESTAMP, "manifest.csv"
    )

    snapshot2 = _snapshot_cfg()
    try:
        # Outer: decile (of REF_CONFIG); middle: rep (seed within decile);
        # inner: config. This produces the desired CSV grouping:
        #   per seed -> 4 rows (no_ancilla, ancilla_total, bridge, shortBridge)
        for decile_ref in DECILES:
            chosen = chosen_per_decile[decile_ref]

            for rep, seed_id in enumerate(chosen, start=1):
                seed_id_int = int(seed_id)

                for config_name in CONFIGS:
                    # Activate this config's CFG flags so save_model /
                    # Training pick up the right architecture.
                    _apply_config(config_name)

                    # Where the materialized checkpoint will live.
                    # We tag the directory with decile_ref + rep so that
                    # the same seed is grouped on disk too.
                    load_ts_rel = os.path.join(
                        SEEDS_ROOT, VARIANCE_TIMESTAMP,
                        config_name,
                        f"Dref{decile_ref:02d}_rep{rep}_seed{seed_id_int}",
                    )
                    ckpt_dir = os.path.join(
                        _PROJECT_ROOT, "generated_data", load_ts_rel
                    )

                    # Materialize gen+dis for THIS config at THIS seed_id.
                    materialize_checkpoint(
                        gen_probe_by_cfg[config_name],
                        dis_by_config[config_name],
                        thetas_by_cfg[config_name][seed_id_int],
                        ckpt_dir,
                    )

                    # Train.
                    status = "ok"
                    error_msg = ""
                    try:
                        train_cell(config_name, decile_ref, rep, load_ts_rel)
                    except Exception as e:
                        status = "failed"
                        error_msg = f"{type(e).__name__}: {e}"
                        tb = traceback.format_exc()
                        print_and_log(
                            f"\nFAILED {config_name}/Dref{decile_ref}/rep{rep}/"
                            f"seed{seed_id_int}\n  {error_msg}\n{tb}",
                            CFG.log_path,
                        )
                    finally:
                        CFG.load_timestamp = None

                    # Record this cell. decile_actual is THIS config's
                    # own decile for this seed_id (may differ from
                    # decile_ref for the ancilla configs).
                    manifest_rows.append({
                        "config": config_name,
                        "decile_ref": decile_ref,
                        "decile_actual": int(
                            decile_of_by_cfg[config_name][seed_id_int]
                        ),
                        "rep": rep,
                        "seed_id": seed_id_int,
                        "norm_sq": float(
                            norms_sq_by_cfg[config_name][seed_id_int]
                        ),
                        "load_timestamp": load_ts_rel,
                        "status": status,
                        "error_msg": error_msg,
                    })
                    write_manifest(manifest_path, manifest_rows)
    finally:
        _restore_cfg(snapshot2)

    # Final summary (manifest itself was flushed after every cell).
    n_ok = sum(1 for r in manifest_rows if r["status"] == "ok")
    n_fail = sum(1 for r in manifest_rows if r["status"] == "failed")
    print(f"\nSaved manifest: {manifest_path}")
    print(f"  total cells: {len(manifest_rows)}  ok: {n_ok}  failed: {n_fail}")


if __name__ == "__main__":
    main()