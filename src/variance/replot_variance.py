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
"""Replot variance_analysis from saved grads_{config}.npy files.
"""

import os
import sys
import numpy as np

# Same path setup as variance_analysis.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import CFG
from qgan.generator import Generator

from variance.variance_analysis import (
    _apply_config,
    _snapshot_cfg,
    _restore_cfg,
    _split_system_ancilla,
    plot_variance_sweep,
)

# ---- EDIT HERE ------------------------------------------------------
TIMESTAMP = "batch50"          
CONFIGS = [                                  
    "no_ancilla",
    "ancilla_total",
    "ancilla_bridge",
    "ancilla_shortBridge",
]
OUT_NAME = "variance_plot_replot.png"        
# Set to None to autoscale, for ylim (1e-6, 1e-1)
XLIM: tuple[float, float] | None = None
YLIM: tuple[float, float] | None = (1e-4, 1e-1)
# ---------------------------------------------------------------------

# STOP EDITING HERE
def replot():
    out_dir = os.path.join(_PROJECT_ROOT, "variance_analysis", TIMESTAMP)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Not found: {out_dir}")

    snapshot = _snapshot_cfg()
    results: dict[str, dict[str, np.ndarray]] = {}
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

            grads_sys, grads_anc = _split_system_ancilla(grads, probe_gen)
            var_sys = np.var(grads_sys, axis=0)
            var_anc = np.var(grads_anc, axis=0) if grads_anc.size > 0 else np.array([])

            results[name] = {"var_sys": var_sys, "var_anc": var_anc}
            if n_sys_params_ref is None:
                n_sys_params_ref = len(var_sys)

            print(f"[ok]   {name}: n_samples={grads.shape[0]}, "
                  f"n_params={grads.shape[1]} "
                  f"(sys={len(var_sys)}, anc={len(var_anc)})")
    finally:
        _restore_cfg(snapshot)

    if not results:
        print("No configs loaded, nothing to plot.")
        return

    plot_path = os.path.join(out_dir, OUT_NAME)
    plot_variance_sweep(results, plot_path, n_sys_params_ref or 0,
                        xlim=XLIM, ylim=YLIM)
    print(f"\nSaved: {plot_path}")


if __name__ == "__main__":
    replot()