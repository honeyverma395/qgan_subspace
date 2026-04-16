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
"""Cost and Fidelity Functions — PyTorch version.

Reduced from the original cost_functions.py:
    - braket()                    -> REMOVED (replaced by torch.vdot _calc_wasserstein)
    - compute_fidelity()          -> kept, torch version
    - compute_fidelity_and_cost() -> kept, calls dis.compute_loss
"""

import torch
import numpy as np
from config import CFG
 
def _calc_wasserstein(g, t, dis_matrices):
        """Computation of the Wasserstein loss.
            Instead of repeating the code, we create this function to reuse
            in Choi and Haar method. 
        """
        A, B, psi, phi = dis_matrices

        Ag = A @ g;  Bg = B @ g;  At = A @ t;  Bt = B @ t

        # <g|A|g> · <t|B|t>
        term1 = torch.vdot(g, Ag) * torch.vdot(t, Bt)
        #cross terms: <t|A|g><g|B|t> + <g|A|t><t|B|g>
        term2 = torch.vdot(t, Ag) * torch.vdot(g, Bt)
        term3 = torch.vdot(g, At) * torch.vdot(t, Bg)
        # <t|A|t> · <g|B|g>
        term4 = torch.vdot(t, At) * torch.vdot(g, Bg)

        psiterm = torch.vdot(t, psi @ t)
        phiterm = torch.vdot(g, phi @ g)

        regterm = (CFG.lamb / np.e) * (
            CFG.cst1 * term1
            - CFG.cst2 * (term2 + term3)
            + CFG.cst3 * term4
        )
        return (psiterm - phiterm - regterm).real


def compute_fidelity(final_target_state: torch.Tensor,
                     final_gen_state: torch.Tensor) -> float:
    """Calculate the fidelity between target and generated states.

    Fidelity = |<target|gen>|^2 (for pure states)

    Args:
        final_target_state: Target state, shape (d,) or (d, 1).
        final_gen_state: Generator state, shape (d,) or (d, 1).

    Returns:
        float: fidelity ∈ [0, 1].
    """
    t = final_target_state.reshape(-1)
    g = final_gen_state.reshape(-1)
    overlap = torch.vdot(t, g) #better to use .vdot than .dot(t.conj(), g)
    return float(torch.abs(overlap) ** 2)

def compute_fidelity_and_cost(dis, final_target_state: torch.Tensor,
                               final_gen_state: torch.Tensor) -> tuple[float, float]:
    """Calculate both fidelity and cost for logging.

    Args:
        dis: Discriminator (torch version).
        final_target_state: Target state.
        final_gen_state: Generator state.

    Returns:
        (fidelity, cost): both as floats.
    """
    fidelity = compute_fidelity(final_target_state, final_gen_state)
    dis_matrices = dis.get_dis_matrices_rep()
    cost = _calc_wasserstein(final_gen_state, final_target_state, dis_matrices)
    return fidelity, cost.item()