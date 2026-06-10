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
"""Training module for the Quantum GAN — PyTorch and PennyLane version.

The training loop logic is unchanged from the original.
Changes:
    - We defined the update_gen in generator.py since the parameters of A and B,
    are detached from the parameters of the circuit.
    - We have now use_choi (bool) to specify if we are going to train using Choi or using Haar
    random states.
"""

from datetime import datetime

import numpy as np
import torch

from config import CFG
from qgan.ancilla import (
    get_final_gen_state_torch,
    get_max_entangled_state_with_ancilla_if_needed,
    get_random_batch,
    prepare_batch_targets,
    extract_gen_unitary, 
    process_fidelity,
    deterministic_batch_size,
    haar_random_batch
)
from qgan.cost_functions import compute_fidelity_and_cost
from qgan.discriminator import Discriminator
from qgan.generator import (
    Generator,
)
from qgan.target import get_target_operator
from tools.data_managers import (
    print_and_log,
    save_fidelity_loss,
    save_gen_final_params,
)
from tools.loading_helpers import load_models_if_specified
from tools.plot_hub import plt_fidelity_vs_iter

np.random.seed()


class Training:
    def __init__(self):
        """Builds the training configuration.

        Choi mode:
            1. Maximally entangled state |\Phi^+> (+ ancilla if needed).
            2. Target state: (I \otimes U_target [\otimes I_ancilla]) |\Phi^+>.

        Batch modes (use_choi=False):
            - "haar": fresh Haar-random batch of size CFG.batch_size each iter.
            - "comp": full deterministic all-pairs set of size d(d+1)/2 each iter.
            Target states: U_target |\psi_i> (with ancilla if needed).
 
        Evaluation (use_choi=False):
            Always Haar-random states, seeded with CFG.eval_seed, of size
            CFG.eval_batch_size. Reused across runs as the generalisation probe.
        """
        # -- Target operator --------
        self.target_op = torch.tensor(get_target_operator(), dtype=torch.complex64)

        if CFG.use_choi:
            # -- Choi mode: apply target_op to |\Phi^+> --
            _, initial_state_final = get_max_entangled_state_with_ancilla_if_needed(CFG.system_size)
            initial_state = torch.tensor(np.asarray(initial_state_final), dtype=torch.complex64).reshape(-1)

            self.final_target_state = (self.target_op @ initial_state).reshape(-1)

        if not CFG.use_choi and CFG.batch_mode == "comp":
            dim = 2**CFG.system_size
            M = deterministic_batch_size(dim)
            if CFG.batch_size > M:
                print_and_log(
                    f"batch_size {CFG.batch_size} -> {M} (capped at d(d+1)/2 for d={dim}).",
                    CFG.log_path,
                )
                CFG.batch_size = M

        # Generator: variational quantum circuit
        self.gen: Generator = Generator()

        # Discriminator: classical Hermitian operator
        self.dis: Discriminator = Discriminator()

        # -- Eval batch (batching only) ----------------------
        # Fixed evaluation set: generated once with a fixed seed so that
        # the fidelity curve reflects generalisation, not the luck of the
        # current training batch. Same set is reused across iters and runs.
        if not CFG.use_choi:
            dim = 2**CFG.system_size
            self.eval_batch_size = CFG.eval_batch_size
            eval_seed = getattr(CFG, "eval_seed", 1234)

            # Snapshot + restore RNG state so eval-set generation does not
            # perturb the training RNG stream.
            torch_state = torch.random.get_rng_state()
            np_state = np.random.get_state()
            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)
            try:
                self.eval_raw, self.eval_inputs = haar_random_batch(
                    dim, self.eval_batch_size
                )
                self.eval_targets = prepare_batch_targets(
                    self.eval_raw, self.eval_inputs, self.target_op
                )
            finally:
                torch.random.set_rng_state(torch_state)
                np.random.set_state(np_state)

    def run(self):
        """Run the training loop.

        For each iteration:
            1. Process ancilla on generator output (torch).
            2. Update discriminator (maximise Wasserstein distance).
            3. Update generator (minimise Wasserstein distance).
            4. Log fidelity and loss periodically.

        Choi mode:
            Each iteration uses the fixed |\Phi^+> state.

        Batch mode (haar or comp):
            Each iteration draws B states via `get_random_batch`:
              - "haar": fresh Haar-random sample of size CFG.batch_size.
              - "comp": full deterministic all-pairs set of size d(d+1)/2
                        (CFG.batch_size was overridden to this value in __init__).
            Loss and gradients are averaged over the batch (mini-batch SGD).

        Stops when max_fidelity is reached or max epochs is reached.
        Saves models, fidelity history, and generator parameters at the end.
        """
        # -- Initialise training -----------------------------------
        print_and_log("\n" + CFG.show_data(), CFG.log_path)

        # Load models if a previous checkpoint is specified
        load_models_if_specified(self)

        fidelities_history, losses_history = [], []
        starttime = datetime.now()
        num_epochs: int = 0

        # -- Main training loop -----------------------------------
        while True:
            fidelities = []
            losses = []
            num_epochs += 1

            for epoch_iter in range(CFG.iterations_epoch):
                if CFG.use_choi:
                    self._train_step_choi(epoch_iter, fidelities, losses)
                else:
                    self._train_step_batch(epoch_iter, fidelities, losses)

                # -- Evaluation on fixed set --
                if epoch_iter % CFG.save_fid_and_loss_every_x_iter == 0:
                    if CFG.use_choi:
                        self._evaluate_choi(fidelities, losses)
                    else:
                        self._evaluate_batch(fidelities, losses)

                # -- Log
                if epoch_iter % CFG.log_every_x_iter == 0 and fidelities:
                    info = (
                        f"\nepoch:{num_epochs:4d} | "
                        f"iters:{epoch_iter + 1:4d} | "
                        f"fidelity:{round(fidelities[-1], 6):8f} | "
                        f"loss:{round(losses[-1], 6):8f}"
                    )
                    print_and_log(info, CFG.log_path)

            # -- End of epoch: store history and plot -----------------------------------
            fidelities_history = np.append(fidelities_history, fidelities)
            losses_history = np.append(losses_history, losses)
            plt_fidelity_vs_iter(fidelities_history, losses_history, CFG, num_epochs)

            # -- Stopping conditions -----------------------------------
            if num_epochs >= CFG.epochs:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe number of epochs exceeds {CFG.epochs}.",
                    CFG.log_path,
                )
                break

            if fidelities[-1] > CFG.max_fidelity:
                print_and_log(
                    "\n==================================================\n",
                    CFG.log_path,
                )
                print_and_log(
                    f"\nThe fidelity {fidelities[-1]} exceeds the maximum {CFG.max_fidelity}.",
                    CFG.log_path,
                )
                break

        # -- End of training: save everything -----------------------------------
        save_fidelity_loss(fidelities_history, losses_history, CFG.fid_loss_path)
        self.gen.save_model(CFG.model_gen_path)
        self.dis.save_model(CFG.model_dis_path)
        save_gen_final_params(self.gen, CFG.gen_final_params_path)
        # Time calculation
        endtime = datetime.now()
        print_and_log(f"\nRun took: {endtime - starttime} time.", CFG.log_path)

        # -- Save gradient history --
        if self.gen.grad_history:
            grad_array = np.stack(self.gen.grad_history, axis=0)
            np.save(f"{CFG.base_data_path}/grad_history.npy", grad_array)
            if self.gen.ancilla:
                anc_idx = np.array(self.gen._get_ancilla_param_indices(),
                                   dtype=int)
                np.save(f"{CFG.base_data_path}/ancilla_indices.npy", anc_idx)
            print_and_log(
                f"\nSaved gradient history: shape {grad_array.shape}",
                CFG.log_path,
            )
        # -- Save discriminator gradient history --
        if self.dis.grad_history:
            dis_grad_array = np.stack(self.dis.grad_history, axis=0)
            np.save(f"{CFG.base_data_path}/dis_grad_history.npy", dis_grad_array)
            print_and_log(
                f"\nSaved discriminator gradient history: shape {dis_grad_array.shape}",
                CFG.log_path,
            )

    def _record_dis_grad(self) -> None:
        """Record flat gradient vector over all discriminator parameters."""
        pieces = []
        for p in self.dis.parameters():
            if p.grad is not None:
                pieces.append(p.grad.detach().cpu().numpy().ravel().copy())
        if pieces:
            self.dis.grad_history.append(np.concatenate(pieces))

    # -- Choi mode (max entangled state) ------------------
    def _train_step_choi(self, epoch_iter: int, fidelities: list, losses: list):
        """One training iteration in Choi mode."""
        # -- Detached state for discriminator
        with torch.no_grad():
            total_gen_detached = self.gen.get_total_gen_state()
            final_gen_detached = get_final_gen_state_torch(total_gen_detached).reshape(-1)

        # -- Discriminator steps
        for _ in range(CFG.steps_dis):
            self.dis.optimizer.zero_grad()
            dis_loss = self.dis.compute_loss(self.final_target_state, final_gen_detached)
            dis_loss.backward()
            self._record_dis_grad()
            self.dis.optimizer.step()

        # -- Generator steps
        for _ in range(CFG.steps_gen):
            self.gen.update_gen(self.dis, self.final_target_state)

    # -- Batching (random states) ------------------
    def _train_step_batch(self, epoch_iter: int, fidelities: list, losses: list):
        """One training iteration in batch mode (Haar or deterministic).
 
        1. Draw B input states via `get_random_batch`:
            - "haar": fresh Haar-random sample of size CFG.batch_size.
            - "comp": the FULL deterministic all-pairs set of size
                      M = d(d+1)/2 (CFG.batch_size is overridden to M in
                      __init__, and `get_random_batch` returns the cached
                      set on every call).
        2. Compute target states: U_target|ψ_i⟩ for each.
        3. Discriminator (maximise) and Generator (minimise).
        4. Evaluate average fidelity over batch periodically.
        """
        B = CFG.batch_size
        dim = 2**CFG.system_size

        # -- Generate Haar batch for each iter (includes ancilla |0> if needed) --
        batch_raw, batch_inputs = get_random_batch(dim, B)
        batch_targets = prepare_batch_targets(batch_raw, batch_inputs, self.target_op)

        # -- Discriminator: detached generator states, averaged loss --
        with torch.no_grad():
            det_gen_states = [
                get_final_gen_state_torch(self.gen.get_total_gen_state(inp)).reshape(-1) for inp in batch_inputs
            ]

        for _ in range(CFG.steps_dis):
            self.dis.optimizer.zero_grad()
            dis_loss = torch.tensor(0.0, dtype=torch.float32)
            for t, g in zip(batch_targets, det_gen_states):
                dis_loss = dis_loss + self.dis.compute_loss(t, g)
            dis_loss = dis_loss / B
            dis_loss.backward()
            self._record_dis_grad()
            self.dis.optimizer.step()

        # -- Generator: averaged loss with gradients --
        for _ in range(CFG.steps_gen):
            self.gen.update_gen(
                self.dis,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
            )

    # -- Evaluation (separated from training) ------------------
    def _evaluate_choi(self, fidelities: list, losses: list) -> None:
        """Evaluate fidelity and loss on the fixed |\Phi^+> state."""
        with torch.no_grad():
            final_gen_eval = get_final_gen_state_torch(self.gen.total_gen_state).reshape(-1)
        fid, loss = compute_fidelity_and_cost(
            self.dis, self.final_target_state, final_gen_eval
        )
        fidelities.append(fid)
        losses.append(loss)

    def _evaluate_batch(self, fidelities: list, losses: list) -> None:
        """Evaluate unitary process fidelity + loss on the fixed eval batch.

        Process fidelity is computed from U_gen (via qml.matrix) vs U_target,
        matching the Choi-mode fidelity exactly:
            F = |Tr(U_t^dagger U_g)|^2 / d^2
        The loss still uses the eval batch + discriminator since it depends on
        the discriminator state.
        """
        with torch.no_grad():
            U_gen = extract_gen_unitary(self.gen)
            fid = process_fidelity(U_gen, self.target_op)

            eval_gen = [
                get_final_gen_state_torch(self.gen.get_total_gen_state(inp)).reshape(-1)
                for inp in self.eval_inputs
            ]
        loss_sum = 0.0
        for t, g in zip(self.eval_targets, eval_gen):
            _, l = compute_fidelity_and_cost(self.dis, t, g)
            loss_sum += l
        fidelities.append(fid)
        losses.append(loss_sum / len(self.eval_inputs))
