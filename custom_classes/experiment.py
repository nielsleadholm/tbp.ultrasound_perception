# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Custom experiment class for ultrasound inference experiments.

Adds optional plotting of the extracted patch features and hypothesis space, and
gracefully ends an episode when the (offline) environment runs out of data.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.geometry import Rotation

from .environment import AGENT_ID, SENSOR_ID
from .plotting import plot_combined_figure

logger = logging.getLogger(__name__)


class UltrasoundInferenceExperiment(MontyObjectRecognitionExperiment):
    """Object recognition experiment for ultrasound data.

    Differs from the base class in two ways:

    1. It catches ``StopIteration`` raised by the environment when it runs out of
       offline data points, ending the episode cleanly.
    2. It optionally plots the extracted patch features and hypothesis space.
    """

    def __init__(self, config):
        super().__init__(config)
        self.plotting_config = config.get("plotting_config", None)

    def run_episode_steps(self) -> int:
        """Run one episode, plotting optionally and handling end-of-data."""
        step = 0
        ctx = RuntimeContext(rng=self.rng)
        actions: list[Action] = []
        while True:
            try:
                observations, proprioceptive_state = self.env_interface.step(actions)
            except StopIteration:
                # Offline dataset exhausted for this object before a definitive match.
                # Treat this exactly like reaching ``max_eval_steps``: set any
                # not-yet-converged LM to ``time_out`` (and mark the model done) so the
                # logger evaluates the most likely hypothesis as ``correct_mlh`` /
                # ``confused_mlh`` rather than leaving the performance blank.
                self.model.deal_with_time_out()
                return step

            if self.model.check_reached_max_matching_steps(self.max_steps):
                logger.info(
                    f"Terminated due to maximum matching steps : {self.max_steps}"
                )
                return step

            if step >= self.max_total_steps:
                logger.info(f"Terminated due to maximum episode steps : {step}")
                self.model.deal_with_time_out()
                return step

            try:
                if self.model.is_motor_only_step:
                    actions = self.model.motor_only_step(
                        ctx, observations, proprioceptive_state
                    )
                else:
                    actions = self.model.step(ctx, observations, proprioceptive_state)
            except StopIteration:
                self.model.deal_with_time_out()
                return step

            self._maybe_plot(step, observations)

            if self.model.is_done:
                return step

            step += 1

    def _maybe_plot(self, step: int, observations) -> None:
        if not (
            self.plotting_config
            and self.plotting_config.get("enabled", False)
            and step % self.plotting_config.get("plot_frequency", 1) == 0
        ):
            return

        patch_data = observations[AGENT_ID][SENSOR_ID]
        sensor_module = self.model.sensor_modules[0]
        features = sensor_module.plotting_data

        save_path = None
        if self.plotting_config.get("save_path"):
            os.makedirs(self.plotting_config["save_path"], exist_ok=True)
            save_path = os.path.join(
                self.plotting_config["save_path"], f"step_{step}.png"
            )

        if self.plotting_config.get("plot_patch_features", False):
            full_image = self.env.get_full_image()
            plot_combined_figure(
                input_image=full_image,
                patch_image=patch_data["img"],
                all_column_points_tuples=features.get("column_points", []),
                center_edge_point=features.get("center_edge", None),
                fitted_circle=features.get("fitted_circle", None),
                point_normal=features.get("point_normal", None),
                curvature=features.get("curvature", 0),
                depth_meters=features.get("mean_depth", 0.0),
                observed_locations=features.get("observed_locations", []),
                normal_rel_world=features.get("normal_rel_world", []),
                save_path=save_path,
                show_hypothesis_space=self.plotting_config.get(
                    "show_hypothesis_space", False
                ),
                lm_instance=self.model.learning_modules[0],
                hypothesis_input_channel=self.plotting_config.get(
                    "hypothesis_input_channel", "patch"
                ),
                hypothesis_evidence_threshold=self.plotting_config.get(
                    "hypothesis_evidence_threshold", -np.inf
                ),
                display_mlh_focus_plot=self.plotting_config.get(
                    "display_mlh_focus_plot", False
                ),
            )


class UltrasoundPretrainingExperiment(MontySupervisedObjectPretrainingExperiment):
    """Supervised pretraining on offline ultrasound data.

    Identical to the base class, except that the exploration loop ends cleanly when the
    (offline) environment runs out of data points for the current object. This lets us
    learn every available point for each object regardless of how many were collected,
    matching the behaviour of the old iterator-based dataloader.
    """

    def run_episode(self) -> None:
        self.pre_episode()
        all_lm_ids = [lm.learning_module_id for lm in self.model.learning_modules]
        if set(self.supervised_lm_ids) == set(all_lm_ids):
            self.model.switch_to_exploratory_step()

        ctx = RuntimeContext(rng=self.rng)

        num_steps = 0
        actions: list[Action] = []
        while True:
            try:
                observations, proprioceptive_state = self.env_interface.step(actions)
            except StopIteration:
                # Offline dataset exhausted for this object; learn what we have.
                break

            num_steps += 1
            try:
                actions = self.model.step(ctx, observations, proprioceptive_state)
            except StopIteration:
                break
            if self.model.is_done:
                break
            if self.model.episode_steps >= self.max_total_steps:
                break

        target = self.env_interface.primary_target
        self.model.detected_object = self.model.primary_target["object"]
        for lm in self.model.learning_modules:
            if lm.learning_module_id in self.supervised_lm_ids:
                lm.detected_object = target["object"]
                lm.buffer.stats["possible_matches"] = [target["object"]]
                lm.buffer.stats["detected_location_on_model"] = (
                    self.first_epoch_object_location[target["object"]]
                )
                lm.buffer.stats["detected_location_rel_body"] = np.array(
                    target["position"]
                )
                lm.buffer.stats["detected_rotation"] = target["euler_rotation"]
                lm.detected_rotation_r = Rotation.from_quat(
                    target["quat_rotation"]
                ).inv()
                lm.buffer.stats["detected_scale"] = target["scale"]
            else:
                lm.possible_matches = []
                lm.detected_object = None
                lm.detected_pose = None
                lm.detected_rotation_r = None
                lm.buffer.stats["detected_location_on_model"] = None
                lm.buffer.stats["detected_location_rel_body"] = None
                lm.buffer.stats["detected_rotation"] = None
                lm.buffer.stats["detected_scale"] = None

        self.post_episode(num_steps)
