"""Custom experiment classes for ultrasound experiments."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.experiment.environment import SaccadeOnImageInterface
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.geometry import Rotation

from custom_classes.interface import (
    ProbeTriggeredUltrasoundInterface,
    UltrasoundJSONInterface,
)
from custom_classes.plotting import plot_combined_figure

logger = logging.getLogger(__name__)


@dataclass
class FeatureLogger:
    """Logger to store features for plotting."""

    column_points: List[np.ndarray] = field(default_factory=list)
    center_edge: Optional[Tuple[float, float]] = None
    fitted_circle: Optional[Tuple[float, float, float]] = None
    point_normal: Optional[np.ndarray] = None
    curvature: Optional[np.ndarray] = None
    mean_depth: float = 0.0

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class MontyUltrasoundSupervisedObjectPretrainingExperiment(
    MontySupervisedObjectPretrainingExperiment
):
    """Supervised learning experiment for ultrasound data."""

    def run_episode(self) -> None:
        """Run supervised episode, ending cleanly when JSON observations are exhausted."""
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
                break

            num_steps += 1
            if self.show_sensor_output:
                is_saccade_on_image_env_interface = isinstance(
                    self.env_interface, SaccadeOnImageInterface
                )
                self.live_plotter.show_observations(
                    *self.live_plotter.hardcoded_assumptions(observations, self.model),
                    num_steps,
                    is_saccade_on_image_env_interface,
                )
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

        if len(self.model.learning_modules) > 1:
            for i, lm in enumerate(self.model.learning_modules):
                if i == 0:
                    first_pos = self.sensor_pos[0]
                else:
                    lm_offset = self.sensor_pos[i] - first_pos
                    if lm.detected_object:
                        lm.buffer.stats["detected_location_rel_body"] += lm_offset
                        lm_offset_model_rf = lm.detected_rotation_r.apply(lm_offset)
                        lm.buffer.stats["detected_location_on_model"] += (
                            lm_offset_model_rf
                        )

        self.post_episode(num_steps)

    @property
    def logger_args(self):
        args = super().logger_args
        if isinstance(self.env_interface, UltrasoundJSONInterface):
            args.update(target=self.env_interface.primary_target)
        return args

    def run_epoch(self):
        self.pre_epoch()
        if isinstance(self.env_interface, UltrasoundJSONInterface):
            for _ in self.env_interface.scenes:
                self.run_episode()
        elif isinstance(self.env_interface, ProbeTriggeredUltrasoundInterface):
            try:
                while True:
                    self.run_episode()
            except KeyboardInterrupt:
                logger.info("Probe-triggered data collection interrupted.")
        else:
            self.run_episode()
        self.post_epoch()


class UltrasoundExperiment(MontyObjectRecognitionExperiment):
    """Custom experiment class that adds plotting functionality."""

    def __init__(self, config: DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
        self.plotting_config = config_dict.get("plotting_config")
        self.feature_logger = FeatureLogger()
        super().__init__(config)

    @property
    def logger_args(self):
        args = super().logger_args
        if isinstance(self.env_interface, UltrasoundJSONInterface):
            args.update(target=self.env_interface.primary_target)
        return args

    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.feature_logger = FeatureLogger()

    def run_epoch(self):
        self.pre_epoch()
        if isinstance(self.env_interface, UltrasoundJSONInterface):
            for _ in self.env_interface.scenes:
                self.run_episode()
        elif isinstance(self.env_interface, ProbeTriggeredUltrasoundInterface):
            try:
                while True:
                    self.run_episode()
            except KeyboardInterrupt:
                logger.info("Probe-triggered inference interrupted.")
        else:
            self.run_episode()
        self.post_epoch()

    def run_episode_steps(self) -> int:
        step = 0
        ctx = RuntimeContext(rng=self.rng)
        actions: list[Action] = []
        agent_id = AgentID("agent_id_0")
        patch_sensor_id = SensorID("patch")

        while True:
            try:
                observations, proprioceptive_state = self.env_interface.step(actions)
            except StopIteration:
                self.model.set_is_done()
                return step

            if self.plotting_config and self.plotting_config.get("enabled", False):
                if step % self.plotting_config.get("plot_frequency", 1) == 0:
                    self._maybe_plot(step, observations, agent_id, patch_sensor_id)

            if self.model.check_reached_max_matching_steps(self.max_steps):
                return step

            if step >= self.max_total_steps:
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
                self.model.set_is_done()
                return step

            if self.model.is_done:
                return step

            step += 1

    def _maybe_plot(
        self,
        step: int,
        observations,
        agent_id: AgentID,
        patch_sensor_id: SensorID,
    ) -> None:
        patch_data = observations[agent_id][patch_sensor_id]
        depth = patch_data.get("patch_depth", 0.0)
        sensor_module = self.model.sensor_modules[0]
        features = sensor_module.plotting_data

        save_path = None
        if self.plotting_config.get("save_path"):
            os.makedirs(self.plotting_config["save_path"], exist_ok=True)
            save_path = os.path.join(
                self.plotting_config["save_path"], f"step_{step}.png"
            )

        if self.plotting_config.get("plot_patch_features", False):
            full_image = self.env_interface.env.get_full_image()
            plot_combined_figure(
                input_image=full_image,
                patch_image=patch_data["img"],
                all_column_points_tuples=features.get("column_points", []),
                center_edge_point=features.get("center_edge", None),
                fitted_circle=features.get("fitted_circle", None),
                point_normal=features.get("point_normal", None),
                curvature=features.get("curvature", 0),
                depth_meters=features.get("mean_depth", depth),
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
