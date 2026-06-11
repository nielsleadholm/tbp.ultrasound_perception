# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Environment interfaces for ultrasound experiments."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from omegaconf import ListConfig

from custom_classes.environment import JSONDatasetUltrasoundEnvironment
from custom_classes.probe_triggered_environment import (
    ProbeTriggeredUltrasoundEnvironment,
)
from tbp.monty.experiment.environment import Interface, normalize_transforms
from tbp.monty.frameworks.experiments.mode import ExperimentMode

logger = logging.getLogger(__name__)


class UltrasoundJSONInterface(Interface):
    """Interface for offline JSON ultrasound datasets with one object per episode."""

    def __init__(
        self,
        env: JSONDatasetUltrasoundEnvironment,
        rng: np.random.RandomState,
        seed: int,
        experiment_mode: ExperimentMode,
        scenes: Sequence[int] | ListConfig | None = None,
        transform=None,
    ):
        self.env = env
        self.rng = rng
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.transforms = normalize_transforms(transform)
        self.scenes = list(scenes) if scenes is not None else list(range(len(env.object_names)))
        self.object_names = env.object_names
        self.current_scene_idx = 0
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self.reset(self.rng)

    def pre_epoch(self):
        self.change_object_by_idx(0)

    def post_episode(self):
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        next_scene_idx = self.current_scene_idx + 1
        if next_scene_idx < len(self.scenes):
            self.change_object_by_idx(next_scene_idx)

    def change_object_by_idx(self, idx: int):
        if not 0 <= idx < len(self.scenes):
            raise IndexError(
                f"idx must satisfy 0 <= idx < {len(self.scenes)}, got {idx}"
            )
        scene = self.scenes[idx]
        logger.info("Changing to ultrasound object index %s (scene %s)", idx, scene)
        self.env.current_scene = scene
        self.env.step_count = 0
        self.current_scene_idx = idx
        self.primary_target = {
            "object": self.object_names[scene],
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [1, 0, 0, 0],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }


class ProbeTriggeredUltrasoundInterface(Interface):
    """Interface for live probe-triggered ultrasound data collection."""

    def __init__(
        self,
        env: ProbeTriggeredUltrasoundEnvironment,
        rng: np.random.RandomState,
        seed: int,
        experiment_mode: ExperimentMode,
        transform=None,
    ):
        self.env = env
        self.rng = rng
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.transforms = normalize_transforms(transform)
        self.episodes = 0
        self.epochs = 0
        self.primary_target = {
            "object": "probe_triggered",
            "rotation": (1.0, 0.0, 0.0, 0.0),
            "euler_rotation": np.array([0, 0, 0]),
            "quat_rotation": [1, 0, 0, 0],
            "position": np.array([0, 0, 0]),
            "scale": [1.0, 1.0, 1.0],
        }
        self.reset(self.rng)

    def post_episode(self):
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1
