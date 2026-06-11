# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Environment interface for ultrasound experiments.

Replaces the old ``UltrasoundDataLoader``. One episode is run per scanned object;
between episodes the interface tells the environment to switch to the next scene
(scanned object) and updates the ``primary_target`` used for supervised learning and
logging.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation
from tbp.monty.experiment.environment import (
    OneObjectPerEpisodeInterface,
    normalize_transforms,
)
from tbp.monty.frameworks.environment_utils.transforms import Transform

logger = logging.getLogger(__name__)


def _scipy_to_numpy_quat(q):
    """Convert a scipy quaternion ``[x, y, z, w]`` to a numpy/monty ``[w, x, y, z]``."""
    return np.array([q[3], q[0], q[1], q[2]])


class UltrasoundEnvironmentInterface(OneObjectPerEpisodeInterface):
    """Interface that cycles through scanned ultrasound objects, one per episode.

    Mirrors the structure of
    :class:`~tbp.monty.experiment.environment.SaccadeOnImageInterface`: we do not use
    the object-initializer / ``add_object`` machinery (the environment is image-based),
    so we set the relevant attributes directly instead of calling ``super().__init__``.
    """

    def __init__(
        self,
        env,
        rng: np.random.RandomState,
        seed: int | None = None,
        experiment_mode=None,
        transform: Transform | Sequence[Transform] | None = None,
        parent_to_child_mapping: Mapping[str, Sequence[str]] | None = None,
        positioning_procedures=None,
        *_args,
        **_kwargs,
    ):
        self.env = env
        self.rng = rng
        self.seed = seed
        self.experiment_mode = experiment_mode
        self.transforms = normalize_transforms(transform)
        self.reset(self.rng)

        self.object_names = self.env.object_names
        self.n_objects = len(self.object_names)
        self.current_object = 0
        self.episodes = 0
        self.epochs = 0
        self.primary_target = None
        self.consistent_child_objects = None
        self.parent_to_child_mapping = parent_to_child_mapping or {}
        self._positioning_procedures = positioning_procedures

    def pre_epoch(self):
        self.change_object_by_idx(0)

    def post_episode(self):
        self.cycle_object()
        self.episodes += 1

    def post_epoch(self):
        self.epochs += 1

    def cycle_object(self):
        """Switch to the next scanned object."""
        next_object = (self.current_object + 1) % self.n_objects
        logger.info(
            f"\n\nGoing from {self.current_object} to {next_object} of {self.n_objects}"
        )
        self.change_object_by_idx(next_object)

    def change_object_by_idx(self, idx: int):
        """Load the scene for object ``idx`` and update the primary target.

        Raises:
            IndexError: If ``idx`` is outside the range [0, self.n_objects).
        """
        if not 0 <= idx < self.n_objects:
            raise IndexError(f"idx must satisfy 0 <= idx < {self.n_objects}, got {idx}")

        self.env.switch_to_scene(idx)
        self.current_object = idx

        # We do not have a ground-truth rotation for the scanned object, so we use the
        # identity rotation. The patch poses come from the probe trackers (via the SM).
        euler_rotation = np.zeros(3)
        q = Rotation.from_euler("xyz", euler_rotation, degrees=True).as_quat()
        quat_rotation = _scipy_to_numpy_quat(q)
        self.primary_target = {
            "object": self.object_names[idx],
            "semantic_id": 0,
            "rotation": quat_rotation,
            "euler_rotation": euler_rotation,
            "quat_rotation": q,
            "position": np.zeros(3),
            "scale": np.ones(3),
        }
        logger.info(f"New primary target: {self.primary_target['object']}")
