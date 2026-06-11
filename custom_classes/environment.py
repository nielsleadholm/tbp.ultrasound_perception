# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Ultrasound environments compatible with the latest tbp.monty APIs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import PIL
import quaternion as qt
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.environment import (
    SimulatedEnvironment,
    SimulatedObjectEnvironment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.path import monty_data_path

logger = logging.getLogger(__name__)

AGENT_ID = AgentID("agent_id_0")
ULTRASOUND_SENSOR_ID = SensorID("ultrasound")


def _to_quaternion(rotation) -> qt.quaternion:
    if isinstance(rotation, qt.quaternion):
        return rotation
    if isinstance(rotation, (list, tuple, np.ndarray)):
        return qt.quaternion(
            rotation[0],
            rotation[1],
            rotation[2],
            rotation[3],
        )
    raise TypeError(f"Unsupported rotation type: {type(rotation)}")


def dict_to_proprioceptive_state(state_dict: dict) -> ProprioceptiveState:
    """Convert a legacy ultrasound state dict to ProprioceptiveState."""
    agent_state = state_dict[AGENT_ID]
    sensors = {}
    for sensor_name, sensor_data in agent_state["sensors"].items():
        sensors[SensorID(sensor_name)] = SensorState(
            position=tuple(sensor_data["position"]),
            rotation=_to_quaternion(sensor_data["rotation"]),
        )
    return ProprioceptiveState(
        {
            AGENT_ID: AgentState(
                sensors=sensors,
                position=tuple(agent_state["position"]),
                rotation=_to_quaternion(agent_state["rotation"]),
            )
        }
    )


class UltrasoundEnvironment(SimulatedEnvironment, SimulatedObjectEnvironment):
    """Base ultrasound environment.

    NOTE: This is not fully functional (doesn't retrieve probe pose from trackers).
    Use JSONDatasetUltrasoundEnvironment or ProbeTriggeredUltrasoundEnvironment for
    actual data loading and state retrieval.
    """

    def __init__(self, data_path: str | Path | None = None):
        self.data_path = monty_data_path(data_path, "ultrasound/ultrasound_stream")
        self.full_image: np.ndarray | None = None
        self.scene_names = sorted(
            [name for name in os.listdir(self.data_path) if not name.startswith(".")]
        )
        self.object_names = [
            "_".join(name.split("_")[1:]) for name in self.scene_names
        ]
        self.current_scene = 0
        self.step_count = 0
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        del actions
        self.current_ultrasound_image = self._load_next_ultrasound_image()
        return self._build_observations(self.current_ultrasound_image), self._state()

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.step_count = 0
        return self.step([])

    def switch_to_next_scene(self) -> None:
        self.current_scene += 1
        self.step_count = 0

    def add_object(self, *args, **kwargs):
        raise NotImplementedError(
            "UltrasoundEnvironment does not support adding objects"
        )

    def remove_all_objects(self) -> None:
        raise NotImplementedError(
            "UltrasoundEnvironment does not support removing all objects"
        )

    def close(self) -> None:
        pass

    def get_full_image(self) -> np.ndarray:
        if self.full_image is None:
            logger.warning("full_image is None, returning fallback image")
            return np.zeros((256, 256), dtype=np.float32)
        return self.full_image

    def _build_observations(self, image: np.ndarray) -> Observations:
        return Observations(
            {
                AGENT_ID: AgentObservations(
                    {
                        ULTRASOUND_SENSOR_ID: SensorObservation(
                            {"img": np.array(image)}
                        ),
                    }
                )
            }
        )

    def _state(self) -> ProprioceptiveState:
        agent_position = np.array([0.0, 0.0, 0.0])
        agent_rotation = qt.quaternion(1, 0, 0, 0)
        return ProprioceptiveState(
            {
                AGENT_ID: AgentState(
                    sensors={
                        ULTRASOUND_SENSOR_ID: SensorState(
                            rotation=qt.quaternion(1, 0, 0, 0),
                            position=(0.0, 0.0, 0.0),
                        )
                    },
                    rotation=agent_rotation,
                    position=tuple(agent_position),
                )
            }
        )

    def _load_next_ultrasound_image(self) -> np.ndarray:
        current_img_path = (
            self.data_path
            / self.scene_names[self.current_scene]
            / f"img_{self.step_count}.png"
        )
        logger.info("Looking for ultrasound image from %s", current_img_path)
        while not current_img_path.exists():
            logger.info("Waiting for new ultrasound data...")
            import time

            time.sleep(1)

        while True:
            try:
                image = self._load_ultrasound_image(current_img_path)
                self.step_count += 1
                return image
            except PIL.UnidentifiedImageError:
                logger.info("Waiting for rgb file to finish streaming")
                import time

                time.sleep(1)

    def _load_ultrasound_image(self, img_path: Path) -> np.ndarray:
        rgb_image = np.array(PIL.Image.open(img_path))
        grayscale_image = np.mean(rgb_image, axis=2)
        self.full_image = grayscale_image
        return grayscale_image


class JSONDatasetUltrasoundEnvironment(UltrasoundEnvironment):
    """Offline ultrasound environment that loads paired JSON observations."""

    def __init__(self, data_path: str | Path | None = None):
        if data_path is not None:
            self.data_path = Path(os.path.expanduser(data_path))
        else:
            self.data_path = monty_data_path(None, "ultrasound_robot_lab_sparse")
        self.full_image = None
        self.scene_names = sorted(
            [name for name in os.listdir(self.data_path) if not name.startswith(".")]
        )
        self.object_names = [
            "_".join(name.split("_")[1:]) for name in self.scene_names
        ]
        self.current_scene = 0
        self.step_count = 0
        self.current_state: ProprioceptiveState | None = None
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        del actions
        image, state = self._load_next_data_point()
        if image is None or state is None:
            raise StopIteration
        self.step_count += 1
        return self._build_observations(image), state

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.step_count = 0
        return self.step([])

    def _load_next_data_point(
        self,
    ) -> tuple[np.ndarray | None, ProprioceptiveState | None]:
        json_path = (
            self.data_path
            / self.scene_names[self.current_scene]
            / f"{self.step_count}.json"
        )
        try:
            with json_path.open() as file:
                data = json.load(file)
        except FileNotFoundError:
            return None, None

        self.full_image = np.array(data["obs"]["agent_id_0"]["ultrasound"]["img"])
        data["state"]["agent_id_0"]["sensors"]["ultrasound"]["position"] = [
            0.0,
            0.028,
            0.105,
        ]
        self.current_state = dict_to_proprioceptive_state(data["state"])
        return self.full_image, self.current_state
