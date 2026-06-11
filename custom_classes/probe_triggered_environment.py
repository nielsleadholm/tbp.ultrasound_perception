# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Live probe-triggered ultrasound environment."""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import quaternion as qt
import requests
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
from tbp.monty.frameworks.models.buffer import BufferEncoder
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

from custom_classes.server import ImageServer

logger = logging.getLogger(__name__)

AGENT_ID = AgentID("agent_id_0")
ULTRASOUND_SENSOR_ID = SensorID("ultrasound")


class ProbeTriggeredUltrasoundEnvironment(SimulatedEnvironment, SimulatedObjectEnvironment):
    """Environment that waits for probe-triggered ultrasound images and Vive poses."""

    def __init__(
        self,
        image_listen_port: int = 8000,
        vive_url: str = "http://localhost:3001/pose",
        save_path: str | Path | None = None,
    ):
        self.image_listen_port = image_listen_port
        self.server = ImageServer()
        self.server.start(port=image_listen_port)
        self.vive_url = vive_url
        self.vive_pose: dict[str, Any] | None = None
        self.save_path = Path(save_path) if save_path is not None else None
        self.full_image: np.ndarray | None = None
        self.step_count = 0
        self._agents = [
            type(
                "FakeAgent",
                (object,),
                {"action_space_type": "distant_agent_no_translation"},
            )()
        ]

        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)

    def step(
        self, actions: Sequence[Action]
    ) -> tuple[Observations, ProprioceptiveState]:
        del actions
        complete_data = False
        while not complete_data:
            self.vive_pose = None
            current_ultrasound_image, metadata = self.server.get_next_image()
            self.full_image = current_ultrasound_image
            self.vive_pose = self._get_vive_pose(metadata["epoch"])
            if self.vive_pose is not None:
                complete_data = True
            else:
                logger.warning(
                    "Did not get vive pose data; ensure the vive service is running"
                )

        observations = Observations(
            {
                AGENT_ID: AgentObservations(
                    {
                        ULTRASOUND_SENSOR_ID: SensorObservation(
                            {
                                "img": current_ultrasound_image,
                                "metadata": metadata,
                            }
                        ),
                    }
                )
            }
        )
        state = self._state()
        if self.save_path is not None:
            self._save_data(observations, state)
        self.step_count += 1
        return observations, state

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self.step_count = 0
        return self.step([])

    def get_full_image(self) -> np.ndarray:
        if self.full_image is None:
            return np.zeros((256, 256), dtype=np.float32)
        return self.full_image

    def add_object(self, *args, **kwargs):
        raise NotImplementedError(
            "ProbeTriggeredUltrasoundEnvironment does not support adding objects"
        )

    def remove_all_objects(self) -> None:
        raise NotImplementedError(
            "ProbeTriggeredUltrasoundEnvironment does not support removing objects"
        )

    def close(self) -> None:
        pass

    def _state(self) -> ProprioceptiveState:
        pos = self.vive_pose["pose"]["position"]
        agent_position = np.array([pos["x"], pos["y"], pos["z"]])
        rot = self.vive_pose["pose"]["rotation"]
        agent_rotation = qt.quaternion(rot["w"], rot["x"], rot["y"], rot["z"])
        return ProprioceptiveState(
            {
                AGENT_ID: AgentState(
                    sensors={
                        ULTRASOUND_SENSOR_ID: SensorState(
                            rotation=qt.quaternion(1, 0, 0, 0),
                            position=(0.0, 0.028, 0.105),
                        )
                    },
                    rotation=agent_rotation,
                    position=tuple(agent_position),
                )
            }
        )

    def _save_data(
        self, observations: Observations, state: ProprioceptiveState
    ) -> None:
        obs_dict = {
            "agent_id_0": {
                "ultrasound": {
                    "img": observations[AGENT_ID][ULTRASOUND_SENSOR_ID]["img"].tolist(),
                }
            }
        }
        agent_state = state[AGENT_ID]
        state_dict = {
            "agent_id_0": {
                "sensors": {
                    "ultrasound": {
                        "rotation": agent_state.sensors[ULTRASOUND_SENSOR_ID].rotation,
                        "position": np.array(
                            agent_state.sensors[ULTRASOUND_SENSOR_ID].position
                        ),
                    }
                },
                "rotation": agent_state.rotation,
                "position": np.array(agent_state.position),
            }
        }
        data = {"obs": copy.deepcopy(obs_dict), "state": state_dict}
        output_path = self.save_path / f"{self.step_count}.json"
        with output_path.open("w") as file:
            json.dump(data, file, cls=BufferEncoder)

    def _get_vive_pose(self, epoch: float) -> dict[str, Any] | None:
        try:
            response = requests.get(f"{self.vive_url}?epoch={epoch}", timeout=5)
            if not 200 <= response.status_code < 300:
                return None
            return response.json()["data"]
        except Exception:
            return None
