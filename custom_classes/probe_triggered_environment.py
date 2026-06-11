# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Live, probe-triggered ultrasound environment.

Streams ultrasound images from an :class:`~custom_classes.server.ImageServer` and
fetches the matching probe pose from a VIVE tracker HTTP service. Each observation is
optionally saved to disk, forming the basis of an offline dataset for a given object.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import numpy as np
import requests
from tbp.monty.frameworks.models.buffer import BufferEncoder

from custom_classes.environment import (
    PROBE_SENSOR_OFFSET,
    AGENT_ID,
    UltrasoundEnvironment,
)
from custom_classes.server import ImageServer


class ProbeTriggeredUltrasoundEnvironment(UltrasoundEnvironment):
    """Ultrasound environment that is interactively triggered by use of the probe."""

    def __init__(
        self,
        patch_size: int = 256,
        image_listen_port: int = 8000,
        vive_url: str = "http://localhost:3001/pose",
        save_path: str | None = None,
    ):
        super().__init__(patch_size=patch_size, data_path=None)
        # A single, live object is being scanned.
        self.scene_names = ["live_object"]
        self.object_names = ["live_object"]

        self.image_listen_port = image_listen_port
        self.server = ImageServer()
        self.server.start(port=image_listen_port)
        self.vive_url = vive_url
        self.vive_pose = None
        self.save_path = save_path
        self._save_counter = 0

        if self.save_path is not None and not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _load_raw_data_point(self, step: int):  # noqa: ARG002
        """Stream the next image + probe pose (the step index is ignored)."""
        complete_data = False
        while not complete_data:
            self.vive_pose = None
            current_ultrasound_image, metadata = self.server.get_next_image()
            self.vive_pose = self.get_vive_pose(metadata["epoch"])
            if self.vive_pose is not None:
                complete_data = True
            else:
                print(
                    "Did not get vive pose data, please ensure the vive service is "
                    "running and then take another image"
                )

        state = self.get_state()

        if self.save_path is not None:
            obs = {
                AGENT_ID: {
                    "ultrasound": {
                        "img": current_ultrasound_image,
                        "metadata": metadata,
                    },
                }
            }
            self.save_data(obs, state)

        return current_ultrasound_image, state

    def get_state(self) -> Dict[str, Any]:
        """Build the agent state from the latest VIVE pose."""
        pos = self.vive_pose["pose"]["position"]
        agent_position = [pos["x"], pos["y"], pos["z"]]

        rot = self.vive_pose["pose"]["rotation"]
        agent_rotation = [rot["w"], rot["x"], rot["y"], rot["z"]]

        return {
            AGENT_ID: {
                "sensors": {
                    "ultrasound": {
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "position": list(PROBE_SENSOR_OFFSET),
                    },
                },
                "rotation": agent_rotation,
                "position": agent_position,
            }
        }

    def save_data(self, obs, state):
        """Save the observation and state to ``{save_counter}.json``."""
        data = {
            "obs": copy.deepcopy(obs),
            "state": state,
        }
        with open(
            os.path.join(self.save_path, f"{self._save_counter}.json"), "w"
        ) as f:
            json.dump(data, f, cls=BufferEncoder)
        self._save_counter += 1

    def get_vive_pose(self, epoch: float) -> Dict[str, Any] | None:
        try:
            response = requests.get(f"{self.vive_url}?epoch={epoch}")
            if not 200 <= response.status_code < 300:
                return None
            return response.json()["data"]
        except Exception:
            return None

    def close(self):
        super().close()
