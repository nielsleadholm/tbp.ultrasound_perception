# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Observation transforms for ultrasound experiments."""

from __future__ import annotations

import numpy as np
import quaternion as qt

from custom_classes.patch_extraction import find_patch_with_highest_gradient
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environment_utils.transforms import TransformContext
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.sensors import SensorID

AGENT_ID = AgentID("agent_id_0")
ULTRASOUND_SENSOR_ID = SensorID("ultrasound")
PATCH_SENSOR_ID = SensorID("patch")


class UltrasoundPatchTransform:
    """Extract an edge-focused patch from a full ultrasound image."""

    def __init__(self, agent_id: AgentID = AGENT_ID, patch_size: int = 256):
        self.agent_id = agent_id
        self.patch_size = patch_size

    def __call__(
        self, observations: Observations, ctx: TransformContext
    ) -> Observations:
        agent_obs = observations[self.agent_id]
        ultrasound_obs = agent_obs[ULTRASOUND_SENSOR_ID]
        full_image = np.array(ultrasound_obs["img"])

        patch, patch_pixel_start = find_patch_with_highest_gradient(
            full_image,
            patch_size=self.patch_size,
        )

        patch_observation = {
            "img": patch,
            "patch_pixel_start": patch_pixel_start,
            "full_image_height": full_image.shape[0],
        }
        if ctx.state is not None:
            agent_state = ctx.state[self.agent_id]
            ultrasound_state = agent_state.sensors[ULTRASOUND_SENSOR_ID]
            patch_observation["proprioceptive_state_patch"] = {
                "position": np.array(ultrasound_state.position),
                "rotation": ultrasound_state.rotation,
            }
            patch_observation["proprioceptive_state_agent"] = {
                "position": np.array(agent_state.position),
                "rotation": agent_state.rotation,
            }

        observations[self.agent_id] = AgentObservations(
            {
                PATCH_SENSOR_ID: SensorObservation(patch_observation),
            }
        )
        return observations
