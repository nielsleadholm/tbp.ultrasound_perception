# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Custom Monty model variants for ultrasound experiments."""

from __future__ import annotations

import logging

import requests
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

logger = logging.getLogger(__name__)


class MontyForEvidenceGraphMatchingWithGoalStateServer(
    MontyForEvidenceGraphMatching
):
    """Evidence-graph Monty that publishes goal states to the probe visualizer."""

    def _step_motor_system(self, ctx, observations, proprioceptive_state):
        super()._step_motor_system(ctx, observations, proprioceptive_state)
        self._send_goal_state_to_server()

    def _send_goal_state_to_server(self) -> None:
        selected_goals = getattr(
            self.motor_system._policy_selector, "_selected_goals", []
        )
        if not selected_goals:
            return

        goal = selected_goals[-1]
        if goal is None or not goal.use_state:
            return

        try:
            pose_vectors = goal.morphological_features["pose_vectors"]
            location = goal.location
        except (AttributeError, KeyError, TypeError) as error:
            logger.warning("Could not serialize goal state: %s", error)
            return

        try:
            goal_state_url = "http://localhost:3003/goal_state"
            location_as_list = [float(x) for x in location]
            pose_vectors_as_list = [float(x) for x in pose_vectors[0, :]]
            response = requests.post(
                goal_state_url,
                json={
                    "goal_state": {
                        "location": location_as_list,
                        "pose_vectors": pose_vectors_as_list,
                    }
                },
                timeout=0.1,
            )
            if response.status_code == 200:
                logger.info("Goal state successfully sent to server.")
            else:
                logger.warning(
                    "Failed to send goal state. Server responded with %s",
                    response.status_code,
                )
        except requests.RequestException as error:
            logger.warning("Error sending goal state to server: %s", error)
