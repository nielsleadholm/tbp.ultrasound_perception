# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Monty model that publishes goal states to an HTTP server.

Used by the probe-triggered data collection experiments so that proposed goal states
can be visualized in the 3D visualization of the phantom setup.
"""

from __future__ import annotations

import logging

import requests
from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

logger = logging.getLogger(__name__)

GOAL_STATE_URL = "http://localhost:3003/goal_state"


class MontyForEvidenceGraphMatchingWithGoalStateServer(MontyForEvidenceGraphMatching):
    """Evidence-matching Monty that POSTs the current driving goal state to a server.

    In the latest ``tbp.monty``, goals proposed by the learning modules' goal-state
    generators are aggregated in ``self._goals`` during ``_pass_goals`` (before being
    handed to the motor system). We override ``_pass_goals`` to additionally publish the
    highest-confidence usable goal to an HTTP server.
    """

    def _pass_goals(self) -> None:
        super()._pass_goals()
        self._publish_goal_state()

    def _publish_goal_state(self) -> None:
        usable_goals = [
            g for g in self._goals if getattr(g, "use_state", False) and g.location is not None
        ]
        if not usable_goals:
            return

        goal = max(usable_goals, key=lambda g: g.confidence)
        try:
            pose_vectors = goal.morphological_features["pose_vectors"]
            location_as_list = [float(x) for x in goal.location]
            pose_vectors_as_list = [float(x) for x in pose_vectors[0, :]]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error serializing goal state: {e}")
            return

        data_to_send = {
            "goal_state": {
                "location": location_as_list,
                "pose_vectors": pose_vectors_as_list,
            }
        }
        try:
            response = requests.post(GOAL_STATE_URL, json=data_to_send, timeout=0.1)
            if response.status_code == 200:
                logger.debug("Goal state successfully sent to server.")
            else:
                logger.warning(
                    f"Failed to send goal state. Server responded with "
                    f"{response.status_code}"
                )
        except requests.RequestException as e:
            logger.warning(f"Error sending goal state to server: {e}")
