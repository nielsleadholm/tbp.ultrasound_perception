# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Motor policies for ultrasound experiments."""

from __future__ import annotations

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    InformedPolicy,
    MotorPolicyResult,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState


class UltrasoundMotorPolicy(InformedPolicy):
    """Motor policy for human-operated ultrasound probes.

    The environment advances to the next stored observation on every step, so the
    policy does not need to generate movement actions during offline experiments.
    """

    def __init__(self, action_sampler, agent_id, use_goal_driven_actions=False):
        super().__init__(
            action_sampler=action_sampler,
            agent_id=agent_id,
            use_goal_driven_actions=use_goal_driven_actions,
        )

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,  # noqa: ARG002
        state: MotorSystemState,  # noqa: ARG002
        percept: Message,  # noqa: ARG002
        goal: Goal | None,
    ) -> MotorPolicyResult:
        if self.use_goal_driven_actions:
            result = self._goal_driven_actions(observations, state, goal)
            if result is not None:
                return result
        return MotorPolicyResult([])
