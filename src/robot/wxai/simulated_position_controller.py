"""Strict headless simulation of the WidowX AI position-controller interface."""

from copy import copy

from robot.widowx.simulated_position_controller import (
    SimulatedPositionController as WidowXSimulatedPositionController,
)

from .position import WXAICommand, WXAIPose


class SimulatedPositionController(WidowXSimulatedPositionController):
    """The pose jumps to the target; no kinematics, no physics, no dependencies."""

    POSE = WXAIPose
    COMMAND = WXAICommand

    def __init__(self, exp):
        super().__init__(exp)
        self.joints = []

    def move(self, command, moving_time=None, blocking=True):
        super().move(command, moving_time, blocking)
        if isinstance(command, WXAICommand) and command.joints is not None:
            self.joints = copy(command.joints)

    def get_state(self):
        state = super().get_state()
        state["joint_positions"] = copy(self.joints)
        return state

    def update(self, dt):
        """Called once per collection tick; nothing to simulate."""
