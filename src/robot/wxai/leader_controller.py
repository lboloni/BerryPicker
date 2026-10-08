"""A WidowX AI leader arm, moved by hand, whose joints are the commands of a follower."""

import trossen_arm

from .position_controller import create_driver


class LeaderController:
    """The leader arm runs in external effort mode with zero effort: it is
    gravity compensated and moves freely by hand. The follower efforts can be
    fed back to the leader, scaled by force_feedback_gain (0 disables it),
    as in the teleoperation demo of trossen_arm."""

    def __init__(self, exp, driver=None):
        self.exp = exp
        self.driver = driver if driver is not None else create_driver(exp)
        self.started = False

    def start(self):
        self.driver.configure(
            getattr(trossen_arm.Model, self.exp["model"]),
            getattr(trossen_arm.StandardEndEffector, self.exp["end_effector"]),
            self.exp["ip_address"],
            self.exp["clear_error"],
        )
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts([0.0] * self.driver.get_num_joints(), 0.0, False)
        self.started = True

    def get_joints(self):
        """Six arm joints in rad and the gripper in m."""
        if not self.started:
            raise RuntimeError("WidowX AI leader is not started")
        return [float(value) for value in self.driver.get_all_positions()]

    def feedback(self, follower_efforts):
        """Reflect the follower efforts to the leader."""
        gain = self.exp["force_feedback_gain"]
        if gain > 0:
            self.driver.set_all_external_efforts(
                [-gain * effort for effort in follower_efforts], 0.0, False)

    def stop(self):
        self.started = False
        self.driver.set_all_modes(trossen_arm.Mode.idle)
        self.driver.cleanup()
