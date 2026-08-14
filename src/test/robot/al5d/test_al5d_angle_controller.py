import pathlib
import sys
import unittest

import numpy as np

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import CapturingPulseController, angle_exp
from robot.al5d.angle_controller import AngleController


class FailingPulseController(CapturingPulseController):
    def control_pulses(self, pulses):
        raise RuntimeError("pulse command failed")


class TestAngleController(unittest.TestCase):
    def make_controller(self, pulse_controller=None):
        controller = AngleController(angle_exp(), pulse_controller or CapturingPulseController())
        return controller

    def test_direct_angle_and_gripper_commands_use_default_speed(self):
        controller = self.make_controller()
        controller.control_servo_angle(2, 90)
        controller.control_gripper(100)
        self.assertEqual(controller.pulse_controller.servo_calls, [(2, 1500, 100), (5, 1000, 100)])

    def test_bulk_command_and_invalid_inputs(self):
        controller = self.make_controller()
        controller.control_angles(np.array([0, 45, 90, 135, 180]), 100)
        np.testing.assert_array_equal(controller.pulse_controller.pulse_calls[0], [500, 1000, 1500, 2000, 2500, 1000])
        for servo, angle in ((-1, 90), (5, 90), (0, -1)):
            with self.assertRaises(ValueError):
                controller.control_servo_angle(servo, angle)
        for gripper in (-1, 101):
            with self.assertRaises(ValueError):
                controller.control_gripper(gripper)
        with self.assertRaises(ValueError):
            controller.control_angles([0] * 4, 100)

    def test_failed_bulk_command_does_not_update_state(self):
        controller = self.make_controller(FailingPulseController())
        before = controller.positions.copy()
        with self.assertRaisesRegex(RuntimeError, "pulse command failed"):
            controller.control_angles(np.array([0, 45, 90, 135, 180]), 100)
        np.testing.assert_array_equal(controller.positions, before)


if __name__ == "__main__":
    unittest.main()
