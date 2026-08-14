import pathlib
import sys
import unittest

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parents[2]))

from robot.al5d_angle_controller import AngleController
from robot.al5d_constants import SERVO_COUNT


class PulseController:
    exp = {
        "CST_PULSE_MIN": 500,
        "CST_PULSE_MAX": 2500,
        "PULSE_CORRECTION": [0] * SERVO_COUNT,
    }

    def control_pulses(self, pulses):
        self.pulses = pulses


class TestAL5DFixedLayout(unittest.TestCase):
    def test_angle_controller_uses_six_fixed_channels(self):
        controller = AngleController.__new__(AngleController)
        controller.exp = {
            "ANGLE_LIMITS": [[0, 90, 180]] * SERVO_COUNT,
            "CST_ANGLE_MIN": 0,
            "CST_ANGLE_MAX": 180,
        }
        controller.pulse_controller = PulseController()

        controller.control_angles(np.array([0, 45, 90, 135, 180]), 100)

        np.testing.assert_array_equal(
            controller.pulse_controller.pulses,
            np.array([500, 1000, 1500, 2000, 2500, 1000]),
        )


if __name__ == "__main__":
    unittest.main()
