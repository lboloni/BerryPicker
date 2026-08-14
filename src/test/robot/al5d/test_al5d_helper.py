import pathlib
import sys
import unittest

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import angle_exp, pulse_exp
from robot.al5d.helper import RobotHelper


class TestRobotHelper(unittest.TestCase):
    def test_range_mapping_and_servo_pulse_boundaries(self):
        self.assertEqual(RobotHelper.map_ranges(0, 0, 10, 10, 20), 10)
        self.assertEqual(RobotHelper.map_ranges(10, 0, 10, 10, 20), 20)
        self.assertEqual(RobotHelper.servo_angle_to_pulse(angle_exp(), pulse_exp(), 0, 0), (500, False))
        self.assertEqual(RobotHelper.servo_angle_to_pulse(angle_exp(), pulse_exp(), 0, 180), (2500, False))
        with self.assertRaises(ValueError):
            RobotHelper.servo_angle_to_pulse(angle_exp(), pulse_exp(), 0, 181)

    def test_corrected_pulse_outside_hardware_limits_raises(self):
        pulse = pulse_exp()
        pulse["PULSE_CORRECTION"][0] = 1
        with self.assertRaises(ValueError):
            RobotHelper.servo_angle_to_pulse(angle_exp(), pulse, 0, 180)

    def test_unconstrained_angle_conversion_returns_not_constrained(self):
        exp = pulse_exp()
        exp.update(angle_exp())
        self.assertEqual(RobotHelper.angle_to_pulse(exp, 90, constrain=False), (1500, False))


if __name__ == "__main__":
    unittest.main()
