import pathlib
import sys
import unittest
from unittest.mock import Mock, call, patch

import numpy as np
import serial

sys.path.extend([str(pathlib.Path(__file__).parents[3]), str(pathlib.Path(__file__).parent)])

from al5d_test_support import FakeSerial, pulse_exp
from robot.al5d.constants import SERVO_COUNT
from robot.al5d.pulse_controller import PulseController


class TestPulseController(unittest.TestCase):
    def test_connection_uses_backup_after_primary_failure(self):
        backup = FakeSerial()
        with patch("robot.al5d.pulse_controller.serial.Serial", side_effect=[serial.SerialException(), backup]) as serial_open:
            controller = PulseController(pulse_exp())
        self.assertIs(controller.sp, backup)
        self.assertEqual(serial_open.call_args_list[0].args, ("/dev/primary", 9600))
        self.assertEqual(serial_open.call_args_list[1].args, ("/dev/backup", 9600))

    def test_bulk_command_serialization_and_state_update(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.sp = FakeSerial([b"+"])
        controller.positions_pulse = np.full(SERVO_COUNT, 1500, dtype=int)
        controller.control_pulses(np.arange(SERVO_COUNT) * 100 + 1000)
        self.assertEqual(controller.sp.writes, [b"#0 P 1000 #1 P 1100 #2 P 1200 #3 P 1300 #4 P 1400 #5 P 1500  T50\r", b"Q\r"])
        np.testing.assert_array_equal(controller.positions_pulse, [1000, 1100, 1200, 1300, 1400, 1500])

    def test_single_servo_command_and_completion_retry(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.sp = FakeSerial([b"-", b"+"])
        controller.positions_pulse = np.full(SERVO_COUNT, 1500, dtype=int)
        controller.control_servo_pulse(2, 1600, 123)
        self.assertEqual(controller.sp.writes, [b"#2 P1600 S123\r", b"Q\r", b"Q\r"])
        self.assertTrue(controller.command_finished)
        self.assertEqual(controller.positions_pulse[2], 1600)

    def test_failed_command_does_not_update_state_and_timeout_raises(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.sp = FakeSerial()
        controller.positions_pulse = np.full(SERVO_COUNT, 1500, dtype=int)
        controller.wait_until_complete = Mock(side_effect=TimeoutError("timed out"))
        with self.assertRaises(TimeoutError):
            controller.control_pulses([1000] * SERVO_COUNT)
        np.testing.assert_array_equal(controller.positions_pulse, [1500] * SERVO_COUNT)
        controller.wait_until_complete = PulseController.wait_until_complete.__get__(controller)
        with self.assertRaises(TimeoutError):
            controller.wait_until_complete(timeout=0)

    def test_start_and_stop_sequences(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.sp = FakeSerial()
        controller.command_finished = True
        controller.start_robot = Mock()
        with patch("robot.al5d.pulse_controller.time.sleep") as sleep:
            controller.stop_robot()
        controller.start_robot.assert_called_once_with(200)
        sleep.assert_called_once_with(5)
        self.assertEqual(controller.sp.writes, [f"#{servo} P0\r".encode() for servo in range(SERVO_COUNT)])

    def test_start_uses_requested_speed_for_every_servo(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.pulse_position_default = 1500
        controller.control_servo_pulse = Mock()
        controller.start_robot(321)
        self.assertEqual(
            controller.control_servo_pulse.call_args_list,
            [call(servo, 1500, 321) for servo in range(SERVO_COUNT)],
        )

    def test_relative_pulse_outside_hardware_limits_raises(self):
        controller = PulseController.__new__(PulseController)
        controller.exp = pulse_exp()
        controller.positions_pulse = np.full(SERVO_COUNT, 1500, dtype=int)
        controller.control_servo_pulse = Mock()
        with self.assertRaises(ValueError):
            controller.control_servo_relative_pulse(0, 2000)


if __name__ == "__main__":
    unittest.main()
