from copy import deepcopy

from robot.al5d.constants import SERVO_COUNT


def robot_exp():
    return {
        "robot_name": "al5d",
        "controller_type": "position_controller",
        "POS_DEFAULT": {
            "height": 4.0, "distance": 7.0, "heading": 0.0,
            "wrist_angle": -45.0, "wrist_rotation": 75.0, "gripper": 50.0,
        },
        "POS_MIN": {
            "height": 1.0, "distance": 3.0, "heading": -90.0,
            "wrist_angle": -90.0, "wrist_rotation": 45.0, "gripper": 0.0,
        },
        "POS_MAX": {
            "height": 5.0, "distance": 10.0, "heading": 90.0,
            "wrist_angle": 0.0, "wrist_rotation": 105.0, "gripper": 100.0,
        },
        "exp_pulsecontroller": "pulse", "run_pulsecontroller": "default",
        "exp_anglecontroller": "angle", "run_anglecontroller": "default",
    }


def pulse_exp():
    return {
        "device": "/dev/primary", "device_backup": "/dev/backup",
        "pulse_position_default": 1500, "CST_PULSE_MIN": 500,
        "CST_PULSE_MAX": 2500, "CST_SPEED_DEFAULT": 100,
        "TIME_DEFAULT": 50, "robot_start_speed": 200,
        "PULSE_CORRECTION": [0] * SERVO_COUNT,
    }


def angle_exp():
    return {
        "ANGLE_LIMITS": [[0, 90, 180]] * SERVO_COUNT,
        "CST_ANGLE_MIN": 0, "CST_ANGLE_MAX": 180,
    }


class FakeSerial:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.writes = []

    def write(self, command):
        self.writes.append(command)

    def read(self):
        return self.responses.pop(0) if self.responses else b""


class CapturingPulseController:
    def __init__(self, exp=None):
        self.exp = deepcopy(pulse_exp() if exp is None else exp)
        self.pulse_position_default = self.exp["pulse_position_default"]
        self.started = 0
        self.stopped = 0
        self.servo_calls = []
        self.pulse_calls = []

    def start_robot(self):
        self.started += 1

    def stop_robot(self):
        self.stopped += 1

    def control_servo_pulse(self, servo, pulse, speed):
        self.servo_calls.append((servo, pulse, speed))

    def control_pulses(self, pulses):
        self.pulse_calls.append(pulses.copy())


class CapturingAngleController:
    def __init__(self, exp, pulse_controller):
        self.exp = exp
        self.pulse_controller = pulse_controller
        self.calls = []
        self.fail = False

    def control_angles(self, angles, gripper):
        if self.fail:
            raise RuntimeError("angle command failed")
        self.calls.append((angles.copy(), gripper))
