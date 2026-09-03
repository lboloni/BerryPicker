from types import SimpleNamespace

import numpy as np


def robot_exp():
    return {
        "robot_name": "widowx",
        "controller_type": "position_controller",
        "robot_model": "wx250s",
        "startup_pose": "hold",
        "shutdown_pose": "hold",
        "moving_time": 0.1,
        "POSE_DEFAULT": {
            "x": 0.3, "y": 0.0, "z": 0.2,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        },
        "POSE_MIN": {
            "x": -1.0, "y": -1.0, "z": -1.0,
            "roll": -3.2, "pitch": -3.2, "yaw": -3.2,
        },
        "POSE_MAX": {
            "x": 1.0, "y": 1.0, "z": 1.0,
            "roll": 3.2, "pitch": 3.2, "yaw": 3.2,
        },
    }


class FakeCore:
    def __init__(self, node):
        self.node = node

    def get_node(self):
        return self.node


class FakeArm:
    def __init__(self):
        self.pose = np.eye(4)
        self.pose[:3, 3] = [0.3, 0.0, 0.2]
        self.joints = [0.0] * 6
        self.capture_count = 0
        self.pose_calls = []
        self.joint_calls = []
        self.cartesian_calls = []
        self.reachable = True
        self.apply_pose_commands = True

    def capture_joint_positions(self):
        self.capture_count += 1

    def get_ee_pose(self):
        return self.pose.copy()

    def get_ee_pose_command(self):
        return self.pose.copy()

    def get_joint_positions(self):
        return list(self.joints)

    def set_ee_pose_components(self, **kwargs):
        self.pose_calls.append(dict(kwargs))
        if kwargs.get("execute", True) and self.reachable and self.apply_pose_commands:
            self.pose[:3, 3] = [kwargs[axis] for axis in ("x", "y", "z")]
        return list(self.joints), self.reachable

    def set_joint_positions(self, joints, **kwargs):
        self.joint_calls.append((list(joints), dict(kwargs)))
        if self.reachable:
            self.joints = list(joints)
        return self.reachable

    def set_ee_cartesian_trajectory(self, **kwargs):
        self.cartesian_calls.append(dict(kwargs))
        return self.reachable

    def go_to_home_pose(self, moving_time=None):
        self.pose[:3, 3] = [0.3, 0.0, 0.2]
        return self.reachable

    def go_to_sleep_pose(self, moving_time=None):
        self.pose[:3, 3] = [0.1, 0.0, 0.1]
        return self.reachable


class FakeGripper:
    def __init__(self):
        self.pressures = []
        self.actions = []
        self.efforts = []

    def set_pressure(self, pressure):
        self.pressures.append(pressure)

    def grasp(self, delay):
        self.actions.append(("grasp", delay))

    def release(self, delay):
        self.actions.append(("release", delay))

    def gripper_controller(self, effort, duration):
        self.efforts.append((effort, duration))

    def get_finger_position(self):
        return 0.012


class FakeBot:
    def __init__(self, node=None):
        self.core = FakeCore(object() if node is None else node)
        self.arm = FakeArm()
        self.gripper = FakeGripper()


class FakeAPI:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.created = []
        self.angle_manipulation = SimpleNamespace(
            rotation_matrix_to_euler_angles=lambda matrix: np.zeros(3)
        )

    def Manipulator(self, **kwargs):
        self.created.append(dict(kwargs))
        return FakeBot(kwargs.get("node"))

    def robot_startup(self, node):
        self.started.append(node)

    def robot_shutdown(self, node):
        self.stopped.append(node)
