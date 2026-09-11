#!/usr/bin/env python3
"""Mirror an Interbotix SDK simulator into Gazebo trajectory controllers."""

from __future__ import annotations

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
GRIPPER_JOINTS = ("left_finger", "right_finger")


def _positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


class InterbotixGazeboBridge(Node):
    """Translate simulated Interbotix joint states into Gazebo trajectories."""

    def __init__(self, sdk_robot_name, gazebo_robot_name, moving_time):
        super().__init__("berrypicker_widowx_gazebo_bridge")
        self.moving_time = moving_time
        self.arm_publisher = self.create_publisher(
            JointTrajectory,
            f"/{gazebo_robot_name}/arm_controller/joint_trajectory",
            10,
        )
        self.gripper_publisher = self.create_publisher(
            JointTrajectory,
            f"/{gazebo_robot_name}/gripper_controller/joint_trajectory",
            10,
        )
        self.subscription = self.create_subscription(
            JointState,
            f"/{sdk_robot_name}/joint_states",
            self._joint_state_callback,
            qos_profile_sensor_data,
        )

    def _trajectory(self, names, positions):
        trajectory = JointTrajectory()
        trajectory.joint_names = list(names)
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.time_from_start.sec = int(self.moving_time)
        point.time_from_start.nanosec = int(
            (self.moving_time - point.time_from_start.sec) * 1e9
        )
        trajectory.points = [point]
        return trajectory

    def _joint_state_callback(self, message):
        if len(message.name) != len(message.position):
            raise ValueError("Interbotix joint state names and positions differ in length")
        positions = dict(zip(message.name, message.position))
        missing = [
            name for name in ARM_JOINTS + GRIPPER_JOINTS if name not in positions
        ]
        if missing:
            raise RuntimeError(
                f"Interbotix joint state is missing Gazebo joints: {missing}"
            )
        self.arm_publisher.publish(self._trajectory(
            ARM_JOINTS, (positions[name] for name in ARM_JOINTS)
        ))
        self.gripper_publisher.publish(self._trajectory(
            GRIPPER_JOINTS, (positions[name] for name in GRIPPER_JOINTS)
        ))


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-robot-name", default="wx250s_sdk")
    parser.add_argument("--gazebo-robot-name", default="wx250s")
    parser.add_argument("--moving-time", type=_positive_float, default=0.1)
    return parser.parse_args()


def main():
    args = _arguments()
    rclpy.init()
    bridge = InterbotixGazeboBridge(
        args.sdk_robot_name, args.gazebo_robot_name, args.moving_time
    )
    try:
        rclpy.spin(bridge)
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
