import pathlib
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).parents[3]))

from robot.widowx.simulation.interbotix_gazebo_bridge import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    InterbotixGazeboBridge,
)
from sensor_msgs.msg import JointState


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class TestInterbotixGazeboBridge(unittest.TestCase):
    def test_joint_state_becomes_arm_and_gripper_trajectories(self):
        bridge = object.__new__(InterbotixGazeboBridge)
        bridge.moving_time = 0.1
        bridge.arm_publisher = FakePublisher()
        bridge.gripper_publisher = FakePublisher()
        message = JointState()
        message.name = list(ARM_JOINTS + ("gripper",) + GRIPPER_JOINTS)
        message.position = [float(index) for index in range(len(message.name))]

        bridge._joint_state_callback(message)

        arm = bridge.arm_publisher.messages[0]
        gripper = bridge.gripper_publisher.messages[0]
        self.assertEqual(arm.joint_names, list(ARM_JOINTS))
        self.assertEqual(list(arm.points[0].positions), list(range(6)))
        self.assertEqual(gripper.joint_names, list(GRIPPER_JOINTS))
        self.assertEqual(list(gripper.points[0].positions), [7.0, 8.0])
        self.assertEqual(arm.points[0].time_from_start.nanosec, 100_000_000)

    def test_missing_joint_raises(self):
        bridge = object.__new__(InterbotixGazeboBridge)
        bridge.moving_time = 0.1
        bridge.arm_publisher = FakePublisher()
        bridge.gripper_publisher = FakePublisher()
        message = JointState()
        message.name = ["waist"]
        message.position = [0.0]

        with self.assertRaisesRegex(RuntimeError, "missing Gazebo joints"):
            bridge._joint_state_callback(message)


if __name__ == "__main__":
    unittest.main()
