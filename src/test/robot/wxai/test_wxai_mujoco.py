"""The PositionController on the MuJoCo driver, and the rendered cameras."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.wxai import PositionController, WXAICommand, WXAIPose
from robot.wxai.simulation.mujoco_runtime import get_runtime
from wxai_test_support import load_exp, robot_exp


class TestWXAIMujoco(unittest.TestCase):
    def setUp(self):
        self.exp = robot_exp("position_controller_mujoco_wxai_00")
        self.controller = PositionController(self.exp)
        self.controller.start_robot()

    def tearDown(self):
        if self.controller.started:
            self.controller.stop_robot()

    def test_blocking_move_tracks_the_target(self):
        target = WXAIPose(self.exp)
        self.controller.move(WXAICommand(target), moving_time=1.0, blocking=True)
        for _ in range(10):
            self.controller.update(0.1)
        reached = self.controller.get_position()
        for field in ("x", "y", "z"):
            self.assertAlmostEqual(reached[field], target[field], delta=0.01)

    def test_non_blocking_move_advances_with_update(self):
        start = self.controller.get_position()
        self.controller.move(WXAICommand(WXAIPose(self.exp)), moving_time=1.0, blocking=False)
        self.assertAlmostEqual(self.controller.get_position()["x"], start["x"], delta=1e-3)
        for _ in range(15):
            self.controller.update(0.1)
        self.assertGreater(self.controller.get_position()["x"], start["x"] + 0.1)

    def test_grasp_closes_and_release_opens(self):
        pose = self.controller.get_position()
        self.controller.move(WXAICommand(pose, "release"))
        self.controller.update(0.5)
        self.assertGreater(self.controller.get_state()["gripper_position"], 0.03)
        self.controller.move(WXAICommand(pose, "grasp"))
        self.controller.update(0.5)
        self.assertLess(self.controller.get_state()["gripper_position"], 0.005)

    def test_cameras_render_the_scene(self):
        cameras = load_exp("controllers", "mujoco_cameras")
        runtime = get_runtime(cameras["scene"])
        for camera in cameras["views"].values():
            image = runtime.render(camera, 320, 240)
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertGreater(float(np.std(image)), 1.0)

    def test_a_second_runtime_of_the_same_scene_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            PositionController(self.exp)


if __name__ == "__main__":
    unittest.main()
