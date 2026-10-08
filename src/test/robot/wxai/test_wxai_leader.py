import pathlib
import sys
import unittest

import trossen_arm

sys.path.extend([
    str(pathlib.Path(__file__).parents[3]),
    str(pathlib.Path(__file__).parent),
])

from robot.wxai.leader_controller import LeaderController
from wxai_test_support import leader_exp


class TestWXAILeader(unittest.TestCase):
    def setUp(self):
        self.exp = leader_exp()
        self.leader = LeaderController(self.exp)

    def test_start_frees_the_leader(self):
        self.leader.start()
        driver = self.leader.driver
        self.assertEqual(driver.modes["arm"], trossen_arm.Mode.external_effort)
        self.assertEqual(driver.calls[-1], ("set_all_external_efforts", [0.0] * 7))
        self.assertEqual(self.leader.get_joints(), [0.0] * 7)

    def test_feedback_only_with_a_positive_gain(self):
        self.leader.start()
        calls = len(self.leader.driver.calls)
        self.leader.feedback([1.0] * 7)
        self.assertEqual(len(self.leader.driver.calls), calls)
        self.leader.exp = dict(self.exp, force_feedback_gain=0.1)
        self.leader.feedback([1.0] * 7)
        self.assertEqual(self.leader.driver.calls[-1], ("set_all_external_efforts", [-0.1] * 7))

    def test_stop_idles_and_cleans_up(self):
        self.leader.start()
        self.leader.stop()
        self.assertEqual(self.leader.driver.modes["arm"], trossen_arm.Mode.idle)
        self.assertEqual(self.leader.driver.calls[-1], ("cleanup",))
        with self.assertRaisesRegex(RuntimeError, "not started"):
            self.leader.get_joints()


if __name__ == "__main__":
    unittest.main()
