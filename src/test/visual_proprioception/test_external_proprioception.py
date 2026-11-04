import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from visual_proprioception import external_proprioception, visproprio_helper

JOINTS = ["panda_joint1", "panda_joint2"]


def robot_exp():
    return {"robot_name": "franka", "action_key": "joint-position", "enforce_limits": True,
            "JOINTS": JOINTS,
            "POS_DEFAULT": {"panda_joint1": 0.0, "panda_joint2": 0.0},
            "POS_MIN": {"panda_joint1": -1.0, "panda_joint2": 0.0},
            "POS_MAX": {"panda_joint1": 1.0, "panda_joint2": 2.0}}


class FakeExp(dict):
    def __init__(self, directory, **values):
        super().__init__(name="external", experiment_name="external_proprioception",
                         position_filter="none", sample_interval=0.1, **values)
        self.directory = directory

    def data_dir(self):
        return self.directory


class FakeDemonstration:
    """A demonstration of two frames, taken from frame offset of a DREAM sequence"""
    OFFSETS = {"demo_a": 10, "demo_b": 3}

    def __init__(self, _exp_demo, demo_name):
        self.metadata = {"source_sequence": "DREAM panda-3cam_realsense",
                         "source_frame_offset": self.OFFSETS[demo_name], "maxsteps": 2}
        self.actions = [{"joint-position": {"panda_joint1": 0.0, "panda_joint2": 1.0}},
                        {"joint-position": {"panda_joint1": 1.0, "panda_joint2": 2.0}}]

    def get_action(self, i, key, exp):
        from robot.position_factory import create_position
        return create_position(exp, self.actions[i][key])


class TestExternalProprioception(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.sequence = root / "panda-3cam_realsense"
        self.sequence.mkdir()
        for i in range(20):
            (self.sequence / f"{i:06d}.rgb.jpg").write_text(f"image {i}")
            (self.sequence / f"{i:06d}.json").write_text(f"json {i}")
        (self.sequence / "_camera_settings.json").write_text("camera")
        self.exp = FakeExp(root / "run", validation_data=[["pack", "demo_a", "dev000"], ["pack", "demo_b", "dev000"]])
        self.settings = {"external_proprioception": {"sequences": {"DREAM panda-3cam_realsense": str(self.sequence)}}}
        settings = self.settings
        self.patches = [
            patch.object(external_proprioception, "Demonstration", FakeDemonstration),
            patch.object(external_proprioception.Config, "get_experiment", lambda self, *args, **kw: None),
            patch.object(external_proprioception.Config, "__getitem__", lambda self, key: settings[key]),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_export_links_the_original_frames_in_the_order_of_the_demonstrations(self):
        target = external_proprioception.export_dream(self.exp, "validation_data")
        self.assertEqual(target.name, "panda-3cam_realsense")
        originals = [pathlib.Path(os.path.realpath(target / f"{n:06d}.rgb.jpg")).name for n in range(4)]
        self.assertEqual(originals, ["000010.rgb.jpg", "000011.rgb.jpg", "000003.rgb.jpg", "000004.rgb.jpg"])
        self.assertEqual((target / "000002.json").read_text(), "json 3")
        self.assertEqual((target / "_camera_settings.json").read_text(), "camera")
        # exporting again replaces the links
        external_proprioception.export_dream(self.exp, "validation_data")

    def test_predictions_are_mapped_by_joint_name_onto_the_robot_run(self):
        self.exp.directory.mkdir()
        # a method predicting an extra joint, in another order, partly outside the limits
        np.savez(external_proprioception.angles_file(self.exp, "validation_data"),
                 angles=np.array([[5.0, 1.0, 0.0], [5.0, 2.0, 1.0], [5.0, 3.0, -1.0], [5.0, 0.0, 0.0]]),
                 joint_names=np.array(["panda_joint7", "panda_joint2", "panda_joint1"]))
        result = external_proprioception.load_external_predictions(self.exp, robot_exp(), "validation_data")
        np.testing.assert_allclose(result["predictions"], [[0.5, 0.5], [1.0, 1.0], [0.0, 1.5], [0.5, 0.0]])
        np.testing.assert_allclose(result["targets"], [[0.5, 0.5], [1.0, 1.0]] * 2)
        self.assertEqual(result["lengths"], [2, 2])

    def test_predict_positions_reads_an_external_run(self):
        self.exp.directory.mkdir()
        np.savez(external_proprioception.angles_file(self.exp, "validation_data"),
                 angles=np.zeros((4, 2)), joint_names=np.array(JOINTS))
        result = visproprio_helper.predict_positions(self.exp, robot_exp(), "validation_data")
        np.testing.assert_allclose(result["filtered"], result["predictions"])
        self.assertEqual(result["predictions"].shape, (4, 2))

    def test_run_external_calls_the_driver_with_the_python_of_the_method(self):
        self.settings["external_proprioception"]["robopepp"] = {
            "code_dir": "/code", "python": "/env/bin/python", "checkpoint_dir": "/ckpt",
            "driver_args": {"jepa-path": "/jepa.pth"}}
        exp = FakeExp(self.exp.directory, method="robopepp", checkpoint="model.pt",
                      driver_args={"batch-size": 8}, validation_data=self.exp["validation_data"])
        with patch.object(external_proprioception.subprocess, "run") as run:
            external_proprioception.run_external(exp, "validation_data")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/env/bin/python")
        self.assertTrue(command[1].endswith("external_drivers/robopepp_predict.py"))
        self.assertEqual(command[command.index("--checkpoint") + 1], "/ckpt/model.pt")
        self.assertEqual(command[command.index("--jepa-path") + 1], "/jepa.pth")
        self.assertEqual(command[command.index("--batch-size") + 1], "8")
        self.assertTrue(command[command.index("--frames") + 1].endswith("export_validation_data/panda-3cam_realsense"))


if __name__ == "__main__":
    unittest.main()
