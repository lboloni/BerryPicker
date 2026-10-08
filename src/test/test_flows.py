"""Tests the exp/run creation styles and the flow report with the real
BerryPicker Config. The generic flow helpers are tested in ExpRunFlow.

Run from the repository root with:
    PYTHONPATH=src python -m unittest src.test.test_flows -v
"""

import pathlib
import tempfile
import unittest

from exp_run_config import Config
from flow import get_flow_report


class TestCreationStyles(unittest.TestCase):
    """Uses the real Config with a temporary results directory."""

    experiment = "robot_al5d"
    run_name = "position_controller_00"

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.results = pathlib.Path(self.temporary_directory.name)
        self.old_results = Config().get_results_path()
        Config().set_results_path(self.results)
        self.data_dir = self.results / self.experiment / self.run_name

    def tearDown(self):
        Config().values["experiment_data"] = self.old_results
        self.temporary_directory.cleanup()

    def test_create_data_dir_false_has_no_side_effect(self):
        exp = Config().get_experiment(
            self.experiment, self.run_name, create_data_dir=False)
        self.assertEqual(pathlib.Path(exp["data_dir"]), self.data_dir)
        self.assertFalse(self.data_dir.exists())

    def test_version_creates_missing_directory(self):
        exp = Config().get_experiment(
            self.experiment, self.run_name, creation_style="version")
        self.assertTrue((self.data_dir / "exprun.yaml").is_file())
        exp.done()
        Config().get_experiment(
            self.experiment, self.run_name, creation_style="version")
        backups = [path for path in self.data_dir.parent.iterdir()
                   if path.name.startswith(self.run_name + "_")]
        self.assertEqual(len(backups), 1)
        report = get_flow_report(
            [{"name": "x", "notebook": "sample/Train.ipynb",
              "experiment": self.experiment, "run": self.run_name}],
            self.results, self.results)
        self.assertFalse(report["all-results-present"])

    def test_unknown_creation_style_raises(self):
        with self.assertRaisesRegex(Exception, "Unknown creation_style"):
            Config().get_experiment(
                self.experiment, self.run_name, creation_style="exists-ok")


if __name__ == "__main__":
    unittest.main()
