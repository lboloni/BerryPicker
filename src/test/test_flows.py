"""Tests for the flow helpers in src/flow.py and the exp/run creation styles.

Run from the repository root with:
    PYTHONPATH=src python -m unittest src.test.test_flows -v
"""

import pathlib
import tempfile
import unittest

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from flow import flow_entry, get_flow_report, run_flow, run_notebook, setup_flow


class FlowConfig:
    """Stand-in for Config, recording the paths set by setup_flow."""

    def __init__(self, exprun_path, flows_path, results_path, experiments=None):
        self.exprun_path = exprun_path
        self.values = {
            "flows_path": flows_path,
            "experiment_data": results_path,
        }
        self.experiments = experiments
        self.create_data_dir_values = []

    def __getitem__(self, key):
        return self.values[key]

    def get_exprun_path(self):
        return self.exprun_path

    def set_exprun_path(self, path):
        self.exprun_path = path

    def set_results_path(self, path):
        self.values["experiment_data"] = path

    def get_experiment(self, experiment, run, create_data_dir=True):
        self.create_data_dir_values.append(create_data_dir)
        return self.experiments[(experiment, run)]


class RecordingProgress:
    def __init__(self):
        self.n = 0
        self.options = None
        self.labels = []
        self.closed = False

    def set_postfix_str(self, label):
        self.labels.append(label)

    def update(self, count):
        self.n += count

    def close(self):
        self.closed = True


class TestFlowHelpers(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.flows = self.root / "flows"
        family = self.source / "sample"
        family.mkdir(parents=True)
        (family / "_defaults_sample.yaml").write_text("input-to-notebook: []\n")
        (family / "run.yaml").write_text("name: sample\n")
        self.config = FlowConfig(self.source, self.flows, self.root / "old")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_setup_flow_copies_from_active_exprun_path(self):
        expruns, results, notebooks = setup_flow(
            "sample-flow", ["sample"], config=self.config)
        self.assertEqual(expruns, self.flows / "sample-flow" / "expruns")
        self.assertTrue((expruns / "sample" / "run.yaml").is_file())
        self.assertEqual(self.config.exprun_path, expruns)
        self.assertEqual(self.config.values["experiment_data"], results)
        self.assertTrue(results.is_dir())
        self.assertTrue(notebooks.is_dir())

    def test_flow_entry_takes_notebook_from_exprun(self):
        self.config.experiments = {
            ("sample", "run"): {
                "input-to-notebook": ["dir/Train.ipynb", "dir/Verify.ipynb"]},
        }
        entry = flow_entry(
            "Verify run", "sample", "run", 1, "exist-ok", config=self.config)
        self.assertEqual(entry, {
            "name": "Verify run",
            "notebook": "dir/Verify.ipynb",
            "experiment": "sample",
            "run": "run",
            "creation_style": "exist-ok",
        })
        self.assertEqual(self.config.create_data_dir_values, [False])

    def test_run_notebook_passes_standard_parameters(self):
        calls = []

        def executor(source, output, **kwargs):
            calls.append((source, output, kwargs))

        notebook_root = self.root / "src"
        notebook = notebook_root / "sample" / "Train.ipynb"
        notebook.parent.mkdir(parents=True)
        notebook.write_text("{}")
        output_root = self.root / "executed"
        output_root.mkdir()
        entry = {
            "notebook": "sample/Train.ipynb",
            "experiment": "sample",
            "run": "run",
            "creation_style": "discard-old",
        }

        output = run_notebook(
            entry, self.source, self.root / "results", output_root,
            notebook_root=notebook_root, executor=executor)

        self.assertEqual(output, output_root / "Train_sample_run.ipynb")
        self.assertEqual(len(calls), 1)
        source, called_output, kwargs = calls[0]
        self.assertEqual(source, notebook)
        self.assertEqual(called_output, output)
        self.assertEqual(kwargs["cwd"], notebook.parent)
        self.assertEqual(kwargs["kernel_name"], "berrypicker")
        self.assertEqual(kwargs["parameters"], {
            "experiment": "sample",
            "run": "run",
            "creation_style": "discard-old",
            "expruns_path": self.source.as_posix(),
            "results_path": (self.root / "results").as_posix(),
        })

    def test_run_flow_runs_entries_in_order(self):
        entries = [{"name": "first"}, {"name": "second"}]
        calls = []
        progress = RecordingProgress()

        def run(entry, expruns, results, notebooks):
            calls.append(entry)

        def progress_factory(**options):
            progress.options = options
            return progress

        run_flow(
            entries, self.source, self.root / "results", self.root,
            notebook_runner=run, progress_factory=progress_factory)

        self.assertEqual(calls, entries)
        self.assertEqual(progress.options, {
            "total": 2, "desc": "Overall flow", "unit": "notebook"})
        self.assertEqual(progress.n, 2)
        self.assertEqual(progress.labels[-1], "0 notebooks left")
        self.assertTrue(progress.closed)

    def test_run_flow_stops_and_propagates_failure(self):
        calls = []
        progress = RecordingProgress()

        def fail(entry, *args):
            calls.append(entry)
            raise RuntimeError("failed")

        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_flow(
                [{"name": "failing"}, {"name": "never"}], self.source,
                self.root / "results", self.root,
                notebook_runner=fail,
                progress_factory=lambda **kwargs: progress)
        self.assertEqual(calls, [{"name": "failing"}])
        self.assertEqual(progress.n, 0)
        self.assertTrue(progress.closed)

    def test_flow_report_requires_time_done(self):
        results = self.root / "results"
        complete = results / "sample" / "complete"
        complete.mkdir(parents=True)
        (complete / "exprun.yaml").write_text(
            "time_done: '2026-09-27 12:00:00.000000'\n")
        started = results / "sample" / "started"
        started.mkdir(parents=True)
        (started / "exprun.yaml").write_text(
            "time_started: '2026-09-27 12:00:00.000000'\n")
        entries = [
            {"name": "complete", "experiment": "sample", "run": "complete"},
            {"name": "started", "experiment": "sample", "run": "started"},
        ]

        report = get_flow_report(entries, results)
        self.assertFalse(report["successful"])
        self.assertFalse(report["all-results-present"])
        self.assertEqual(
            [stage["name"] for stage in report["completed"]], ["complete"])
        self.assertEqual(
            [stage["name"] for stage in report["missing"]], ["started"])

        (started / "exprun.yaml").write_text(
            "time_done: '2026-09-27 12:01:00.000000'\n")
        report = get_flow_report(entries, results)
        self.assertTrue(report["successful"])
        report = get_flow_report(entries, results, RuntimeError("failed"))
        self.assertFalse(report["successful"])
        self.assertTrue(report["all-results-present"])


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
            [{"name": "x", "experiment": self.experiment, "run": self.run_name}],
            self.results)
        self.assertFalse(report["all-results-present"])

    def test_unknown_creation_style_raises(self):
        with self.assertRaisesRegex(Exception, "Unknown creation_style"):
            Config().get_experiment(
                self.experiment, self.run_name, creation_style="exists-ok")


if __name__ == "__main__":
    unittest.main()
