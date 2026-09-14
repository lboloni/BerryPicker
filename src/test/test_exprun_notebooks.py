"""Consistency tests for exprun notebook entry points."""

import json
import pathlib
import re
import unittest

import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
EXPRUN_ROOT = REPOSITORY_ROOT / "data" / "expruns"
NOTEBOOKS = {
    path.relative_to(SOURCE_ROOT).as_posix() for path in SOURCE_ROOT.rglob("*.ipynb")
}
FLOW_NOTEBOOKS = (
    SOURCE_ROOT / "behavior_cloning" / "Flow_BehaviorCloning.ipynb",
    SOURCE_ROOT / "visual_proprioception" / "Flow_VisualProprioception.ipynb",
    SOURCE_ROOT / "visual_proprioception" / "Flow_VisualProprioception_multi.ipynb",
)


def load_yaml(path):
    with path.open("rt") as handle:
        return yaml.safe_load(handle) or {}


class TestExprunNotebooks(unittest.TestCase):
    def assert_notebooks(self, exp, source):
        notebooks = exp["input-to-notebook"]
        self.assertIsInstance(notebooks, list, source)
        for notebook in notebooks:
            self.assertIn(notebook, NOTEBOOKS, f"{source}: {notebook}")

    def test_every_exprun_has_existing_notebook_entries(self):
        for experiment_dir in sorted(path for path in EXPRUN_ROOT.iterdir() if path.is_dir()):
            defaults_path = experiment_dir / f"_defaults_{experiment_dir.name}.yaml"
            defaults = load_yaml(defaults_path)
            self.assert_notebooks(defaults, defaults_path)
            for run_path in sorted(experiment_dir.glob("*.yaml")):
                if run_path == defaults_path:
                    continue
                self.assert_notebooks(defaults | load_yaml(run_path), run_path)

    def test_automation_notebooks_exist(self):
        automate_dir = EXPRUN_ROOT / "automate"
        for path in sorted(automate_dir.glob("*.yaml")):
            if path.name.startswith("_defaults_"):
                continue
            for entry in load_yaml(path)["exps_to_run"]:
                notebook = entry["notebook"]
                self.assertIn(notebook, NOTEBOOKS, f"{path}: {notebook}")

    def test_flow_notebook_references_exist(self):
        pattern = re.compile(r"[A-Za-z0-9_./-]+\.ipynb")
        declared_notebooks = {
            notebook
            for path in EXPRUN_ROOT.glob("*/*.yaml")
            for notebook in load_yaml(path).get("input-to-notebook", [])
        }
        for path in FLOW_NOTEBOOKS:
            notebook = json.loads(path.read_text())
            source = "".join(
                line for cell in notebook["cells"] for line in cell.get("source", [])
            )
            for referenced in pattern.findall(source):
                self.assertIn(referenced, NOTEBOOKS, f"{path}: {referenced}")
                self.assertIn(referenced, declared_notebooks, f"{path}: {referenced}")


if __name__ == "__main__":
    unittest.main()
