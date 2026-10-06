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
    SOURCE_ROOT / "visual_proprioception" / "Flow_FilteredVsUnfiltered.ipynb",
    SOURCE_ROOT / "visual_proprioception" / "Flow_PtunVsRandProj_128.ipynb",
    SOURCE_ROOT / "robot_controller" / "Flow_RCCO_BehaviorCloning.ipynb",
    SOURCE_ROOT / "robot_controller" / "Flow_RCCO_Compare.ipynb",
)
# Runs Flow_VisualProprioception.ipynb once per camera, not exp/run stages
MULTIFLOW_NOTEBOOK = (
    SOURCE_ROOT / "visual_proprioception" / "MultiFlow_VisualProprioception.ipynb")
STANDARD_PARAMETERS = (
    "experiment", "run", "creation_style", "expruns_path", "results_path")
# Stage notebooks that do not call exp.done(), with the reason
NOT_DONE = {
    # redirects data_dir to an external import directory
    "demonstration/Verify_Demonstration.ipynb",
}


def load_yaml(path):
    with path.open("rt") as handle:
        return yaml.safe_load(handle) or {}


def load_notebook(path):
    return json.loads(path.read_text())


def code_source(notebook):
    return "".join(
        "".join(cell["source"]) + "\n"
        for cell in notebook["cells"] if cell["cell_type"] == "code")


def declared_notebooks():
    return {
        notebook
        for path in EXPRUN_ROOT.glob("*/*.yaml")
        for notebook in load_yaml(path).get("input-to-notebook", [])
    }


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

    def test_flow_notebook_references_exist(self):
        pattern = re.compile(r"[A-Za-z0-9_./-]+\.ipynb")
        declared = declared_notebooks()
        for path in FLOW_NOTEBOOKS:
            notebook = load_notebook(path)
            source = code_source(notebook)
            for referenced in pattern.findall(source):
                self.assertIn(referenced, NOTEBOOKS, f"{path}: {referenced}")
                self.assertIn(referenced, declared, f"{path}: {referenced}")

    def test_flow_notebooks_use_flow_helpers(self):
        for path in FLOW_NOTEBOOKS:
            notebook = load_notebook(path)
            source = code_source(notebook)
            self.assertIn("setup_flow(", source, path.name)
            self.assertIn("run_flow(", source, path.name)
            self.assertNotIn("import papermill", source, path.name)
            final = "".join(notebook["cells"][-1]["source"])
            self.assertIn("display_flow_report(", final, path.name)
            self.assertIn("raise flow_error", final, path.name)

    def test_stage_notebooks_have_standard_parameters(self):
        for notebook_path in sorted(declared_notebooks()):
            notebook = load_notebook(SOURCE_ROOT / notebook_path)
            parameter_cells = [
                cell for cell in notebook["cells"]
                if "parameters" in cell["metadata"].get("tags", [])
            ]
            self.assertEqual(len(parameter_cells), 1, notebook_path)
            source = "".join(parameter_cells[0]["source"])
            for name in STANDARD_PARAMETERS:
                self.assertRegex(
                    source, rf"(?m)^{name} =", f"{notebook_path}: {name}")

    def test_stage_notebooks_mark_exprun_done(self):
        for notebook_path in sorted(declared_notebooks() - NOT_DONE):
            notebook = load_notebook(SOURCE_ROOT / notebook_path)
            final = "".join(notebook["cells"][-1]["source"])
            self.assertRegex(final, r"^\w+\.done\(\)$", notebook_path)

    def test_notebooks_have_no_saved_execution_state(self):
        paths = [SOURCE_ROOT / notebook for notebook in declared_notebooks()]
        paths += [*FLOW_NOTEBOOKS, MULTIFLOW_NOTEBOOK]
        for path in sorted(paths):
            for cell in load_notebook(path)["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"], path.name)
                    self.assertEqual(cell["outputs"], [], path.name)


if __name__ == "__main__":
    unittest.main()
