"""Graphviz view of a controller training recipe and its current stage."""

import json
from pathlib import Path

from robot_controller.graph_robot_controller import load_controller_spec
from robot_controller.visualize_rcco import RCCOVisualizer


class TrainingRecipeVisualizer:
    """Visualize stage progress without loading any model checkpoints."""

    def __init__(self, exp_trec, *, experiment_loader=None, status=None):
        self.exp = exp_trec
        if experiment_loader is None:
            from exp_run_config import Config
            experiment_loader = Config().get_experiment
        reference = exp_trec["controller"]
        controller_exp = experiment_loader(
            reference.get("exp", "robot_controller"), reference["run"]
        )
        self.spec = load_controller_spec(controller_exp, experiment_loader)
        self.status = status or self._load_status()

    def _load_status(self):
        path = Path(self.exp["data_dir"]) / "recipe_status.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "state": "not_started", "current_stage_index": None,
            "stages": [
                {"name": stage["name"], "state": "pending", "epoch": 0,
                 "epochs": stage["epochs"],
                 "monitor": stage.get("monitor", "validation_loss"),
                 "best_metric": None}
                for stage in self.exp["stages"]
            ],
        }

    def build(self):
        current = self.status.get("current_stage_index")
        trainable = set()
        if current is not None:
            trainable = set(self.exp["stages"][current]["trainable_components"])
        colors = {}
        annotations = {}
        styles = {}
        initialized = self.status.get("state") not in {"not_started", "initializing"}
        for label, item in self.spec["components"].items():
            if item["type"] in {"Input", "Output"}:
                colors[label] = "lightgray"
                continue
            if label in trainable:
                colors[label] = "palegreen"
                annotations[label] = "trainable in current stage"
                styles[label] = "bold"
            else:
                colors[label] = "lightskyblue"
                annotations[label] = "frozen"
            if not initialized:
                styles[label] = "dashed"
                annotations[label] = "not initialized"
        dot = RCCOVisualizer(
            self.spec, component_colors=colors,
            component_annotations=annotations, component_styles=styles,
        ).build()
        dot.attr(label=f"Training state: {self.status.get('state', 'unknown')}",
                 labelloc="t", fontsize="16")
        with dot.subgraph(name="cluster_training_stages") as stages:
            stages.attr(label="Training stages", color="gray60", rankdir="LR")
            previous = None
            for index, stage in enumerate(self.status["stages"]):
                state = stage["state"]
                color = {
                    "completed": "palegreen", "running": "khaki",
                    "pending": "white",
                }.get(state, "lightcoral")
                metric = stage.get("best_metric")
                monitor = stage.get("monitor", "validation metric")
                metric_text = (
                    "" if metric is None
                    else f"\\nbest {monitor}={metric:.5g}"
                )
                label = (
                    f"{index + 1}. {stage['name']}\\n{state}\\n"
                    f"epoch {stage.get('epoch', 0)}/{stage['epochs']}"
                    f"{metric_text}"
                )
                node = f"training_stage_{index}"
                stages.node(node, label, shape="box", style="filled", fillcolor=color)
                if previous is not None:
                    stages.edge(previous, node)
                previous = node
        return dot


def build_training_recipe_graph(exp_trec, **kwargs):
    return TrainingRecipeVisualizer(exp_trec, **kwargs).build()
