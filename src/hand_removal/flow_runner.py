"""Build an isolated, repeatable offline hand-removal experiment."""
from pathlib import Path

import yaml

from exp_run_config import Config
from flow import setup_flow, flow_entry, run_flow, display_flow_report


def run_hand_removal(flow_name, manifests=(), *, learn=True, multiview=True, flows_path=None,
                     epochs=3, creation_style="version", methods=None):
    Config().set_exprun_path(Config().exprun_path_internal)
    expruns, results, books = setup_flow(flow_name, ["hand_removal", "hand_removal_comparison"],
                                         flows_path=flows_path)
    methods = list(methods if methods is not None else ("plate","reference","past","bidirectional"))
    if learn:
        methods.append("learned")
    queue = [("prepare","Prepare_HandRemoval_Benchmark"), ("masks","Verify_HandMasks")]
    if learn:
        queue.append(("train","Train_HandRemoval"))
    queue.append(("single","Verify_SingleCamera_HandRemoval"))
    if multiview:
        queue.append(("multi","Verify_MultiCamera_HandRemoval"))
    queue.append(("compare","Compare_HandRemoval"))
    entries = []
    for stage, notebook in queue:
        family = "hand_removal_comparison" if stage=="compare" else "hand_removal"
        run = flow_name if stage=="compare" else stage
        config = dict(results_root=str(results), manifests=[str(Path(p).resolve()) for p in manifests],
                      methods=methods, epochs=epochs, compare_stages=["single","multi"] if multiview else ["single"])
        config["input-to-notebook"] = [f"hand_removal/{notebook}.ipynb"]
        (expruns/family/f"{run}.yaml").write_text(yaml.safe_dump(config))
        entries.append(flow_entry(stage,family,run,0,creation_style))
    execution = run_flow(entries,expruns,results,books)
    final = results/"hand_removal_comparison"/flow_name
    display_flow_report(entries,results,books,final,execution=execution)
    return final
