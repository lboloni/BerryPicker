"""Exp/run generators shared by the RCCO behavior cloning flows.

Each generator writes exp/runs into the active exprun path (the flow
workspace set up by flow.setup_flow) and returns the flow entries that run
them. See robot_controller/DESIGN-BehaviorCloningFlow.md.
"""

import pathlib

import yaml

from exp_run_config import Config
from flow import flow_entry


# The sensor processors an encoder--MLP controller can be built on
SP_TYPES = {
    "resnet50": {"experiment": "sensorprocessing_propriotuned_cnn",
                 "base_run": "resnet50_256", "rcco_type": "SP_CNN"},
    "vgg19": {"experiment": "sensorprocessing_propriotuned_cnn",
              "base_run": "vgg19_256", "rcco_type": "SP_CNN"},
    "vae": {"experiment": "sensorprocessing_conv_vae_neo",
            "base_run": "sp_vae_neo_256_256px", "rcco_type": "SP_VAE"},
    "vae_gan": {"experiment": "sensorprocessing_vae_gan",
                "base_run": "sp_vae_gan_256_256px", "rcco_type": "SP_VAE"},
}


def flow_families(sp_types):
    """The experiment families setup_flow must copy for these sp_types."""
    families = ["demonstration", "robot_al5d", "robot_controller",
                "robot_controller_training", "robot_controller_verify",
                "robot_controller_compare"]
    for sp_type in sp_types:
        if SP_TYPES[sp_type]["experiment"] not in families:
            families.append(SP_TYPES[sp_type]["experiment"])
    return families


def load_run(experiment, run):
    """Load the values of a run file (without its defaults) as a base."""
    path = pathlib.Path(Config().get_exprun_path(), experiment, run + ".yaml")
    with path.open() as f:
        return yaml.safe_load(f)


def save_run(experiment, run, values):
    path = pathlib.Path(Config().get_exprun_path(), experiment, run + ".yaml")
    with path.open("w") as f:
        yaml.dump(values, f)


def generate_controller_stages(sp_type, data, epochs_sp, epochs_warmup,
                               epochs_end_to_end, creation_style):
    """Write the SP, controller, recipe, and verify exp/runs for one
    encoder--MLP controller, and return the flow entries
    [TrainSP, TrainRCCO, VerifyRCCO]. data maps the demopack groups
    (sp_training, ..., bc_testing) to their demonstration entries."""
    sp = SP_TYPES[sp_type]
    run_sp = f"_flow_sp_{sp_type}"
    run_rcco_sp = f"_flow_rcco_sp_{sp_type}"
    run_roco = f"_flow_roco_{sp_type}"
    run_trec = f"_flow_trec_{sp_type}"
    run_verify = f"_flow_verify_{sp_type}"

    # the sensor processing, trained on the sp groups
    values = load_run(sp["experiment"], sp["base_run"])
    values["epochs"] = epochs_sp
    values["training_data"] = data["sp_training"]
    values["validation_data"] = data["sp_validation"]
    save_run(sp["experiment"], run_sp, values)

    # the controller graph, with the encoder from the sensor processing
    save_run("robot_controller", run_rcco_sp, {
        "rcco-type": sp["rcco_type"],
        "sp_experiment": sp["experiment"],
        "sp_run": run_sp})
    values = load_run("robot_controller", "roco_cnn_mlp_sample")
    values["name"] = f"{sp_type} encoder + deterministic MLP controller"
    values["components"]["cnn_encoder"]["run"] = run_rcco_sp
    save_run("robot_controller", run_roco, values)

    # the staged training recipe, trained on the bc groups
    values = load_run("robot_controller_training", "trec_cnn_mlp_sample")
    values["name"] = f"Staged {sp_type} encoder + MLP training"
    values["controller"]["run"] = run_roco
    epochs = {"mlp_warmup": epochs_warmup, "end_to_end": epochs_end_to_end}
    for stage in values["stages"]:
        stage["epochs"] = epochs[stage["name"]]
    values["training_data"] = data["bc_training"]
    values["validation_data"] = data["bc_validation"]
    save_run("robot_controller_training", run_trec, values)

    # teacher-forced verification on the bc testing group
    save_run("robot_controller_verify", run_verify, {
        "name": f"Verify the {sp_type} encoder + MLP controller",
        "trec_experiment": "robot_controller_training",
        "trec_run": run_trec,
        "testing_data": data["bc_testing"]})

    return [
        flow_entry(f"TrainSP {run_sp}", sp["experiment"], run_sp, 0,
                   creation_style),
        flow_entry(f"TrainRCCO {run_trec}", "robot_controller_training",
                   run_trec, 0, creation_style),
        flow_entry(f"VerifyRCCO {run_verify}", "robot_controller_verify",
                   run_verify, 0, creation_style),
    ]


def generate_compare(sp_types, name, creation_style):
    """Write the comparison of the verify runs of sp_types, and return its
    flow entry."""
    run_compare = "_flow_compare"
    save_run("robot_controller_compare", run_compare, {
        "name": name,
        "verify_experiment": "robot_controller_verify",
        "verify_runs": [f"_flow_verify_{sp_type}" for sp_type in sp_types],
        "labels": list(sp_types)})
    return flow_entry(f"CompareRCCO {run_compare}", "robot_controller_compare",
                      run_compare, 0, creation_style)
