"""Exp/run generators shared by the RCCO behavior cloning flows.

Each generator writes exp/runs into the active exprun path (the flow
workspace set up by flow.setup_flow) and returns the flow entries that run
them. A controller is an encoder (SP_TYPES) followed by a controller type
(CONTROLLER_TYPES). See robot_controller/DESIGN-BehaviorCloningFlow.md.
"""

import pathlib

import yaml

from exp_run_config import Config
from flow import flow_entry


# The sensor processors an encoder can come from
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

# What follows the encoder: an optional LSTM component and a head component
# (their exp/run values, without the sizes, which follow from the chain),
# and the training batches (window or chunk) of the recipe.
CONTROLLER_TYPES = {
    # bc_MLP
    "mlp": {
        "lstm": None,
        "head": {"rcco-type": "MLP", "hidden_sizes": [128, 64],
                 "output_activation": "sigmoid"},
        "training": {"batch_size": 16},
    },
    # bc_LSTM, run statefully
    "lstm_mlp": {
        "lstm": {"rcco-type": "LSTM", "architecture": "plain",
                 "num_layers": 2, "hidden_size": 128,
                 "context_mode": "stateful"},
        "head": {"rcco-type": "MLP", "hidden_sizes": [64],
                 "output_activation": "sigmoid"},
        "training": {"batch_size": 4, "chunk_length": 32},
    },
    # bc_LSTM_Residual, run statefully
    "lstm_residual_mlp": {
        "lstm": {"rcco-type": "LSTM", "architecture": "residual",
                 "num_layers": 3, "hidden_size": 128,
                 "context_mode": "stateful"},
        "head": {"rcco-type": "MLP", "hidden_sizes": [64],
                 "output_activation": "sigmoid"},
        "training": {"batch_size": 4, "chunk_length": 32},
    },
    # bc_LSTM_MDN (Rahmatizadeh et al., ICRA 2018), on sliding windows
    "lstm_residual_mdn": {
        "lstm": {"rcco-type": "LSTM", "architecture": "residual",
                 "num_layers": 3, "hidden_size": 32,
                 "context_mode": "sliding_window", "sequence_length": 10},
        "head": {"rcco-type": "MDN", "hidden_size": 32, "num_gaussians": 5,
                 "action_selection": "expected_value"},
        "training": {"batch_size": 4},
    },
}

ACTION_SIZE = 6  # the AL5D normalized position


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


def build_controller_graph(name, encoder_run, lstm_run, head_run, head_type):
    """The controller graph image_input -> encoder -> [lstm] -> head ->
    robot_output, with the given component runs (lstm_run may be None)."""
    chain = ["encoder"] + ([] if lstm_run is None else ["lstm"]) + ["head"]
    ports = {"encoder": ("image", "z"), "lstm": ("z", "h"),
             "head": ("h" if head_type == "MDN" else "z", "a")}
    connections = [{"from_component": "image_input", "from_output": "input",
                    "to_component": "encoder", "to_input": "image"}]
    for source, target in zip(chain, chain[1:]):
        connections.append({
            "from_component": source, "from_output": ports[source][1],
            "to_component": target, "to_input": ports[target][0]})
    connections.append({"from_component": "head", "from_output": "a",
                        "to_component": "robot_output", "to_input": "output"})
    components = {"image_input": {"run": "rcco_input_default"},
                  "encoder": {"run": encoder_run}}
    if lstm_run is not None:
        components["lstm"] = {"run": lstm_run}
    components["head"] = {"run": head_run}
    components["robot_output"] = {"run": "rcco_robot_action_output_6"}
    return {"name": name, "class": "GraphRobotController",
            "components": components, "connections": connections}


def generate_sp_stage(sp_type, data, epochs_sp, creation_style):
    """Write the sensor processing exp/run of sp_type, trained on the sp
    groups of data, and the encoder component built on it; return the flow
    entry of its training."""
    sp = SP_TYPES[sp_type]
    run_sp = f"_flow_sp_{sp_type}"
    values = load_run(sp["experiment"], sp["base_run"])
    values["epochs"] = epochs_sp
    values["training_data"] = data["sp_training"]
    values["validation_data"] = data["sp_validation"]
    save_run(sp["experiment"], run_sp, values)
    save_run("robot_controller", f"_flow_rcco_sp_{sp_type}", {
        "rcco-type": sp["rcco_type"],
        "sp_experiment": sp["experiment"],
        "sp_run": run_sp})
    return flow_entry(f"TrainSP {run_sp}", sp["experiment"], run_sp, 0,
                      creation_style)


def generate_controller_stages(sp_type, controller_type, data, epochs_warmup,
                               epochs_end_to_end, creation_style):
    """Write the controller, recipe, and verify exp/runs of the controller
    type on the encoder of sp_type (whose sensor processing is generated by
    generate_sp_stage), and return the flow entries [TrainRCCO, VerifyRCCO].
    data maps the demopack groups (sp_training, ..., bc_testing) to their
    demonstration entries."""
    controller = CONTROLLER_TYPES[controller_type]
    key = f"{sp_type}_{controller_type}"
    latent_size = load_run(
        SP_TYPES[sp_type]["experiment"], f"_flow_sp_{sp_type}")["latent_size"]

    # the components after the encoder, sized along the chain
    feature_size = latent_size
    lstm_run = None
    if controller["lstm"] is not None:
        lstm_run = f"_flow_rcco_lstm_{key}"
        lstm = dict(controller["lstm"], input_size=latent_size)
        save_run("robot_controller", lstm_run, lstm)
        feature_size = lstm["hidden_size"]
    head = dict(controller["head"])
    if head["rcco-type"] == "MDN":
        head.update(input_dim=feature_size, output_dim=ACTION_SIZE)
    else:
        head.update(input_size=feature_size, output_size=ACTION_SIZE)
    save_run("robot_controller", f"_flow_rcco_head_{key}", head)
    run_roco = f"_flow_roco_{key}"
    save_run("robot_controller", run_roco, build_controller_graph(
        f"{sp_type} encoder + {controller_type} controller",
        f"_flow_rcco_sp_{sp_type}", lstm_run, f"_flow_rcco_head_{key}",
        head["rcco-type"]))

    # the staged training recipe: warm up the policy on the frozen encoder,
    # then train end to end
    policy = [] if lstm_run is None else ["lstm"]
    policy.append("head")
    run_trec = f"_flow_trec_{key}"
    values = load_run("robot_controller_training", "trec_cnn_mlp_sample")
    values.update(controller["training"])
    values["name"] = f"Staged {sp_type} encoder + {controller_type} training"
    values["controller"] = {"exp": "robot_controller", "run": run_roco}
    values["initial_states"] = {
        "encoder": {"mode": "configured"},
        **{label: {"mode": "random"} for label in policy}}
    values["stages"] = [
        {"name": "warmup", "trainable_components": policy,
         "epochs": epochs_warmup, "optimizer": "Adam",
         "learning_rates": {label: 0.001 for label in policy},
         "grad_clip_norm": 1.0},
        {"name": "end_to_end", "trainable_components": ["encoder", *policy],
         "epochs": epochs_end_to_end, "optimizer": "Adam",
         "learning_rates": {"encoder": 0.00001,
                            **{label: 0.0001 for label in policy}},
         "grad_clip_norm": 1.0,
         "scheduler": {"class": "ReduceLROnPlateau", "factor": 0.5,
                       "patience": 5}},
    ]
    values["training_data"] = data["bc_training"]
    values["validation_data"] = data["bc_validation"]
    save_run("robot_controller_training", run_trec, values)

    # teacher-forced verification on the bc testing group
    run_verify = f"_flow_verify_{key}"
    save_run("robot_controller_verify", run_verify, {
        "name": f"Verify the {sp_type} encoder + {controller_type} controller",
        "trec_experiment": "robot_controller_training",
        "trec_run": run_trec,
        "testing_data": data["bc_testing"]})

    return [
        flow_entry(f"TrainRCCO {run_trec}", "robot_controller_training",
                   run_trec, 0, creation_style),
        flow_entry(f"VerifyRCCO {run_verify}", "robot_controller_verify",
                   run_verify, 0, creation_style),
    ]


def generate_compare(controllers, name, creation_style):
    """Write the comparison of the verify runs of the controllers, given as
    (sp_type, controller_type) pairs, and return its flow entry."""
    run_compare = "_flow_compare"
    save_run("robot_controller_compare", run_compare, {
        "name": name,
        "verify_experiment": "robot_controller_verify",
        "verify_runs": [f"_flow_verify_{sp_type}_{controller_type}"
                        for sp_type, controller_type in controllers],
        "labels": [f"{sp_type} + {controller_type}"
                   for sp_type, controller_type in controllers]})
    return flow_entry(f"CompareRCCO {run_compare}", "robot_controller_compare",
                      run_compare, 0, creation_style)
