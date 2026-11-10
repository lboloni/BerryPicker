"""
transfer_flow.py

The generators of the transfer flows (Flow_Transfer.ipynb and
Flow_Transfer_Smoke.ipynb; see DESIGN-Transfer.md). TransferFlow imports the
demonstration packs, writes the generated exp/runs into the flow workspace and
returns the flow entries: the trainings of the sensor processors (SPs), one
regressor stage per (method, target robot, seed), and one comparison of all of
them.

The packs come from a run of the transfer_datasets family (one run per
machine, e.g. transfer_datasets/szenes.yaml). list_available_packs() shows the
packs of the current machine.
"""
import sys
sys.path.append("..")

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

import pathlib
import yaml

from flow import flow_entry
from demonstration.demopack import import_demopack, group_chooser_sp_vp_standard
from robot.position_factory import robot_output_size

# the families the flows copy into their workspace
FAMILIES = [
    "demonstration", "transfer_datasets",
    "robot_al5d", "robot_franka", "robot_g1", "robot_inspire_hand",
    "sensorprocessing_conv_vae", "sensorprocessing_random_projection_cnn",
    "sensorprocessing_transfer_encoder", "sensorprocessing_transfer_meta_ridge",
    "sensorprocessing_transfer_adabn", "sensorprocessing_transfer_randproj_vit",
    "visual_proprioception_transfer", "visual_proprioception_transfer_compare",
]

# the designs a flow can include
DESIGNS = ["baseline", "D1", "D2", "D3", "D4a", "D4b", "D5", "D5-fomaml"]

# the image size of each backbone
IMAGE_SIZE = {"vgg19": [256, 256], "vgg19_keypoints": [256, 256], "vit_b_16": [224, 224]}


def list_available_packs(demopacks_path=None):
    """Prints the demonstration packs in the demopacks_path of the current
    machine: the robot named in the pack yaml, the cameras and the number of
    demonstrations. Use it to write the transfer_datasets run of a machine."""
    if demopacks_path is None:
        demopacks_path = Config()["demopacks_path"]
    for path in sorted(pathlib.Path(demopacks_path).expanduser().iterdir()):
        if not path.is_dir():
            continue
        demos = [d for d in path.iterdir() if d.is_dir()]
        pack_yaml = path / f"{path.name}.yaml"
        info = yaml.safe_load(pack_yaml.read_text()) if pack_yaml.exists() else {}
        robot = f"{info.get('robot_exp', '?')}/{info.get('robot_run', '?')}"
        cameras = list(info.get("cameras", {}))
        print(f"{path.name}: {len(demos)} demonstrations, robot {robot}, cameras {cameras}")


class TransferFlow:
    """Generates the exp/runs and the flow entries of a transfer flow.

    params (the parameters of the flow notebook):
        flow_name, datasets_run, packs (the short names of the packs to use),
        targets, backbones, designs, latent_size, seeds, k_shots, frame_step,
        epochs_sp, epochs_vae, meta_steps, regressor_steps, regressors,
        eval_group, creation_style
    """

    def __init__(self, params):
        self.p = params
        datasets = Config().get_experiment("transfer_datasets", params["datasets_run"], create_data_dir=False)
        self.packs = {name: datasets["packs"][name] for name in params["packs"]}
        self.entries = []
        self.trained = set()
        self.vp_runs = []

    # -- packs ---------------------------------------------------------------

    def import_packs(self):
        """Imports every pack into the workspace and records its groups"""
        self.selections = {}
        for name, pack in self.packs.items():
            path = pathlib.Path(Config()["demopacks_path"], pack["pack"]).expanduser()
            self.selections[name] = import_demopack(path, group_chooser_sp_vp_standard)

    def items(self, name, group):
        """The [pack, demonstration, camera] items of a group of a pack"""
        pack = self.packs[name]
        return [[pack["pack"], demo, pack["camera"]] for demo in self.selections[name][group]]

    def output_size(self, name):
        pack = self.packs[name]
        return robot_output_size(Config().get_experiment(pack["robot_exp"], pack["robot_run"], create_data_dir=False))

    def sources(self, target):
        """The source robots of a target: all the other packs of the flow"""
        return [name for name in self.packs if name != target]

    # -- writing -------------------------------------------------------------

    def write(self, family, run, values):
        path = pathlib.Path(Config().get_exprun_path(), family, run + ".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(values, f)

    def train(self, family, run, values):
        """Writes an SP run and queues its training, once"""
        if run in self.trained:
            return family, run
        self.write(family, run, values)
        self.entries.append(flow_entry(f"Train_{run}", family, run, 0, self.p["creation_style"]))
        self.trained.add(run)
        return family, run

    # -- the sensor processors ------------------------------------------------

    def robot_entry(self, name):
        pack = self.packs[name]
        return {"name": name, "robot_exp": pack["robot_exp"], "robot_run": pack["robot_run"],
                "output_size": self.output_size(name),
                "training_data": self.items(name, "sp_training"),
                "validation_data": self.items(name, "sp_validation")}

    def transfer_encoder(self, backbone, sources, seed, mask=None, target=None):
        """A TransferEncoder trained on the sources. mask: None, "gray" or
        "swap" (D3); a swap encoder depends on the target, whose packs it must
        not use as backgrounds."""
        suffix = {None: "", "gray": "-mask", "swap": "-maskswap"}[mask]
        run = f"te_{backbone}{suffix}_{self.p['latent_size']}_{'+'.join(sources)}_s{seed}"
        values = {"backbone": backbone, "latent_size": self.p["latent_size"],
                  "image_size": IMAGE_SIZE[backbone], "seed": seed, "epochs": self.p["epochs_sp"],
                  "frame_step": self.p["frame_step"],
                  "robots": [self.robot_entry(name) for name in sources]}
        if mask is not None:
            values["robot_mask"] = {"threshold": 0.08, "dilate": 2,
                                    "fill": "gray" if mask == "gray" else "background_swap"}
        if mask == "swap":
            run += f"_for-{target}"
            others = [name for name in self.packs if name not in sources and name != target]
            values["background_swap_items"] = [self.items(name, "sp_training")[0] for name in others]
            values["color_jitter"] = 0.2
        return self.train("sensorprocessing_transfer_encoder", run, values)

    def conv_vae(self, pack, group, seed):
        """A Conv-VAE trained on the frames of a group of a pack (no labels)"""
        run = f"vae_{self.p['latent_size']}_{pack}-{group}_s{seed}"
        values = {"latent_size": self.p["latent_size"], "epochs": self.p["epochs_vae"],
                  # Train_Conv_VAE saves the final model itself; no intermediate checkpoints
                  "save_period": self.p["epochs_vae"], "seed": seed,
                  "training_data": self.items(pack, group),
                  "validation_data": self.items(pack, "sp_validation")}
        return self.train("sensorprocessing_conv_vae", run, values)

    def randproj_cnn(self, seed):
        """Reference R1: VGG19 features pooled to 4x4 and a random projection"""
        run = f"rp_vgg19_pool4_{self.p['latent_size']}_s{seed}"
        self.write("sensorprocessing_random_projection_cnn", run, {
            "model": "VGG19RandomProjection", "latent_size": self.p["latent_size"],
            "feature_pool_size": [4, 4], "projection_seed": 41983 + seed})
        return "sensorprocessing_random_projection_cnn", run

    def randproj_vit(self, seed):
        """Reference R1 with the ViT of the paper"""
        run = f"rp_vit_b_16_{self.p['latent_size']}_s{seed}"
        self.write("sensorprocessing_transfer_randproj_vit", run, {
            "latent_size": self.p["latent_size"], "projection_seed": 52817 + seed})
        return "sensorprocessing_transfer_randproj_vit", run

    def adabn(self, base, image_size, target, scope):
        """D4a: the base SP adapted to the unlabeled vp_training frames of the target"""
        family, base_run = base
        run = f"adabn-{scope}_{base_run}_on-{target}"
        self.write("sensorprocessing_transfer_adabn", run, {
            "base_sp_experiment": family, "base_sp_run": base_run,
            "image_size": image_size, "latent_size": self.p["latent_size"], "adapt_scope": scope,
            "adaptation_data": self.items(target, "vp_training"),
            "adaptation_frame_step": self.p["frame_step"]})
        return "sensorprocessing_transfer_adabn", run

    def meta_ridge(self, sources, seed, solver):
        """D5: the meta-trained projection on the sources"""
        run = f"meta-{solver}_{self.p['latent_size']}_{'+'.join(sources)}_s{seed}"
        values = {"latent_size": self.p["latent_size"], "seed": seed, "inner_solver": solver,
                  "meta_steps": self.p["meta_steps"], "frame_step": self.p["frame_step"],
                  "robots": [self.robot_entry(name) for name in sources]}
        return self.train("sensorprocessing_transfer_meta_ridge", run, values)

    # -- the regressor stages and the comparison ------------------------------

    def vp(self, method, sp, target, seed, regressors=None):
        """The regressor stage of a method on a target"""
        sp_family, sp_run = sp
        run = f"vp_{method}_to-{target}_s{seed}"
        pack = self.packs[target]
        self.write("visual_proprioception_transfer", run, {
            "method": method, "target": target, "seed": seed,
            "sp_experiment": sp_family, "sp_run": sp_run,
            "robot_exp": pack["robot_exp"], "robot_run": pack["robot_run"],
            "training_data": self.items(target, "vp_training"),
            "testing_data": self.items(target, self.p["eval_group"]),
            "frame_step": self.p["frame_step"], "k_shots": self.p["k_shots"],
            "regressors": regressors if regressors is not None else self.p["regressors"],
            "regressor_steps": self.p["regressor_steps"]})
        self.entries.append(flow_entry(f"VP_{run}", "visual_proprioception_transfer", run, 0, self.p["creation_style"]))
        self.vp_runs.append(run)

    def compare(self):
        run = f"compare_{self.p['flow_name']}"
        self.write("visual_proprioception_transfer_compare", run, {"tocompare": self.vp_runs})
        self.entries.append(flow_entry(f"Compare_{run}", "visual_proprioception_transfer_compare", run, 0,
                                       self.p["creation_style"]))
        return run

    # -- the queue -----------------------------------------------------------

    def build(self):
        """All the entries of the flow, in order: each SP before its regressors,
        the comparison last"""
        designs = self.p["designs"]
        for seed in self.p["seeds"]:
            for target in self.p["targets"]:
                sources = self.sources(target)
                for backbone in self.p["backbones"]:
                    if "baseline" in designs:
                        self.vp(f"native-{backbone}", self.transfer_encoder(backbone, [target], seed), target, seed)
                        for source in sources:
                            self.vp(f"single-{backbone}-from-{source}",
                                    self.transfer_encoder(backbone, [source], seed), target, seed)
                    if "D1" in designs:
                        self.vp(f"multi-{backbone}", self.transfer_encoder(backbone, sources, seed), target, seed)
                    if "D4a" in designs and backbone == "vit_b_16":
                        for source in sources:
                            base = self.transfer_encoder(backbone, [source], seed)
                            self.vp(f"adabn-{backbone}-from-{source}",
                                    self.adabn(base, IMAGE_SIZE[backbone], target, "projection"), target, seed)
                if "baseline" in designs:
                    self.vp("randproj-vgg19", self.randproj_cnn(seed), target, seed)
                    if "vit_b_16" in self.p["backbones"]:
                        self.vp("randproj-vit_b_16", self.randproj_vit(seed), target, seed)
                if "D2" in designs:
                    self.vp("kp-native", self.transfer_encoder("vgg19_keypoints", [target], seed), target, seed)
                    for source in sources:
                        self.vp(f"kp-from-{source}", self.transfer_encoder("vgg19_keypoints", [source], seed),
                                target, seed)
                    if "D1" in designs:
                        self.vp("kp-multi", self.transfer_encoder("vgg19_keypoints", sources, seed), target, seed)
                if "D3" in designs:
                    for source in sources:
                        self.vp(f"mask-from-{source}",
                                self.transfer_encoder("vgg19", [source], seed, mask="gray"), target, seed)
                        self.vp(f"maskswap-from-{source}",
                                self.transfer_encoder("vgg19", [source], seed, mask="swap", target=target),
                                target, seed)
                if "D4b" in designs or "D4a" in designs:
                    vaes = {source: self.conv_vae(source, "sp_training", seed) for source in sources}
                if "D4b" in designs:
                    self.vp("vae-native", self.conv_vae(target, "sp_training", seed), target, seed)
                    self.vp("vae-target", self.conv_vae(target, "vp_training", seed), target, seed)
                    for source in sources:
                        self.vp(f"vae-from-{source}", vaes[source], target, seed)
                if "D4a" in designs:
                    for source in sources:
                        self.vp(f"adabn-vae-from-{source}", self.adabn(vaes[source], [64, 64], target, "all"),
                                target, seed)
                if "D5" in designs:
                    self.vp("meta-ridge", self.meta_ridge(sources, seed, "ridge"), target, seed,
                            regressors=["ridge_meta", "mlp"])
                if "D5-fomaml" in designs:
                    self.vp("meta-fomaml", self.meta_ridge(sources, seed, "fomaml"), target, seed,
                            regressors=["ridge", "mlp"])
        compare_run = self.compare()
        return self.entries, compare_run
