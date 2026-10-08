"""Tests for the exp/run generators of the RCCO behavior cloning flows."""

import pathlib
import sys
import tempfile
import unittest


SOURCE_ROOT = pathlib.Path(__file__).parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
from robot_controller.chain_training_model import ChainTrainingModel
from robot_controller.graph_robot_controller import load_controller_spec
from robot_controller.rcco_flow import (
    CONTROLLER_TYPES, SP_TYPES, flow_families, generate_compare,
    generate_controller_stages, generate_sp_stage)


SP_NOTEBOOKS = {
    "resnet50": "sensorprocessing/Train_ProprioTuned_CNN.ipynb",
    "vgg19": "sensorprocessing/Train_ProprioTuned_CNN.ipynb",
    "vae": "sensorprocessing/Train_Conv_VAE_Neo.ipynb",
    "vae_gan": "sensorprocessing/Train_VAE_GAN.ipynb",
}


class TestRCCOFlowGenerators(unittest.TestCase):
    """Generates into a temporary exprun directory with the real families."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.temporary.name)
        self.old_exprun_path = Config().get_exprun_path()
        self.old_results_path = Config().get_results_path()
        for name in ["expruns", "results"]:
            (root / name).mkdir()
        Config().set_exprun_path(root / "expruns")
        Config().set_results_path(root / "results")
        for family in flow_families(list(SP_TYPES)):
            Config().copy_experiment(family)
        self.data = {
            group: [["pack", f"{group}_00000", "dev2"]]
            for group in ["sp_training", "sp_validation", "bc_training",
                          "bc_validation", "bc_testing"]}

    def tearDown(self):
        Config().set_exprun_path(self.old_exprun_path)
        Config().values["experiment_data"] = self.old_results_path
        self.temporary.cleanup()

    def test_sp_stage_for_every_sp_type(self):
        for sp_type in SP_TYPES:
            entry = generate_sp_stage(sp_type, self.data, 3, "exist-ok")
            self.assertEqual(entry["notebook"], SP_NOTEBOOKS[sp_type], sp_type)
            sp = SP_TYPES[sp_type]
            exp_sp = Config().get_experiment(
                sp["experiment"], f"_flow_sp_{sp_type}", create_data_dir=False)
            self.assertEqual(exp_sp["epochs"], 3)
            self.assertEqual(exp_sp["training_data"], self.data["sp_training"])

    def test_controller_stages_for_every_controller_type(self):
        for sp_type in ["vgg19", "vae"]:
            generate_sp_stage(sp_type, self.data, 3, "exist-ok")
            for controller_type, controller in CONTROLLER_TYPES.items():
                key = f"{sp_type}_{controller_type}"
                entries = generate_controller_stages(
                    sp_type, controller_type, self.data, 2, 1, "exist-ok")
                self.assertEqual(
                    [entry["notebook"] for entry in entries],
                    ["robot_controller/Train_RCCO.ipynb",
                     "robot_controller/Verify_RCCO.ipynb"], key)

                exp_trec = Config().get_experiment(
                    "robot_controller_training", f"_flow_trec_{key}",
                    create_data_dir=False)
                self.assertEqual(
                    [stage["epochs"] for stage in exp_trec["stages"]], [2, 1])
                self.assertEqual(
                    exp_trec["validation_data"], self.data["bc_validation"])

                # the generated graph validates, and trains as a chain
                exp_roco = Config().get_experiment(
                    "robot_controller", f"_flow_roco_{key}",
                    create_data_dir=False)
                spec = load_controller_spec(exp_roco, None)
                self.assertEqual(spec["components"]["encoder"]["type"],
                                 SP_TYPES[sp_type]["rcco_type"])
                model = ChainTrainingModel(spec)
                self.assertEqual(model.head_type, controller["head"]["rcco-type"])
                self.assertEqual(
                    model.context_mode,
                    None if controller["lstm"] is None
                    else controller["lstm"]["context_mode"])
                self.assertEqual(
                    set(exp_trec["initial_states"]), set(model.component_labels))

                exp_verify = Config().get_experiment(
                    "robot_controller_verify", f"_flow_verify_{key}",
                    create_data_dir=False)
                self.assertEqual(exp_verify["trec_run"], f"_flow_trec_{key}")
                self.assertEqual(exp_verify["testing_data"], self.data["bc_testing"])

    def test_compare_lists_the_verify_runs(self):
        entry = generate_compare(
            [("vgg19", "mlp"), ("vae", "lstm_mlp")], "comparison", "exist-ok")
        self.assertEqual(entry["notebook"], "robot_controller/Compare_RCCO.ipynb")
        exp = Config().get_experiment(
            "robot_controller_compare", "_flow_compare", create_data_dir=False)
        self.assertEqual(
            exp["verify_runs"], ["_flow_verify_vgg19_mlp", "_flow_verify_vae_lstm_mlp"])
        self.assertEqual(exp["labels"], ["vgg19 + mlp", "vae + lstm_mlp"])


if __name__ == "__main__":
    unittest.main()
