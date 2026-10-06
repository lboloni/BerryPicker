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
from robot_controller.graph_robot_controller import load_controller_spec
from robot_controller.rcco_flow import (
    SP_TYPES, flow_families, generate_compare, generate_controller_stages)


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
        self.old_exprun_path = Config().experiment_path
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
        Config().experiment_path = self.old_exprun_path
        Config().values["experiment_data"] = self.old_results_path
        self.temporary.cleanup()

    def test_controller_stages_for_every_sp_type(self):
        for sp_type in SP_TYPES:
            entries = generate_controller_stages(
                sp_type, self.data, 3, 2, 1, "exist-ok")
            self.assertEqual(
                [entry["notebook"] for entry in entries],
                [SP_NOTEBOOKS[sp_type], "robot_controller/Train_RCCO.ipynb",
                 "robot_controller/Verify_RCCO.ipynb"], sp_type)

            sp = SP_TYPES[sp_type]
            exp_sp = Config().get_experiment(
                sp["experiment"], f"_flow_sp_{sp_type}", create_data_dir=False)
            self.assertEqual(exp_sp["epochs"], 3)
            self.assertEqual(exp_sp["training_data"], self.data["sp_training"])

            exp_trec = Config().get_experiment(
                "robot_controller_training", f"_flow_trec_{sp_type}",
                create_data_dir=False)
            self.assertEqual(
                [stage["epochs"] for stage in exp_trec["stages"]], [2, 1])
            self.assertEqual(exp_trec["validation_data"], self.data["bc_validation"])

            exp_roco = Config().get_experiment(
                exp_trec["controller"]["exp"], exp_trec["controller"]["run"],
                create_data_dir=False)
            spec = load_controller_spec(exp_roco, None)
            encoder = spec["components"]["cnn_encoder"]
            self.assertEqual(encoder["type"], sp["rcco_type"])
            self.assertEqual(encoder["sensor_exp"]["latent_size"], 256)

            exp_verify = Config().get_experiment(
                "robot_controller_verify", f"_flow_verify_{sp_type}",
                create_data_dir=False)
            self.assertEqual(exp_verify["trec_run"], f"_flow_trec_{sp_type}")
            self.assertEqual(exp_verify["testing_data"], self.data["bc_testing"])

    def test_compare_lists_the_verify_runs(self):
        entry = generate_compare(["vgg19", "vae"], "comparison", "exist-ok")
        self.assertEqual(entry["notebook"], "robot_controller/Compare_RCCO.ipynb")
        exp = Config().get_experiment(
            "robot_controller_compare", "_flow_compare", create_data_dir=False)
        self.assertEqual(
            exp["verify_runs"], ["_flow_verify_vgg19", "_flow_verify_vae"])
        self.assertEqual(exp["labels"], ["vgg19", "vae"])

    def test_run_names_differ_between_sp_types(self):
        names = [
            entry["run"]
            for sp_type in SP_TYPES
            for entry in generate_controller_stages(
                sp_type, self.data, 1, 1, 1, "exist-ok")]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
