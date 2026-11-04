"""
external_proprioception.py

Access to external robot pose estimators (RoboPEPP, HPE) from BerryPicker; see
DESIGN-Import-Competitors.md. An external estimator is an exp/run of the
family external_proprioception. Its stage notebook
(Run_External_Proprioception.ipynb):

1. exports the frames of its demonstrations as a DREAM-format folder
   (export_dream);
2. runs the driver of the method (external_drivers/<method>_predict.py) in
   the method's own Python environment, which writes the predicted joint
   angles of every frame to an .npz file (run_external);
3. predict_positions() in visproprio_helper then reads them back as positions
   of the run's robot (load_external_predictions), like the predictions of
   our own models.

The machine-specific paths are in the machine settings, under
external_proprioception: the directories of the source sequences, and for
every method its code directory, its Python and its checkpoint directory.
"""

import os
import pathlib
import subprocess

import numpy as np

from exp_run_config import Config
from demonstration.demonstration import Demonstration
from robot.position_factory import action_key, create_position


def export_dir(exp, datasetname):
    """The DREAM-format folder into which the frames of exp[datasetname] are exported.
    It is named after the source sequence, because the methods recognize the real
    DREAM sequences by the name of their folder (e.g. panda-3cam_realsense)."""
    sequence = source_sequence(exp, datasetname)
    return pathlib.Path(exp.data_dir(), f"export_{datasetname}", sequence.split()[-1])


def angles_file(exp, datasetname):
    """The .npz file into which the driver writes the joint angles predicted for exp[datasetname]"""
    return pathlib.Path(exp.data_dir(), f"angles_{datasetname}.npz")


def demonstrations(exp, datasetname):
    """The demonstrations of exp[datasetname]"""
    return [Demonstration(Config().get_experiment("demonstration", pack), demo_name)
            for pack, demo_name, _camera in exp[datasetname]]


def source_sequence(exp, datasetname):
    """The source sequence of the demonstrations of exp[datasetname]; they must all come from the same one"""
    sequences = {demo.metadata["source_sequence"] for demo in demonstrations(exp, datasetname)}
    assert len(sequences) == 1, f"{exp['name']}: the {datasetname} come from several sequences {sequences}"
    return sequences.pop()


def export_dream(exp, datasetname):
    """Exports the frames of the demonstrations of exp[datasetname] as a DREAM-format folder:
    NNNNNN.rgb.jpg and NNNNNN.json, numbered in the order of the demonstrations, and
    _camera_settings.json. The demonstrations must come from a DREAM sequence
    (source_sequence and source_frame_offset in their metadata); the files are symbolic
    links to the original frames of the sequence, at their original resolution, so a
    driver finds the original frame of an exported one by resolving the link.
    Returns the folder."""
    sequence = source_sequence(exp, datasetname)
    source = pathlib.Path(Config()["external_proprioception"]["sequences"][sequence]).expanduser()
    target = export_dir(exp, datasetname)
    target.mkdir(parents=True, exist_ok=True)

    def link(name, original):
        path = pathlib.Path(target, name)
        path.unlink(missing_ok=True)
        path.symlink_to(pathlib.Path(source, original))

    count = 0
    for demo in demonstrations(exp, datasetname):
        offset = demo.metadata["source_frame_offset"]
        for i in range(demo.metadata["maxsteps"]):
            for extension in ("rgb.jpg", "json"):
                link(f"{count:06d}.{extension}", f"{offset + i:06d}.{extension}")
            count += 1
    link("_camera_settings.json", "_camera_settings.json")
    print(f"***external_proprioception***: exported {count} frames of {sequence} to {target}")
    return target


def run_external(exp, datasetname):
    """Runs the driver of exp["method"] on the exported frames of exp[datasetname], with the
    Python of the method's environment. The driver writes the predicted joint angles to
    angles_file(exp, datasetname). The arguments of the driver are the machine-specific
    ones (machine settings) and those of the run (driver_args); a leading ~ in them is
    expanded."""
    settings = Config()["external_proprioception"][exp["method"]]
    driver = pathlib.Path(__file__).parent / "external_drivers" / f"{exp['method']}_predict.py"
    arguments = {"code-dir": settings["code_dir"],
                 "checkpoint": str(pathlib.Path(settings["checkpoint_dir"], exp["checkpoint"]).expanduser()),
                 "frames": str(export_dir(exp, datasetname)),
                 "out": str(angles_file(exp, datasetname))}
    arguments.update(settings.get("driver_args", {}))
    arguments.update(exp["driver_args"])
    command = [str(pathlib.Path(settings["python"]).expanduser()), str(driver)]
    for name, value in arguments.items():
        command += [f"--{name}", os.path.expanduser(str(value))]
    print("***external_proprioception***: running", " ".join(command))
    subprocess.run(command, check=True)


def load_external_predictions(exp, exp_robot, datasetname):
    """The positions an external run predicted for exp[datasetname], normalized by the robot
    run exp_robot, with the targets from the demonstrations. The joints of the robot run are
    taken from the predicted ones by name, so the run's joints, order and limits apply.
    Predictions outside the limits map outside [0, 1], as those of our regressors can.
    Returns a dictionary with 'targets', 'predictions' and 'lengths'."""
    data = np.load(angles_file(exp, datasetname))
    names = [str(name) for name in data["joint_names"]]
    predictions = np.asarray([
        create_position(exp_robot, dict(zip(names, angles.tolist()))).to_normalized_vector(
            exp_robot, enforce_limits=False)
        for angles in data["angles"]], dtype=np.float32)
    targets = []
    lengths = []
    for demo in demonstrations(exp, datasetname):
        for i in range(demo.metadata["maxsteps"]):
            targets.append(demo.get_action(i, action_key(exp_robot), exp_robot).to_normalized_vector(exp_robot))
        lengths.append(demo.metadata["maxsteps"])
    targets = np.asarray(targets, dtype=np.float32)
    assert len(targets) == len(predictions), (
        f"{exp['name']}: {len(predictions)} predicted frames for {len(targets)} frames of {datasetname}")
    return {"targets": targets, "predictions": predictions, "lengths": lengths}
