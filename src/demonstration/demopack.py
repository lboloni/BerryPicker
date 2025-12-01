"""
demopack.py

A demo pack is a self contained file that contains a set of demonstrations that later can be used to be imported into an experiment, and used as training and validation data at various levels.

Normally a demo pack is a zip file or a directory, containing a list of demonstrations
"""
import pathlib
import zipfile
import shutil
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

def group_sortout(prefix, frompos, topos, demo_names, retval, selection):
    """Utility function for sorting out a group"""
    demo_prefix = demo_names[frompos:topos]
    selection[prefix] = []
    for i, demo_name in enumerate(demo_prefix):
        name = f"{prefix}_{i:05d}"
        retval[name] = demo_name
        selection[prefix].append(name)

def group_chooser_sp_bc_trivial(demo_names):
    """Copy all the data to sp, bc both training and testing. Note that this overlaps the training and testing so it is not a good idea in general."""
    retval = {}; selection = {}
    sepend = len(demo_names)
    group_sortout("sp_training", 0, sepend, demo_names, retval, selection)
    group_sortout("sp_validation", 0, sepend, demo_names, retval, selection)
    group_sortout("sp_testing", 0, sepend, demo_names, retval, selection)
    group_sortout("bc_training", 0, sepend, demo_names, retval, selection)
    group_sortout("bc_validation", 0, sepend, demo_names, retval, selection)
    group_sortout("bc_testing", 0, sepend, demo_names, retval, selection)
    return retval, selection

def group_chooser_sp_bc_standard(demo_names):
    """Standard group chooser for the data for the behavior cloning flow
    Groups will be as follows:
       * sp training data: 40%
       * sp validation data: 20%, from the bc training
       * sp testing data: 20%, shared with bc testing
       * bc training data: 40%
       * bc validation data: 20%, from the sp training
       * bc testing data: 20%, shared with sp testing
    """
    retval = {}; selection = {}
    # the separating numbers
    sep1half = int(len(demo_names) * 0.2)
    sep1 = int(len(demo_names) * 0.4)
    sep2half = int(len(demo_names) * 0.6)
    sep2 = int(len(demo_names) * 0.8)
    sepend = len(demo_names)
    group_sortout("sp_training", 0, sep1, demo_names, retval, selection)
    group_sortout("sp_validation", sep1, sep2half, demo_names, retval, selection)
    group_sortout("sp_testing", sep2, sepend, demo_names, retval, selection)
    group_sortout("bc_training", sep1, sep2, demo_names, retval, selection)
    group_sortout("bc_validation", 0, sep1half, demo_names, retval, selection)
    group_sortout("bc_testing", sep2, sepend, demo_names, retval, selection)
    return retval, selection

def group_chooser_sp_vp_trivial(demo_names):
    """Copy all the data to sp, bc both training and testing. Note that this overlaps the training and testing so it is not a good idea in general."""
    retval = {}; selection = {}
    # the separating numbers
    sep1half = int(len(demo_names) * 0.2)
    sep1 = int(len(demo_names) * 0.4)
    sep2half = int(len(demo_names) * 0.6)
    sep2 = int(len(demo_names) * 0.8)
    sepend = len(demo_names)
    group_sortout("sp_training", 0, sep1, demo_names, retval, selection)
    group_sortout("sp_validation", sep1, sep2half, demo_names, retval, selection)
    group_sortout("sp_testing", sep2, sepend, demo_names, retval, selection)
    group_sortout("vp_training", sep1, sep2, demo_names, retval, selection)
    group_sortout("vp_validation", 0, sep1half, demo_names, retval, selection)
    group_sortout("vp_testing", sep2, sepend, demo_names, retval, selection)
    return retval, selection

def group_chooser_sp_vp_standard(demo_names):
    """Standard group chooser for the data for visual proprioception.
    Groups will be as follows:
       * sp training data: 40%
       * sp validation data: 20%, from the vp training
       * sp testing data: 20%, shared with vp testing
       * vp training data: 40%
       * vp validation data: 20%, from the sp training
       * vp testing data: 20%, shared with sp testing
    """
    retval = {}; selection = {}
    # the separating numbers
    sep1half = int(len(demo_names) * 0.2)
    sep1 = int(len(demo_names) * 0.4)
    sep2half = int(len(demo_names) * 0.6)
    sep2 = int(len(demo_names) * 0.8)
    sepend = len(demo_names)
    group_sortout("sp_training", 0, sep1, demo_names, retval, selection)
    group_sortout("sp_validation", sep1, sep2half, demo_names, retval, selection)
    group_sortout("sp_testing", sep2, sepend, demo_names, retval, selection)
    group_sortout("vp_training", sep1, sep2, demo_names, retval, selection)
    group_sortout("vp_validation", 0, sep1half, demo_names, retval, selection)
    group_sortout("vp_testing", sep2, sepend, demo_names, retval, selection)
    return retval, selection


# def import_demopack(demo_path, group_chooser):
#     """Imports a demopack into the results directory of the """
#     assert(demo_path.is_dir())
#     # get these from the config
#     exprun_path = Config().get_exprun_path()
#     results_path = Config().get_results_path()
#     demoname = demo_path.stem
#     demonstration_yaml = pathlib.Path(demo_path, demoname + ".yaml")
#     demo_names = [d.name for d in demo_path.iterdir() if d.is_dir()]
#     target_yaml = pathlib.Path(exprun_path, "demonstration", demoname + ".yaml")
#     target_dir = pathlib.Path(results_path, "demonstration", demoname)
#     do_copy = True
#     if target_dir.exists():
#         print(f"*** import_demopack: {demo_path}, target directory {target_dir}  already exists, not copying")
#         do_copy = False
#     else:
#         target_dir.mkdir(exist_ok=True, parents=True)
#     demo_exprun = pathlib.Path(exprun_path, "demonstration")
#     demo_exprun.mkdir(exist_ok=True, parents=True)
#     # create the defaults
#     defaults_yaml = pathlib.Path(demo_exprun, "_defaults_demonstration.yaml")
#     defaults_yaml.touch()
#     # copy the target yaml both to exprun and results
#     target_yaml.parent.mkdir(exist_ok=True, parents=True)
#     shutil.copy2(demonstration_yaml, target_yaml)
#     shutil.copy2(demonstration_yaml, pathlib.Path(demo_exprun, demoname + ".yaml"))
#     # copy the different demonstrations with different names
#     tocopy, selection = group_chooser(demo_names)
#     for target in tocopy:
#         destdir = pathlib.Path(target_dir, target)
#         sourcedir = pathlib.Path(demo_path, tocopy[target])
#         if do_copy:
#             shutil.copytree(sourcedir, destdir)
#     return selection





def group_chooser_sp_vp_multiview(demo_names):
    """Group chooser for multiview visual proprioception.

    This chooser works the same as sp_vp_standard but is specifically
    labeled for multiview use cases. The actual multiview handling
    (providing lists of cameras instead of single camera) is done
    at the Flow level when constructing training_data lists.

    Groups will be as follows:
       * sp_training: 40% - for training the sensor processing encoder
       * sp_validation: 20% - for validating the encoder during training
       * sp_testing: 20% - for testing encoder performance
       * vp_training: 40% - for training the visual proprioception regressor
       * vp_validation: 20% - for validating the regressor
       * vp_testing: 20% - for testing regressor performance
    """
    retval = {}
    selection = {}
    # the separating numbers
    sep1half = int(len(demo_names) * 0.2)
    sep1 = int(len(demo_names) * 0.4)
    sep2half = int(len(demo_names) * 0.6)
    sep2 = int(len(demo_names) * 0.8)
    sepend = len(demo_names)
    group_sortout("sp_training", 0, sep1, demo_names, retval, selection)
    group_sortout("sp_validation", sep1, sep2half, demo_names, retval, selection)
    group_sortout("sp_testing", sep2, sepend, demo_names, retval, selection)
    group_sortout("vp_training", sep1, sep2, demo_names, retval, selection)
    group_sortout("vp_validation", 0, sep1half, demo_names, retval, selection)
    group_sortout("vp_testing", sep2, sepend, demo_names, retval, selection)
    return retval, selection


def get_available_cameras(demo_path):
    """Get the list of available cameras from a demopack.

    Scans the first demonstration directory to find available camera names
    by looking at the image file naming convention ({frame}_{camera}.jpg).

    Args:
        demo_path: Path to the demopack directory

    Returns:
        List of camera names found in the demonstrations
    """
    cameras = set()

    # Find first demonstration directory
    for item in demo_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('_'):
            # Look for image files in this demo
            for img_file in item.glob("*.jpg"):
                # Expected format: {frame}_{camera}.jpg
                name = img_file.stem
                if '_' in name:
                    parts = name.rsplit('_', 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        cameras.add(parts[1])
            # Only check first demo
            if cameras:
                break

    return sorted(list(cameras))


def import_demopack(demo_path, group_chooser):
    """Imports a demopack into the results directory.

    Args:
        demo_path: Path to the demopack directory
        group_chooser: Function that determines how to split demos into groups

    Returns:
        selection: Dictionary mapping group names to lists of demo names
    """
    assert demo_path.is_dir(), f"Demo path {demo_path} is not a directory"

    # Get paths from config
    exprun_path = Config().get_exprun_path()
    results_path = Config().get_results_path()
    demoname = demo_path.stem
    demonstration_yaml = pathlib.Path(demo_path, demoname + ".yaml")
    demo_names = [d.name for d in demo_path.iterdir() if d.is_dir()]
    target_yaml = pathlib.Path(exprun_path, "demonstration", demoname + ".yaml")
    target_dir = pathlib.Path(results_path, "demonstration", demoname)

    do_copy = True
    if target_dir.exists():
        print(f"*** import_demopack: {demo_path}, target directory {target_dir} already exists, not copying")
        do_copy = False
    else:
        target_dir.mkdir(exist_ok=True, parents=True)

    demo_exprun = pathlib.Path(exprun_path, "demonstration")
    demo_exprun.mkdir(exist_ok=True, parents=True)

    # Create the defaults file
    defaults_yaml = pathlib.Path(demo_exprun, "_defaults_demonstration.yaml")
    defaults_yaml.touch()

    # Copy the target yaml both to exprun and results
    target_yaml.parent.mkdir(exist_ok=True, parents=True)
    if demonstration_yaml.exists():
        shutil.copy2(demonstration_yaml, target_yaml)
        shutil.copy2(demonstration_yaml, pathlib.Path(demo_exprun, demoname + ".yaml"))

    # Copy the different demonstrations with different names
    tocopy, selection = group_chooser(demo_names)
    for target in tocopy:
        destdir = pathlib.Path(target_dir, target)
        sourcedir = pathlib.Path(demo_path, tocopy[target])
        if do_copy:
            shutil.copytree(sourcedir, destdir)

    # Also store available cameras in selection for convenience
    cameras = get_available_cameras(demo_path)
    if cameras:
        selection["_cameras"] = cameras
        print(f"*** import_demopack: Found cameras: {cameras}")

    return selection


def import_demopack_multiview(demo_path, group_chooser, cameras=None):
    """Imports a demopack with explicit multiview camera configuration.

    This is a convenience wrapper around import_demopack that also
    returns information about available cameras.

    Args:
        demo_path: Path to the demopack directory
        group_chooser: Function that determines how to split demos into groups
        cameras: Optional list of camera names to use. If None, auto-detect.

    Returns:
        tuple: (selection dict, list of camera names)
    """
    selection = import_demopack(demo_path, group_chooser)

    if cameras is None:
        cameras = selection.get("_cameras", get_available_cameras(demo_path))

    return selection, cameras
