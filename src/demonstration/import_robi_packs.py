"""Import the demonstration packs of Robi's multi-robot experiments (2025-10).

Robi converted Unitree G1 (Isaac Sim), Franka Panda (DREAM) and Inspire hand
recordings into BerryPicker demonstrations, but stored the joints of the G1
and the Panda under the field names of the AL5D (rc-position-target). This
script copies such a pack into a clean demopack:

* the joints are stored under their own names, under the action key
  joint-position (see data/expruns/robot_franka, robot_g1, robot_inspire_hand
  and robot/position_factory.py);
* _metadata.yaml lists the one camera that exists (dev000), and
  _annotation.yaml has one empty annotation per step (Robi's were stale
  copies of an AL5D demonstration);
* the pack yaml names the robot (robot_exp, robot_run);
* the frames (NNNNN_dev000.jpg) are copied unchanged.

The source is never modified. See src/visual_proprioception/PLAN-MultiRobot.md.

Usage:
    python import_robi_packs.py SOURCE_PACK TARGET_PACK {g1,panda,hand}
"""

import argparse
import pathlib
import shutil

import yaml

# the AL5D field names under which Robi stored the joints, mapped back to the
# joints (from his convert_yaml_g1.py and convert_jsons_to_yaml.py)
FIELD_TO_JOINT = {
    "g1": {
        "height": "left_elbow_joint",
        "distance": "left_shoulder_pitch_joint",
        "heading": "left_shoulder_yaw_joint",
        "wrist_angle": "left_wrist_pitch_joint",
        "wrist_rotation": "left_wrist_roll_joint",
        "gripper": "left_shoulder_roll_joint",
        "7th_number": "left_wrist_yaw_joint",
    },
    "panda": {
        "distance": "panda_joint1",
        "heading": "panda_joint2",
        "height": "panda_joint3",
        "wrist_angle": "panda_joint4",
        "wrist_rotation": "panda_joint5",
        "7th_number": "panda_joint6",
        "gripper": "panda_joint7",
    },
    # the hand pack already uses the joint names
    "hand": None,
}

# the robot of the pack: exp and run
ROBOT = {"g1": ("robot_g1", "left_arm"), "panda": ("robot_franka", "panda"),
         "hand": ("robot_inspire_hand", "hand_6dof")}

DESCRIPTION = {
    "g1": "The left arm of the Unitree G1 in Isaac Sim, smooth freeform motion (Robi, 2025-10, freeform_smooth).",
    "panda": "The Franka Panda of the DREAM dataset (panda-3cam_realsense), freeform motion (Robi, 2025-10, freeform_dream_ds).",
    "hand": "The Inspire hand in Isaac Sim, 6 actuated joints, freeform motion (Robi, 2025-10, freeform_Hand_6DOF).",
}

CAMERA = "dev000"


def convert_action(action, robot):
    """One step of Robi's _action.yaml as a joint-position action"""
    values = action["rc-position-target"]
    if FIELD_TO_JOINT[robot] is not None:
        values = {FIELD_TO_JOINT[robot][field]: value for field, value in values.items()}
    return {"joint-position": {joint: float(value) for joint, value in values.items()}}


def import_demo(source, target, robot):
    """Copy one demonstration directory, converting its yaml files"""
    with open(source / "_action.yaml") as f:
        actions = [convert_action(action, robot) for action in yaml.load(f, Loader=yaml.CSafeLoader)]
    frames = sorted(source.glob(f"*_{CAMERA}.jpg"))
    assert len(frames) == len(actions), f"{source}: {len(frames)} frames, {len(actions)} actions"
    target.mkdir(parents=True)
    for frame in frames:
        shutil.copy2(frame, target / frame.name)
    with open(target / "_action.yaml", "w") as f:
        yaml.dump(actions, f, Dumper=yaml.CSafeDumper)
    with open(target / "_annotation.yaml", "w") as f:
        yaml.dump([{"comment": "", "labels": [], "reward": 0.0} for _ in actions], f)
    with open(target / "_metadata.yaml", "w") as f:
        yaml.dump({"cameras": [CAMERA], "maxsteps": len(actions),
                   "stored_as_images": True, "stored_as_video": False}, f)
    return actions


def import_robi_pack(source, target, robot):
    """Import the pack in the directory source as the demopack in the directory target"""
    source, target = pathlib.Path(source), pathlib.Path(target)
    assert not target.exists(), f"{target} already exists"
    demos = sorted(d for d in source.iterdir() if d.is_dir())
    joint_ranges = {}
    for demo in demos:
        for action in import_demo(demo, target / demo.name, robot):
            for joint, value in action["joint-position"].items():
                low, high = joint_ranges.get(joint, (value, value))
                joint_ranges[joint] = (min(low, value), max(high, value))
    pack = {
        "name": target.name,
        "robot_exp": ROBOT[robot][0],
        "robot_run": ROBOT[robot][1],
        "cameras": {CAMERA: {"fps": 10, "width": 256, "height": 256}},
    }
    with open(target / f"{target.name}.yaml", "w") as f:
        f.write(f"# {DESCRIPTION[robot]}\n")
        yaml.dump(pack, f, sort_keys=False)
    with open(target / "dataset_info.yaml", "w") as f:
        yaml.dump({joint: {"min": low, "max": high} for joint, (low, high) in joint_ranges.items()}, f)
    with open(target / "README.md", "w") as f:
        f.write(f"# {target.name}\n\n{DESCRIPTION[robot]}\n\n"
                f"Imported from `{source}` by `src/demonstration/import_robi_packs.py` "
                f"({len(demos)} demonstrations, camera {CAMERA}). "
                f"Robot: `{ROBOT[robot][0]}/{ROBOT[robot][1]}`. Joint ranges: `dataset_info.yaml`.\n")
    shutil.make_archive(str(target), "zip", target.parent, target.name)
    print(f"Imported {len(demos)} demonstrations from {source} to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("robot", choices=sorted(FIELD_TO_JOINT))
    args = parser.parse_args()
    import_robi_pack(args.source, args.target, args.robot)
