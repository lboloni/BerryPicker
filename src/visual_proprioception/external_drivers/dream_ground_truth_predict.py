"""
dream_ground_truth_predict.py

A reference "method" for the access to external pose estimators: it predicts
the ground-truth joint angles recorded in the JSON files of a DREAM-format
folder (sim_state.joints). It needs no model and no GPU, and runs with any
Python that has numpy, so it checks the whole path of an external run
(export, driver, predictions, predict_positions, Verify, Compare) on any
machine; its predictions must have zero error.

Writes the same .npz as the other drivers: angles (N x J, radians),
joint_names, frames (the original frame files).
"""

import argparse
import json
import os
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # the arguments every driver gets; this one has no code and no checkpoint
    parser.add_argument("--code-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--frames", required=True, help="the exported DREAM-format folder")
    parser.add_argument("--out", required=True, help="the .npz file to write")
    args = parser.parse_args()

    frames = sorted(pathlib.Path(args.frames).glob("*.rgb.jpg"))
    names = None
    angles = []
    for frame in frames:
        with open(str(frame).replace(".rgb.jpg", ".json")) as f:
            joints = json.load(f)["sim_state"]["joints"]
        if names is None:
            names = [joint["name"].split("/")[-1] for joint in joints]
        angles.append([joint["position"] for joint in joints])
    np.savez(args.out, angles=np.array(angles), joint_names=np.array(names),
             frames=np.array([os.path.realpath(frame) for frame in frames]))
    print(f"dream_ground_truth_predict: {len(frames)} frames")


if __name__ == "__main__":
    main()
