"""
hpe_predict.py

The BerryPicker driver of HPE / HoRoPose (Ban et al., ECCV 2024;
github.com/Oliverbansk/Holistic-Robot-Pose-Estimation). Run by
external_proprioception.run_external with the Python of the HPE environment,
NOT in the BerryPicker venv.

Loads an HPE model (an experiment folder of its model zoo: config.yaml and
ckpt/<model>.pk), predicts the joint angles of every frame of an exported
DREAM-format folder and writes them to an .npz file: angles (N x 8,
radians), joint_names (panda_joint1..7, panda_finger_joint1), frames (the
original frame files).

The model, the dataset and the inputs are those of HPE's scripts/test.py:
make_cfg reads the experiment's config.yaml, DreamDataset reads the exported
folder (it recognizes a real DREAM sequence by the name of the folder, e.g.
panda-3cam_realsense), and the depth scale k comes from the bounding box as
in test.py. As in HPE's own evaluation, the bounding box, and thus the image
crop, comes from the ground-truth keypoints of the frames' JSON files.

Run from the HPE checkout (its lib/config.py expects the data/ folder there,
with data/deps/panda-description and models/hrnet_w32-36af842e_roc.pth).
Needs CUDA: HPE's robot model and test inputs are hard-wired to it.
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", required=True, help="the HPE checkout")
    parser.add_argument("--checkpoint", required=True, help="the experiment folder of the model (config.yaml, ckpt/)")
    parser.add_argument("--frames", required=True, help="the exported DREAM-format folder")
    parser.add_argument("--out", required=True, help="the .npz file to write")
    parser.add_argument("--model-name", default="curr_best_auc(add)_model", help="the checkpoint in ckpt/, without .pk")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    code_dir = pathlib.Path(args.code_dir).expanduser()
    frames_dir = pathlib.Path(args.frames).resolve()
    out = pathlib.Path(args.out).resolve()
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "lib"))
    from lib.dataset.const import INITIAL_JOINT_ANGLE, JOINT_NAMES
    from lib.dataset.dream import DreamDataset
    from lib.models.full_net import get_rootNetwithRegInt_model
    from scripts.test import make_cfg

    test_args = argparse.Namespace(exp_path=str(pathlib.Path(args.checkpoint).expanduser()), known_joint=False,
                                   model_name=args.model_name, vis_skeleton=False, dataset=frames_dir.name)
    cfg = make_cfg(test_args)
    cfg.test_ds_names = str(frames_dir)
    device = torch.device("cuda")

    init_param_dict = {"robot_type": cfg.urdf_robot_name, "pose_params": INITIAL_JOINT_ANGLE,
                       "cam_params": np.eye(4, dtype=float), "init_pose_from_mean": True}
    model = get_rootNetwithRegInt_model(init_param_dict, cfg)
    ckpt = torch.load(pathlib.Path(cfg.exp_path, "ckpt", f"{cfg.model_name}.pk"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.float().to(device).eval()

    dataset = DreamDataset(cfg.test_ds_names,
                           rootnet_resize_hw=(cfg.rootnet_image_size, cfg.rootnet_image_size),
                           other_resize_hw=(cfg.other_image_size, cfg.other_image_size),
                           color_jitter=False, rgb_augmentation=False, occlusion_augmentation=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    scene_ids = []
    angles = []
    with torch.no_grad():
        for batch in loader:
            root_images = batch["root"]["images"].to(device).float() / 255.
            root_K = batch["root"]["K"].to(device).float()
            reg_images = batch["other"]["images"].to(device).float() / 255.
            other_K = batch["other"]["K"].to(device).float()
            # the bounding box and the depth scale k of scripts/test.py
            if cfg.use_origin_bbox:
                bboxes = batch["bbox_strict_bounded_original"].to(device).float()
                fx, fy = batch["K_original"][:, 0, 0].to(device), batch["K_original"][:, 1, 1].to(device)
            else:
                bboxes = batch["root"]["bbox_strict_bounded"].to(device).float()
                fx, fy = root_K[:, 0, 0], root_K[:, 1, 1]
            if cfg.use_extended_bbox:
                bboxes = batch["root"]["bbox_gt2d_extended"].to(device).float()
                fx, fy = root_K[:, 0, 0], root_K[:, 1, 1]
            area = torch.max(torch.abs(bboxes[:, 2] - bboxes[:, 0]), torch.abs(bboxes[:, 3] - bboxes[:, 1])) ** 2
            k_values = torch.sqrt(fx * fy * 1000.0 * 1000.0 / area).float()
            pred_pose = model(reg_images, root_images, k_values, K=other_K)[0]
            angles.append(pred_pose.cpu().numpy())
            scene_ids.extend(batch["scene_id"].numpy().tolist())
            print(f"hpe_predict: {len(scene_ids)}/{len(dataset)} frames")

    # the dataset does not keep the order of the frames; the scene_id is the frame number
    order = np.argsort(scene_ids)
    frames = [os.path.realpath(frames_dir / f"{scene_ids[i]:06d}.rgb.jpg") for i in order]
    np.savez(out, angles=np.concatenate(angles)[order],
             joint_names=np.array(JOINT_NAMES[cfg.urdf_robot_name]), frames=np.array(frames))


if __name__ == "__main__":
    main()
