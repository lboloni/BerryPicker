"""
robopepp_predict.py

The BerryPicker driver of RoboPEPP (Goswami et al., CVPR 2025;
github.com/raktimgg/RoboPEPP). Run by external_proprioception.run_external
with the Python of the RoboPEPP environment, NOT in the BerryPicker venv.

Loads a RoboPEPP checkpoint, predicts the joint angles of every frame of an
exported DREAM-format folder (NNNNNN.rgb.jpg, in order) and writes them to an
.npz file: angles (N x 6, radians), joint_names (panda_joint1..6), frames
(the original frame files). RoboPEPP predicts only joints 1-6 of the Panda.

The preprocessing is that of RoboPEPP's test dataset (datasets/dream_ssl.py),
reimplemented here because that module cannot be imported without
GroundingDINO: the precomputed GroundingDINO bounding box of the frame, extended
as for the RealSense sequence (image_proc.get_extended_bbox), clipped to the
image, crop, ResizeLongerSide(224), PadToSquare(224) with edge padding,
ImageNet normalization. The joint angles come from the encoder, the predictor
and the joint head of the model (RobotPoseNet.forward up to joint_net),
denormalized with the joint statistics of test.py; the keypoint heads and the
PnP of the full forward pass are not needed for them.

The bounding boxes are files {"bonuding_boxes": [xmin, ymin, xmax, ymax]}
named after the original frames (NNNNNN.rgb.json), made by RoboPEPP's
bbox_grounding_dino/create_bbox.py; an exported frame is a symbolic link to
its original frame, which gives the name.

Needs CUDA: RoboPEPP's JointNet and robot model are hard-wired to it.
"""

import argparse
import json
import os
import pathlib
import sys

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

# the joint statistics of RoboPEPP's test.py for the Panda
JOINT_MEAN = [-5.21536266e-02, 2.67705064e-01, 6.04037923e-03, -2.00518761e+00,
              1.49018339e-02, 1.98561130e+00]
JOINT_STD = [1.02490399e+00, 6.45399420e-01, 5.11071070e-01, 5.08277903e-01,
             7.68895279e-01, 5.10862160e-01]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 7)]
CROP_SIZE = 224
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480


def preprocess(frame, bbox_dir, sequence):
    """The input tensor of one frame, as RoboPEPP's test dataset makes it"""
    original = pathlib.Path(os.path.realpath(frame))
    with open(pathlib.Path(bbox_dir, original.name.replace(".jpg", ".json"))) as f:
        wmin, hmin, wmax, hmax = json.load(f)["bonuding_boxes"]
    # image_proc.get_extended_bbox, the case of the sequence
    assert "realsense" in sequence, f"only the extension of the RealSense sequence is implemented, not {sequence}"
    wmin, hmin, wmax, hmax = wmin - 150, hmin, wmax + 50, hmax
    bbox_min = np.clip([wmin, hmin], [0.0, 0.0], [IMAGE_WIDTH, IMAGE_HEIGHT])
    bbox_max = np.clip([wmax, hmax], [0.0, 0.0], [IMAGE_WIDTH, IMAGE_HEIGHT])
    img = np.array(Image.open(original).convert("RGB"))
    img = img[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]
    # augmentations.ResizeLongerSide
    h, w = img.shape[:2]
    if w > h:
        new_w, new_h = CROP_SIZE, int(CROP_SIZE * h / w)
    else:
        new_w, new_h = int(CROP_SIZE * w / h), CROP_SIZE
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # augmentations.PadToSquare
    pad_h, pad_w = (CROP_SIZE - new_h) // 2, (CROP_SIZE - new_w) // 2
    img = np.pad(img, ((pad_h, CROP_SIZE - new_h - pad_h), (pad_w, CROP_SIZE - new_w - pad_w), (0, 0)), mode="edge")
    img = torchvision.transforms.ToTensor()(Image.fromarray(img))
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code-dir", required=True, help="the RoboPEPP checkout")
    parser.add_argument("--checkpoint", required=True, help="e.g. .../checkpoints/robopepp_ssl_realsense.pt")
    parser.add_argument("--frames", required=True, help="the exported DREAM-format folder")
    parser.add_argument("--out", required=True, help="the .npz file to write")
    parser.add_argument("--jepa-path", required=True, help="the I-JEPA pre-trained encoder (jepa_joints-ep200.pth.tar)")
    parser.add_argument("--bbox-dir", required=True, help="the GroundingDINO bounding boxes of the sequence")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    code_dir = pathlib.Path(args.code_dir).expanduser()
    sys.path.insert(0, str(code_dir))
    from models.model import make_robotposenet

    device = torch.device(args.device)
    joint_mean = torch.Tensor(JOINT_MEAN).to(device)
    joint_std = torch.Tensor(JOINT_STD).to(device)
    # the settings of configs/panda.yaml
    model = make_robotposenet("vit_base", input_shape=(CROP_SIZE, CROP_SIZE), patch_size=16,
                              r_path=args.jepa_path, pred_emb_dim=384, pred_depth=12,
                              joint_mean=joint_mean, joint_std=joint_std,
                              urdf_file=str(code_dir / "urdfs" / "Panda" / "panda.urdf")).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    if "module_state_dict" in checkpoint:
        checkpoint = checkpoint["module_state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.items()})
    model.eval()

    frames = sorted(pathlib.Path(args.frames).glob("*.rgb.jpg"))
    sequence = pathlib.Path(args.frames).name
    angles = []
    with torch.no_grad():
        for start in range(0, len(frames), args.batch_size):
            x = torch.stack([preprocess(frame, args.bbox_dir, sequence)
                             for frame in frames[start:start + args.batch_size]]).to(device)
            # RobotPoseNet.forward, up to the joint angles
            img_feat = model.predictor_backbone(model.context_backbone(x), None, None)
            pred_joints = model.joint_net(img_feat.mean(dim=1))
            angles.append((pred_joints * model.joint_std + model.joint_mean).cpu().numpy())
            print(f"robopepp_predict: {min(start + args.batch_size, len(frames))}/{len(frames)} frames")
    np.savez(args.out, angles=np.concatenate(angles), joint_names=np.array(JOINT_NAMES),
             frames=np.array([os.path.realpath(frame) for frame in frames]))


if __name__ == "__main__":
    main()
