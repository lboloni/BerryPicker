# Real-time AL5D Robot Segmentation (`_2026_11_robot_segmentation`)

Standalone experimental implementation of `PLAN-Robot-Segmentation.md` and `REVIEW-Robot-Segmentation.md`.
Has **no imports** from BerryPicker's collector, robot controllers, sensor-processing factory, transfer package, or `ExpRunFlow`; runs directly from a demopack path or video path using Python, OpenCV, PyTorch (and optional `ultralytics`, `sam2`, `sam3`).

## Runtime API

```python
from _experimental._2026_11_robot_segmentation import load_segmenter

seg = load_segmenter("/path/to/results/models/unet_motion_dev2/best", device="auto")
mask = seg.segment(frame_rgb)  # uint8 (H, W), 0=background, 1=robot
```

`segment(frame_rgb) -> mask` takes one RGB uint8 frame, preserves aspect ratio via letterboxing, maps the binary mask back to original camera coordinates, and works identically on the first frame or during stationary pauses.

## Label & Reference Mask Conventions

- **Offline teacher training labels** (single-channel uint8 PNG):
  - `0`: confident background (`BG`)
  - `1`: confident visible robot (`FG`)
  - `255`: ambiguous/unknown (`UNKNOWN`, excluded from partial-BCE loss; preserves jaw holes)
- **Independent evaluation masks**:
  - Either plain binary (`0`/`1` or `0`/`255`), or part-coded PNG (`0=bg`, `1=arm`, `2=base`, `3=gripper`, `4=jaw_gap`, `5=shadow`), or sibling masks `<stem>_{base,gripper,jaw_gap,shadow}.png`.

## Running the 7-Step Workflow via CLI

Run from `BerryPicker/src` (so `_experimental._2026_11_robot_segmentation` is on `PYTHONPATH`):

```bash
PACK=/Users/lboloni/Documents/Develop/Data/BerryPicker-Demopacks/random-both-cameras-video
OUT=/tmp/al5d_seg_run
PY=/Users/lboloni/Documents/Develop/VirtualEnvs/BerryPicker/bin/python

# Step 1: Inventory 10 episodes, freeze 6/2/2 split, extract disjoint val + sealed test frames
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli inventory \
  --pack-path "$PACK" --result-dir "$OUT" --extract-eval

# Step 2: Fit median-background baseline and static positional control (dev2)
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli baseline \
  --pack-path "$PACK" --result-dir "$OUT" --camera dev2 --split "$OUT/split.json"

# Step 3: Generate offline partial labels (motion, sam2, or sam3)
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli label \
  --pack-path "$PACK" --result-dir "$OUT" --teacher motion --camera dev2 --split "$OUT/split.json"

# Steps 4-5: Train single-frame student (unet, yolo26_semantic, or yolo26_instance)
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli train \
  --pack-path "$PACK" --result-dir "$OUT" --teacher motion \
  --arch yolo26_semantic --weights yolo26n-seg.yaml --camera dev2 --epochs 15

# Steps 4-7: Evaluate against independent reference masks (val first; test_sealed only once frozen)
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli evaluate \
  --artifact-dir "$OUT/models/yolo26_semantic_motion_dev2/best" \
  --eval-manifest "$OUT/eval_pack/eval_manifest.json" \
  --partition val --result-dir "$OUT/eval_val" --pack-path "$PACK"

# Steps 2 & 7: Paced 10 Hz latency & freshness benchmark (1 or 2 views)
PYTHONPATH=src $PY -m _experimental._2026_11_robot_segmentation.cli benchmark \
  --artifacts "$OUT/models/yolo26_semantic_motion_dev2/best" \
  --videos "$PACK/2025_03_08__14_29_07/video_dev2.mp4" \
  --duration-s 600 --hz 10 --out-json "$OUT/bench_10hz.json"
```

## Notebooks (BerryPicker style)

Two standalone notebooks live alongside the package (using the `berrypicker` kernel and a tagged `parameters` cell, with zero `ExpRunFlow` imports):

- `Explore_RobotSegmentation.ipynb` — Steps 1–3: episode inventory and frozen 6/2/2 split plot, median-background + MAD + static-control inspection, motion teacher vs. Meta SAM 3.1 (`sam3.1_multiplex.pt`) coverage, prompt/resolution checks, and overlays on moving vs. paused frames.
- `Verify_RobotSegmentation.ipynb` — Steps 4–7: trains or loads `SmallSegUNet` and `YOLO26SemanticNet` on both motion and Meta SAM 3.1 labels, compares them side-by-side against Meta SAM 3.1 (zero-shot ceiling) in strips, pause stability, and an embedded H.264 validation review video, and runs the paced 10 Hz latency & freshness benchmark.
- `Flow_SegmentedVP.ipynb` (with `segmented_demo.py`) — BerryPicker flow comparing white-background segmented visual proprioception (`<demopack>_seg_white`) against unsegmented (`orig`) visual proprioception across `Conv-VAE-Neo`, `ProprioTuned ViT`, and optional CNNs on held-out `vp_testing` and `vp_validation`.

Run them interactively in VS Code or via `papermill` into a results directory (so source notebooks stay output-free in git):

```bash
PY=/Users/lboloni/Documents/Develop/VirtualEnvs/BerryPicker/bin/python
OUT=/Users/lboloni/Documents/Develop/Data/BerryPicker-Flows/RobotSegmentation_pilot
mkdir -p "$OUT"
$PY -m papermill \
  src/_experimental/_2026_11_robot_segmentation/Explore_RobotSegmentation.ipynb \
  "$OUT/Explore_RobotSegmentation.ipynb" \
  -p result_dir "$OUT" -k berrypicker
$PY -m papermill \
  src/_experimental/_2026_11_robot_segmentation/Verify_RobotSegmentation.ipynb \
  "$OUT/Verify_RobotSegmentation.ipynb" \
  -p result_dir "$OUT" -k berrypicker
```

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/lboloni/Documents/Develop/VirtualEnvs/BerryPicker/bin/python -m pytest \
  src/test/_experimental/_2026_11_robot_segmentation
```
