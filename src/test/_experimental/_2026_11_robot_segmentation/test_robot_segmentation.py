"""Tests for standalone real-time robot segmentation experiment."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

# Ensure BerryPicker/src is importable when running pytest from repo root or test dir
_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from _experimental._2026_11_robot_segmentation.baselines import (
    MedianBackgroundModel,
    StaticMaskControl,
)
from _experimental._2026_11_robot_segmentation.benchmark import run_paced_benchmark
from _experimental._2026_11_robot_segmentation.data import (
    BG,
    FG,
    UNKNOWN,
    inventory_demopack,
    make_episode_split,
    read_video_frames,
    sample_pose_balanced_indices,
    save_label_png,
    save_rgb_png,
)
from _experimental._2026_11_robot_segmentation.evaluate import (
    PART_ARM,
    PART_BASE,
    PART_GRIPPER,
    PART_JAW_GAP,
    PART_SHADOW,
    annotation_repeatability,
    boundary_f1,
    evaluate_predictor,
    extract_annotation_pack,
    foreground_iou,
    jaw_gap_metrics,
    part_recall,
    pause_stability,
    shadow_leakage,
)
from _experimental._2026_11_robot_segmentation.models import (
    RobotSegmenter,
    SmallSegUNet,
    YOLO26SemanticNet,
    audit_polygon_roundtrip,
    letterbox_rgb,
    load_segmenter,
    save_semantic_checkpoint,
    unletterbox_mask,
)
from _experimental._2026_11_robot_segmentation.teachers import (
    MotionLabelTeacher,
    SAM3Teacher,
    TeacherEpisodeResult,
    apply_occlusion_safe_pcb_inclusion,
    apply_static_consensus_gating,
    save_teacher_outputs,
    select_consistent_hypothesis,
    select_motion_anchors,
)
from _experimental._2026_11_robot_segmentation.train import (
    partial_balanced_bce_loss,
    train_semantic_student,
)
from _experimental._2026_11_robot_segmentation.viz import (
    make_comparison_strip,
    write_review_video,
)


def _make_synthetic_episode(video_path: Path, n_frames: int = 24, size: int = 96, seed: int = 0) -> None:
    """Write a synthetic fixed-camera video with a static base, moving link + open jaws, and a pause."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Textured neutral background
    bg = np.full((size, size, 3), 185, dtype=np.uint8)
    bg[:, :, 0] = 175
    bg[:, :, 2] = 165
    noise = rng.integers(-6, 7, size=(size, size, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (size, size))
    assert writer.isOpened()
    for t in range(n_frames):
        frame = bg.copy()
        # Static base (never moves)
        cv2.rectangle(frame, (38, 72), (58, 92), (35, 35, 45), -1)
        # Moving link + jaws, paused in frames 10..15
        phase = 10 if 10 <= t <= 15 else t
        tip_x = int(20 + (phase / max(1, n_frames - 1)) * 55)
        tip_y = int(24 + 12 * np.sin(phase * 0.35))
        cv2.line(frame, (48, 74), (tip_x, tip_y), (40, 45, 55), 6)
        # Two thin jaws with a visible gap
        cv2.line(frame, (tip_x, tip_y), (tip_x - 4, tip_y - 10), (25, 30, 35), 2)
        cv2.line(frame, (tip_x, tip_y), (tip_x + 4, tip_y - 10), (25, 30, 35), 2)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _make_synthetic_pack(tmp_path: Path, n_episodes: int = 10) -> Path:
    pack = tmp_path / "synthetic-pack"
    for i in range(n_episodes):
        ep = pack / f"2025_03_08__14_{i:02d}_00"
        _make_synthetic_episode(ep / "video_dev2.mp4", n_frames=20, size=96, seed=i)
        _make_synthetic_episode(ep / "video_dev3.mp4", n_frames=20, size=96, seed=100 + i)
    return pack


def test_standalone_no_berrypicker_core_imports():
    """Verify no module in _2026_11_robot_segmentation imports BerryPicker core or ExpRunFlow."""
    forbidden = {
        "exp_run_config",
        "exprunflow",
        "flow",
        "demonstration",
        "robot",
        "robot_controller",
        "sensorprocessing",
        "visual_proprioception",
        "behavior_cloning",
    }
    pkg_dir = _SRC_DIR / "_experimental" / "_2026_11_robot_segmentation"
    for py_file in pkg_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".")[0] not in forbidden, f"{py_file.name} imports {node.module}"


def test_split_inventory_and_baselines(tmp_path: Path):
    pack = _make_synthetic_pack(tmp_path, n_episodes=10)
    split = make_episode_split(pack, camera="dev2")
    assert len(split.train) == 6
    assert len(split.val) == 2
    assert len(split.test) == 2
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)

    inv = inventory_demopack(pack, cameras=("dev2", "dev3"), eval_frames_per_episode=4, seed=0)
    assert len(inv["episodes"]) == 10
    assert set(inv["eval_candidates"]["val"].keys()) == set(split.val)
    assert set(inv["eval_candidates"]["test"].keys()) == set(split.test)

    eval_root = extract_annotation_pack(inv, tmp_path / "eval_pack", camera="dev2")
    manifest = json.loads((eval_root / "eval_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["val"]) == 8
    assert len(manifest["test_sealed"]) == 8

    bg = MedianBackgroundModel.fit_from_demopack(pack, split.train, camera="dev2", frame_step=2, threshold=0.12)
    bg.save(tmp_path / "bg")
    bg2 = load_segmenter(tmp_path / "bg")
    _, frames = read_video_frames(pack / split.val[0] / "video_dev2.mp4")
    mask = bg2.segment(frames[0])
    assert mask.shape == frames[0].shape[:2]
    assert mask.dtype == np.uint8
    assert mask.sum() > 0

    static = StaticMaskControl.fit_from_masks([bg2.segment(f) for f in frames], threshold=0.5, camera_id="dev2")
    static.save(tmp_path / "static")
    static2 = load_segmenter(tmp_path / "static")
    assert static2.segment(frames[0]).shape == frames[0].shape[:2]


def test_motion_teacher_partial_labels_and_anchors(tmp_path: Path):
    pack = _make_synthetic_pack(tmp_path, n_episodes=10)
    split = make_episode_split(pack, camera="dev2")
    bg = MedianBackgroundModel.fit_from_demopack(pack, split.train, camera="dev2", frame_step=2, threshold=0.10)
    teacher = MotionLabelTeacher(bg, offsets=(1, 2, 4), flow_min_px=0.3, min_fg_pixels=15)

    ids, frames = read_video_frames(pack / split.train[0] / "video_dev2.mp4")
    res = teacher.label_episode(frames, ids, episode=split.train[0], camera="dev2")
    assert res.labels.shape == frames.shape[:3]
    uniq = set(np.unique(res.labels).tolist())
    assert uniq.issubset({BG, FG, UNKNOWN})
    assert FG in uniq and BG in uniq and UNKNOWN in uniq
    assert res.retained.any()

    anchors = select_motion_anchors(res.labels, max_anchors=2, min_fg_pixels=15, seed=0)
    assert 1 <= len(anchors) <= 2
    assert all(len(a["points"]) == len(a["labels"]) and len(a["box"]) == 4 for a in anchors)

    summary = save_teacher_outputs(res, frames, tmp_path / "teachers", save_overlays_every=10)
    assert summary["total_frames"] == len(ids)
    assert (tmp_path / "teachers" / "motion" / "dev2" / split.train[0] / "raw_masks" / "frame_000000.png").is_file()
    assert (tmp_path / "teachers" / "motion" / "dev2" / split.train[0] / "training_labels" / "frame_000000.png").is_file()


def test_partial_bce_ignores_unknown_pixels():
    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    labels = torch.full((1, 1, 8, 8), UNKNOWN, dtype=torch.long)
    labels[0, 0, :2, :2] = FG
    labels[0, 0, 6:, 6:] = BG
    loss = partial_balanced_bce_loss(logits, labels)
    loss.backward()
    grad = logits.grad[0, 0]
    assert (grad[labels[0, 0] == UNKNOWN] == 0).all()
    assert (grad[labels[0, 0] == FG] < 0).all()
    assert (grad[labels[0, 0] == BG] > 0).all()


def test_letterbox_and_models_roundtrip(tmp_path: Path):
    # Aspect-ratio preserving round-trip on non-square image
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    canvas, meta = letterbox_rgb(rgb, 128)
    assert canvas.shape == (128, 128, 3)
    restored = unletterbox_mask(np.ones((128, 128), dtype=np.uint8), meta)
    assert restored.shape == (240, 320)

    # Test both SmallSegUNet and YOLO26SemanticNet training + save/load + segment()
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(6)]
    lbls = []
    for _ in range(6):
        m = np.full((64, 64), BG, dtype=np.uint8)
        m[20:44, 20:44] = FG
        m[18:20, 18:46] = UNKNOWN
        lbls.append(m)

    for arch, kwargs in [
        ("unet", {"base_channels": 8}),
        ("yolo26_semantic", {"weights_or_cfg": "yolo26n-seg.yaml", "decoder_channels": 16}),
    ]:
        out_dir = tmp_path / arch
        summary = train_semantic_student(
            imgs[:4],
            lbls[:4],
            imgs[4:],
            lbls[4:],
            out_dir=out_dir,
            arch=arch,
            arch_kwargs=kwargs,
            input_size=64,
            epochs=2,
            batch_size=2,
            device="cpu",
            seed=0,
        )
        assert len(summary["history"]) == 2
        seg = load_segmenter(out_dir / "best", device="cpu")
        pred = seg.segment(rgb)
        assert pred.shape == (240, 320)
        assert pred.dtype == np.uint8
        assert set(np.unique(pred).tolist()).issubset({0, 1})


def test_polygon_roundtrip_audit_detects_hole_fill():
    # Ring mask with an interior hole (models closed/narrow jaw loop)
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (54, 54), 1, -1)
    cv2.rectangle(mask, (24, 24), (40, 40), 0, -1)
    audit = audit_polygon_roundtrip([mask], approx_eps_px=1.0)
    assert audit["frames_with_filled_holes"] == 1
    assert audit["max_filled_hole_px"] > 100


def test_evaluation_metrics_and_full_evaluate(tmp_path: Path):
    # Construct part-coded reference mask: base, arm, gripper, open jaw gap, shadow
    part = np.zeros((64, 64), dtype=np.uint8)
    part[48:60, 26:38] = PART_BASE
    part[24:48, 30:35] = PART_ARM
    part[12:24, 26:29] = PART_GRIPPER
    part[12:24, 36:39] = PART_GRIPPER
    part[12:24, 29:36] = PART_JAW_GAP  # 7 px wide gap
    part[40:55, 45:58] = PART_SHADOW

    gt_robot = np.isin(part, [PART_ARM, PART_BASE, PART_GRIPPER]).astype(np.uint8)
    pred_good = gt_robot.copy()
    pred_blob = gt_robot.copy()
    pred_blob[part == PART_JAW_GAP] = 1  # fills the jaw gap
    pred_blob[part == PART_SHADOW] = 1  # leaks onto shadow

    assert foreground_iou(pred_good, gt_robot) == pytest.approx(1.0)
    assert boundary_f1(pred_good, gt_robot, tolerance_px=1) == pytest.approx(1.0)
    assert part_recall(pred_good, part == PART_BASE) == pytest.approx(1.0)
    assert part_recall(pred_good, part == PART_GRIPPER) == pytest.approx(1.0)
    assert shadow_leakage(pred_good, part == PART_SHADOW) == pytest.approx(0.0)
    assert shadow_leakage(pred_blob, part == PART_SHADOW) == pytest.approx(1.0)

    gap_good = jaw_gap_metrics(pred_good, part == PART_JAW_GAP, input_size=64)
    assert gap_good["resolvable_at_input_size"] is True
    assert gap_good["gap_kept_open"] is True
    assert gap_good["false_fg_in_gap"] == pytest.approx(0.0)

    gap_blob = jaw_gap_metrics(pred_blob, part == PART_JAW_GAP, input_size=64)
    assert gap_blob["gap_kept_open"] is False
    assert gap_blob["false_fg_in_gap"] == pytest.approx(1.0)

    # At very low resolution the gap should be marked unresolved
    gap_tiny = jaw_gap_metrics(pred_good, part == PART_JAW_GAP, input_size=6)
    assert gap_tiny["resolvable_at_input_size"] is False

    # Repeatability & pause stability
    rep = annotation_repeatability([gt_robot], [gt_robot])
    assert rep["mean_iou"] == pytest.approx(1.0)
    pstab = pause_stability([gt_robot, gt_robot, gt_robot], np.array([True, True, True]))
    assert pstab["mean_pause_flicker_rate"] == pytest.approx(0.0)
    assert pstab["max_pause_area_drop_frac"] == pytest.approx(0.0)

    # End-to-end evaluate_predictor
    eval_dir = tmp_path / "eval"
    save_rgb_png(eval_dir / "val/dev2/ep0/frame_000000.png", np.full((64, 64, 3), 120, dtype=np.uint8))
    save_label_png(eval_dir / "val/dev2/ep0/frame_000000_mask.png", part)
    manifest = {
        "pack_path": str(tmp_path),
        "camera": "dev2",
        "val": [
            {
                "episode": "ep0",
                "camera": "dev2",
                "frame_idx": 0,
                "image_rel": "val/dev2/ep0/frame_000000.png",
                "mask_rel": "val/dev2/ep0/frame_000000_mask.png",
                "is_pause": False,
            }
        ],
        "test_sealed": [],
    }
    manifest_path = eval_dir / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class _Oracle:
        def segment(self, frame_rgb):
            return pred_good

    summary = evaluate_predictor(_Oracle(), manifest_path, "val", tmp_path / "eval_out", input_size=64)
    assert summary["overall"]["mean_iou"] == pytest.approx(1.0)
    assert summary["overall"]["jaw_gap_open_rate"] == pytest.approx(1.0)
    assert (tmp_path / "eval_out" / "per_frame.csv").is_file()
    assert (tmp_path / "eval_out" / "report.md").is_file()


def test_paced_10hz_benchmark(tmp_path: Path):
    v1 = tmp_path / "cam1.mp4"
    v2 = tmp_path / "cam2.mp4"
    _make_synthetic_episode(v1, n_frames=12, size=64, seed=1)
    _make_synthetic_episode(v2, n_frames=12, size=64, seed=2)

    seg = RobotSegmenter(SmallSegUNet(base_channels=8), input_size=64, device="cpu")
    report = run_paced_benchmark(
        predictors=[seg, seg],
        video_paths=[v1, v2],
        duration_s=0.6,
        target_hz=10.0,
        warmup_ticks=2,
        out_path=tmp_path / "bench.json",
    )
    assert report["camera_count"] == 2
    assert report["completed_ticks"] >= 1
    assert report["cold_start_ms"] > 0
    assert (tmp_path / "bench.json").is_file()
    assert "observed_10hz_pass" in report


def test_viz_comparison_strip_and_review_video(tmp_path: Path):
    v1 = tmp_path / "cam1.mp4"
    _make_synthetic_episode(v1, n_frames=6, size=64, seed=3)
    _, frames = read_video_frames(v1)
    seg = RobotSegmenter(SmallSegUNet(base_channels=8), input_size=64, device="cpu")
    strip = make_comparison_strip(frames[0], {"UNet": seg.segment(frames[0])}, badge="f0 MOVE")
    assert strip.shape[0] == 64 + 24
    assert strip.shape[1] == 64 * 2
    out_mp4 = write_review_video(frames, {"UNet": seg}, tmp_path / "review.mp4", fps=10.0)
    assert out_mp4.is_file() and out_mp4.stat().st_size > 0


def test_sam3_teacher_motion_crosscheck(tmp_path: Path):
    """Verify SAM3Teacher retains static/matching masks and rejects identity jumps."""
    v1 = tmp_path / "cam1.mp4"
    _make_synthetic_episode(v1, n_frames=16, size=64, seed=7)
    ids, frames = read_video_frames(v1)
    bg = MedianBackgroundModel.fit_from_frames(frames, threshold=0.10)
    mres = MotionLabelTeacher(bg, offsets=(1, 2), flow_min_px=0.3, min_fg_pixels=10).label_episode(
        frames, ids, episode="ep0", camera="dev2"
    )

    class _FakeSAM3:
        def __init__(self):
            self.call = 0

        def segment(self, frame_rgb):
            t = self.call
            self.call += 1
            m = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            if t == 2:
                # Simulate an identity jump onto the top-left corner away from motion
                m[:12, :12] = 1
            else:
                # Cover robot area (base + arm)
                m[10:62, 12:56] = 1
            return m

    teacher = SAM3Teacher(prompt="robot arm", min_motion_overlap=0.1, min_fg_pixels=20, segmenter=_FakeSAM3())
    res = teacher.label_episode(frames, ids, mres, episode="ep0", camera="dev2")
    assert res.teacher_name == "sam3"
    assert res.retained.sum() >= 10
    # Frame 2 jumped away from moving robot -> marked UNKNOWN and not retained
    if (mres.labels[2] == FG).any() and (not res.pause_flags[2]):
        assert not res.retained[2]
        assert (res.labels[2] == UNKNOWN).all()


def test_sam3_stabilization_stages_a_b_c():
    """Verify Stage A occlusion-safe PCB inclusion, Stage B consensus gating, and Stage C selection."""
    bg_img = np.full((64, 64, 3), 180, dtype=np.uint8)
    bg = MedianBackgroundModel.fit_from_frames(np.stack([bg_img, bg_img], axis=0), threshold=0.10)

    # Static PCB on the right; raw SAM 3 mask covers arm+base but missed the PCB
    pcb = np.zeros((64, 64), dtype=np.uint8)
    pcb[20:40, 44:60] = 1
    raw_arm_only = np.zeros((64, 64), dtype=np.uint8)
    raw_arm_only[20:40, 20:44] = 1

    # Stage A: non-robot occluder covers top half of PCB (diff >= threshold) -> NOT forced to FG;
    # unoccluded bottom half of PCB matches median bg -> included as FG (1).
    frame_occ = bg_img.copy()
    frame_occ[20:30, 44:60] = 30  # foreign hand/object in front of top half of PCB
    included = apply_occlusion_safe_pcb_inclusion(raw_arm_only, frame_occ, bg, pcb, bg_match_threshold=0.10)
    assert (included[20:30, 44:60] == 0).all()  # occluded PCB pixels not forced FG
    assert (included[30:40, 44:60] == 1).all()  # unoccluded static PCB included as FG
    assert (included[20:40, 20:44] == 1).all()  # rest of robot preserved

    # Stage C: nested hypotheses (whole arm+base+PCB vs arm-only) -> prefer whole-assembly mask
    m_whole = np.zeros((64, 64), dtype=np.uint8)
    m_whole[20:40, 20:60] = 1
    chosen = select_consistent_hypothesis(
        [(m_whole, 0.43), (raw_arm_only, 0.45)], prev_mask=m_whole, static_pcb=pcb
    )
    assert np.array_equal(chosen, m_whole)

    # Stage B: static consensus gating across 10 frames:
    # - base [40:55, 20:35]: FG on all 10 frames (freq=1.0 -> FG)
    # - pcb [20:40, 44:60]: confirmed static PCB -> forced FG even if raw freq was lower
    # - flickering wire rim [30:40, 35:44]: FG on 5/10 frames (freq=0.5 -> UNKNOWN)
    # - bg [0:15, 0:15]: FG on 0/10 frames (freq=0.0 -> BG)
    n = 10
    frames = np.stack([bg_img.copy() for _ in range(n)], axis=0)
    labels = np.full((n, 64, 64), BG, dtype=np.uint8)
    labels[:, 40:55, 20:35] = FG
    labels[:5, 30:40, 35:44] = FG
    res = TeacherEpisodeResult(
        episode="ep0",
        camera="dev2",
        teacher_name="sam3",
        frame_indices=list(range(n)),
        raw_masks=(labels == FG).astype(np.uint8),
        labels=labels,
        retained=np.ones(n, dtype=bool),
        pause_flags=np.zeros(n, dtype=bool),
        metadata={},
    )
    gated, freq = apply_static_consensus_gating(
        {"ep0": res}, {"ep0": frames}, bg, low_freq=0.15, high_freq=0.85, pcb_mask=pcb
    )
    g0 = gated["ep0"].labels[0]
    assert (g0[40:55, 20:35] == FG).all()
    assert (g0[20:40, 44:60] == FG).all()
    assert (g0[30:40, 35:44] == UNKNOWN).all()
    assert (g0[0:15, 0:15] == BG).all()
    assert freq[35, 38] == pytest.approx(0.5)
