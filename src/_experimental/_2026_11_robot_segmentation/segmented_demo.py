"""Helpers for building white-background segmented BerryPicker demonstrations.

This module bridges the standalone robot segmentation package
(``_experimental._2026_11_robot_segmentation``) with BerryPicker's
``Demonstration`` / ``ExpRunFlow`` flow notebooks without modifying any core
BerryPicker package.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from .data import read_video_frames


def apply_white_background(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image with background pixels (``mask == 0``) set to white."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB frame, got {frame_rgb.shape}")
    if mask.shape != frame_rgb.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} != frame shape {frame_rgb.shape[:2]}")
    fg = mask > 0
    return np.where(fg[..., None], frame_rgb, 255).astype(np.uint8)


def segment_episode_video(
    src_demo_dir: Path,
    dst_demo_dir: Path,
    camera: str,
    segmenter: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Segment one episode video onto a white background and copy metadata verbatim.

    Copies ``_action.yaml``, ``_annotation.yaml``, and ``_metadata.yaml`` so
    ground-truth 6-DoF targets and frame counts match the original episode.
    """
    src_demo_dir = Path(src_demo_dir)
    dst_demo_dir = Path(dst_demo_dir)
    dst_demo_dir.mkdir(parents=True, exist_ok=True)

    for meta_name in ("_action.yaml", "_annotation.yaml", "_metadata.yaml"):
        src_meta = src_demo_dir / meta_name
        if not src_meta.is_file():
            raise FileNotFoundError(src_meta)
        dst_meta = dst_demo_dir / meta_name
        if overwrite or not dst_meta.is_file():
            shutil.copy2(src_meta, dst_meta)

    with open(src_demo_dir / "_metadata.yaml", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    expected_steps = int(meta["maxsteps"])

    src_video = src_demo_dir / f"video_{camera}.mp4"
    dst_video = dst_demo_dir / f"video_{camera}.mp4"
    if not src_video.is_file():
        raise FileNotFoundError(src_video)

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {src_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if dst_video.is_file() and not overwrite:
        _, raw_frames = read_video_frames(src_video)
        _, seg_frames = read_video_frames(dst_video)
        if len(seg_frames) != expected_steps:
            raise RuntimeError(
                f"Cached segmented video {dst_video} has {len(seg_frames)} frames "
                f"(expected {expected_steps}); re-run with overwrite=True"
            )
        mid = len(raw_frames) // 2
        fg_frac = float((seg_frames[mid].min(axis=2) < 250).mean())
        return {
            "episode": src_demo_dir.name,
            "frames": expected_steps,
            "mean_fg_frac": fg_frac,
            "cached": True,
            "sample_raw": raw_frames[mid],
            "sample_white": seg_frames[mid],
        }

    _, raw_frames = read_video_frames(src_video)
    if len(raw_frames) < expected_steps:
        raise RuntimeError(
            f"Source video {src_video} has {len(raw_frames)} frames < maxsteps {expected_steps}"
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_video = dst_video.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {tmp_video}")

    fg_fracs: list[float] = []
    sample_idx = expected_steps // 2
    sample_raw = None
    sample_white = None
    try:
        for idx in range(expected_steps):
            rgb = raw_frames[idx]
            mask = segmenter.segment(rgb)
            white_rgb = apply_white_background(rgb, mask)
            fg_fracs.append(float((mask > 0).mean()))
            if idx == sample_idx:
                sample_raw = rgb
                sample_white = white_rgb
            writer.write(cv2.cvtColor(white_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    tmp_video.replace(dst_video)

    # Verify the written video is readable and preserves frame count
    check_cap = cv2.VideoCapture(str(dst_video))
    try:
        if not check_cap.isOpened():
            raise RuntimeError(f"Written video cannot be opened: {dst_video}")
        written_count = int(check_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if written_count != expected_steps:
            raise RuntimeError(
                f"Written video {dst_video} has {written_count} frames, expected {expected_steps}"
            )
    finally:
        check_cap.release()

    return {
        "episode": src_demo_dir.name,
        "frames": expected_steps,
        "mean_fg_frac": float(np.mean(fg_fracs)),
        "cached": False,
        "sample_raw": sample_raw,
        "sample_white": sample_white,
    }


def save_white_bg_preview(
    episode_stats: Sequence[dict[str, Any]],
    out_path: Path,
    max_episodes: int = 6,
) -> Path:
    """Save a contact sheet of (raw | white-background) pairs for visual verification."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = list(episode_stats[:max_episodes])
    if not chosen:
        raise ValueError("No episode stats provided for preview")

    rows: list[np.ndarray] = []
    for st in chosen:
        raw = st["sample_raw"]
        white = st["sample_white"]
        h, w = raw.shape[:2]
        banner = np.full((24, w * 2, 3), 240, dtype=np.uint8)
        label = f"{st['episode']} | raw vs seg-white (fg={st['mean_fg_frac']:.2%})"
        cv2.putText(
            banner,
            label,
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        pair = np.concatenate([raw, white], axis=1)
        rows.append(np.concatenate([banner, pair], axis=0))

    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def build_segmented_demopack(
    src_yaml_path: Path,
    dst_yaml_path: Path,
    src_root: Path,
    dst_root: Path,
    dst_demo_run: str,
    episodes: Sequence[str],
    camera: str,
    segmenter: Any,
    *,
    preview_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a sibling white-background demonstration run from explicit paths.

    Accepts explicit YAML and data directory paths so this module keeps zero
    imports from BerryPicker core packages or ``ExpRunFlow``.
    """
    src_yaml_path = Path(src_yaml_path)
    dst_yaml_path = Path(dst_yaml_path)
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_yaml_path.is_file():
        raise FileNotFoundError(src_yaml_path)

    with open(src_yaml_path, encoding="utf-8") as fh:
        demo_cfg = yaml.safe_load(fh) or {}
    src_demo_run = demo_cfg.get("name", src_yaml_path.stem)
    demo_cfg["name"] = dst_demo_run
    demo_cfg["segmented_source"] = src_demo_run
    demo_cfg["segmented_background"] = "white"
    dst_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(demo_cfg, fh)

    dst_root.mkdir(parents=True, exist_ok=True)

    stats: list[dict[str, Any]] = []
    for ep in episodes:
        st = segment_episode_video(
            src_root / ep,
            dst_root / ep,
            camera=camera,
            segmenter=segmenter,
            overwrite=overwrite,
        )
        stats.append(st)

    saved_preview = None
    if preview_path is not None and stats:
        saved_preview = save_white_bg_preview(stats, Path(preview_path))

    return {
        "src_demo_run": src_demo_run,
        "dst_demo_run": dst_demo_run,
        "dst_root": str(dst_root),
        "camera": camera,
        "episodes": len(stats),
        "total_frames": int(sum(s["frames"] for s in stats)),
        "mean_fg_frac": float(np.mean([s["mean_fg_frac"] for s in stats])) if stats else 0.0,
        "preview_path": str(saved_preview) if saved_preview is not None else None,
        "per_episode": [
            {k: v for k, v in s.items() if k not in ("sample_raw", "sample_white")}
            for s in stats
        ],
    }


def summarize_comparison_csv(csv_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Parse ``Compare_VisualProprioception`` CSV and pair ``*_orig`` vs ``*_seg`` runs."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"Comparison CSV has no metric rows: {csv_path}")

    header = rows[0]
    run_names = header[1:]
    per_run_dofs: dict[str, dict[str, float]] = {name: {} for name in run_names}
    for row in rows[1:]:
        dof = row[0]
        for name, val in zip(run_names, row[1:]):
            per_run_dofs[name][dof] = float(val)

    paired: list[dict[str, Any]] = []
    for name in run_names:
        if not name.endswith("_orig"):
            continue
        base = name[: -len("_orig")]
        seg_name = f"{base}_seg"
        if seg_name not in per_run_dofs:
            continue
        orig_mean = float(np.mean(list(per_run_dofs[name].values())))
        seg_mean = float(np.mean(list(per_run_dofs[seg_name].values())))
        delta = seg_mean - orig_mean
        rel_pct = (100.0 * delta / orig_mean) if orig_mean > 0 else 0.0
        paired.append(
            {
                "model": base,
                "orig_mean_rmse": orig_mean,
                "seg_mean_rmse": seg_mean,
                "delta": delta,
                "rel_pct": rel_pct,
                "orig_dofs": per_run_dofs[name],
                "seg_dofs": per_run_dofs[seg_name],
            }
        )

    lines = [
        "| Model | Orig Mean RMSE | Seg-White Mean RMSE | Delta (Seg - Orig) | Rel % |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in paired:
        lines.append(
            f"| {item['model']} | {item['orig_mean_rmse']:.4f} | "
            f"{item['seg_mean_rmse']:.4f} | {item['delta']:+.4f} | "
            f"{item['rel_pct']:+.1f}% |"
        )
    return paired, "\n".join(lines)
