"""Visualization and review-video helpers for robot segmentation notebooks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .data import BG, FG, UNKNOWN, render_mask_overlay


def _put_banner(rgb: np.ndarray, title: str, badge: str | None = None) -> np.ndarray:
    """Attach a crisp dark header strip above an RGB image."""
    h, w = rgb.shape[:2]
    bar_h = 24
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h] = (28, 30, 34)
    canvas[bar_h:] = rgb
    cv2.putText(
        canvas,
        title,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if badge:
        color = (240, 175, 40) if "PAUSE" in badge else (90, 220, 110)
        cv2.putText(
            canvas,
            badge,
            (max(6, w - 72), 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_comparison_strip(
    rgb: np.ndarray,
    labeled_masks: Mapping[str, np.ndarray],
    badge: str | None = None,
) -> np.ndarray:
    """Build a horizontal RGB strip: original frame + one overlay panel per method."""
    panels = [_put_banner(rgb, "RGB", badge=badge)]
    for name, mask in labeled_masks.items():
        ovl = render_mask_overlay(rgb, mask)
        panels.append(_put_banner(ovl, name))
    return np.concatenate(panels, axis=1)


def make_zoom_crop(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    box_frac: tuple[float, float, float, float] = (0.15, 0.15, 0.85, 0.85),
    out_size: int = 256,
) -> np.ndarray:
    """Crop a fractional region ``(x0, y0, x1, y1)`` and upscale with nearest/linear for inspection."""
    h, w = rgb.shape[:2]
    x0 = max(0, int(round(box_frac[0] * w)))
    y0 = max(0, int(round(box_frac[1] * h)))
    x1 = min(w, int(round(box_frac[2] * w)))
    y1 = min(h, int(round(box_frac[3] * h)))
    img = render_mask_overlay(rgb, mask) if mask is not None else rgb
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def write_review_video(
    frames_rgb: np.ndarray,
    predictors: Mapping[str, Any],
    out_mp4: str | Path,
    fps: float = 10.0,
    extra_masks: Mapping[str, np.ndarray] | None = None,
    pause_flags: Sequence[bool] | None = None,
    max_frames: int | None = None,
) -> Path:
    """Render a side-by-side comparison MP4 (re-encoded to H.264 when ffmpeg is available)."""
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(frames_rgb) if max_frames is None else min(len(frames_rgb), int(max_frames))

    # Build first frame to determine output video dimensions
    def _render_frame(t: int) -> np.ndarray:
        rgb = frames_rgb[t]
        masks: dict[str, np.ndarray] = {}
        if extra_masks:
            for name, seq in extra_masks.items():
                masks[name] = seq[t]
        for name, pred in predictors.items():
            masks[name] = pred.segment(rgb)
        badge = None
        if pause_flags is not None:
            badge = f"f{t} {'PAUSE' if pause_flags[t] else 'MOVE'}"
        else:
            badge = f"f{t}"
        return make_comparison_strip(rgb, masks, badge=badge)

    first = _render_frame(0)
    vh, vw = first.shape[:2]
    raw_mp4 = out.with_suffix(".raw.mp4") if shutil.which("ffmpeg") else out
    writer = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (vw, vh))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {raw_mp4}")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for t in range(1, n):
            strip = _render_frame(t)
            writer.write(cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    if raw_mp4 != out:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(raw_mp4),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )
        raw_mp4.unlink(missing_ok=True)
    return out


def plot_episode_inventory(inventory: dict, camera: str = "dev2") -> plt.Figure:
    """Plot moving vs. paused frame counts across the 6/2/2 split."""
    split = inventory["split"]
    eps = split["train"] + split["val"] + split["test"]
    parts = (
        ["train"] * len(split["train"])
        + ["val"] * len(split["val"])
        + ["test"] * len(split["test"])
    )
    moving = [inventory["episodes"][e][camera]["moving_frames"] for e in eps]
    paused = [inventory["episodes"][e][camera]["pause_frames"] for e in eps]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(eps))
    ax.bar(x, moving, label="Moving frames", color="#3b82f6")
    ax.bar(x, paused, bottom=moving, label="Paused frames", color="#f59e0b")
    for i, (e, part) in enumerate(zip(eps, parts)):
        tot = moving[i] + paused[i]
        ax.text(i, tot + 6, part.upper(), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([e.split("__")[-1] for e in eps], rotation=30, ha="right")
    ax.set_ylabel("Frames")
    ax.set_title(f"Episode inventory & frozen 6/2/2 split ({camera})")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_training_curves(summaries: Mapping[str, dict]) -> plt.Figure:
    """Plot partial-BCE loss and validation confident-IoU across trained students."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in summaries.items():
        hist = s["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["train_loss"] for h in hist], "-o", label=f"{name} train")
        axes[0].plot(epochs, [h["val_loss"] for h in hist], "--s", label=f"{name} val")
        axes[1].plot(epochs, [h["val_confident_iou"] for h in hist], "-o", label=name)
    axes[0].set_title("Partial Balanced BCE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Val Confident-Pixel IoU (vs. Teacher)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_latency_comparison(reports: Mapping[str, dict]) -> plt.Figure:
    """Plot per-stage median latency + p95/max markers against the 100 ms 10 Hz deadline."""
    names = list(reports.keys())
    y = np.arange(len(names))
    stages = [
        ("capture_ms", "Capture", "#64748b"),
        ("preprocess_ms", "Preprocess", "#0ea5e9"),
        ("transfer_ms", "Transfer", "#8b5cf6"),
        ("inference_ms", "Inference", "#22c55e"),
        ("resize_ms", "Resize", "#f59e0b"),
    ]
    fig, ax = plt.subplots(figsize=(10, max(3.2, 1.0 * len(names) + 1.4)))
    left = np.zeros(len(names), dtype=np.float64)
    for key, label, color in stages:
        vals = np.array([reports[n][key]["median"] for n in names], dtype=np.float64)
        ax.barh(y, vals, left=left, height=0.45, label=label, color=color)
        left += vals

    p95 = [reports[n]["tick_total_ms"]["p95"] for n in names]
    mx = [reports[n]["tick_total_ms"]["max"] for n in names]
    ax.scatter(p95, y, color="#ef4444", marker="|", s=220, linewidths=2.5, label="Tick p95", zorder=5)
    ax.scatter(mx, y, color="#991b1b", marker="x", s=70, linewidths=2.0, label="Tick max", zorder=6)
    ax.axvline(100.0, color="#dc2626", linestyle="--", linewidth=1.5, label="100 ms (10 Hz) budget")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Milliseconds per tick")
    ax.set_title("Paced 10 Hz per-stage latency breakdown")
    ax.set_xlim(0, max(105.0, max(mx) * 1.15))
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig
