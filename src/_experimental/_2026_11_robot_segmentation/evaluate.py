"""Evaluation metrics, part/jaw-gap/shadow checks, pause stability, and worst-case overlays."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .data import (
    camera_video_path,
    detect_pause_flags,
    frame_motion_scores,
    load_label_png,
    read_video_frames,
    save_rgb_png,
)

# Optional single-channel part-annotated reference mask codes:
#   0 = background, 1 = robot arm/links, 2 = static base, 3 = gripper/jaws,
#   4 = open jaw gap (background), 5 = cast shadow (background)
PART_BG = 0
PART_ARM = 1
PART_BASE = 2
PART_GRIPPER = 3
PART_JAW_GAP = 4
PART_SHADOW = 5


def foreground_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union > 0 else 1.0


def part_recall(pred: np.ndarray, gt_part: np.ndarray) -> float | None:
    p = np.asarray(pred) > 0
    g = np.asarray(gt_part) > 0
    denom = int(g.sum())
    if denom == 0:
        return None
    return float((p & g).sum()) / float(denom)


def shadow_leakage(pred: np.ndarray, shadow_mask: np.ndarray) -> float | None:
    p = np.asarray(pred) > 0
    s = np.asarray(shadow_mask) > 0
    denom = int(s.sum())
    if denom == 0:
        return None
    return float((p & s).sum()) / float(denom)


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    m = (np.asarray(mask) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(m, k, iterations=1)
    return (m - eroded) > 0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 2) -> float:
    """Boundary F1 at native resolution with pixel tolerance ``tolerance_px``."""
    pb = _extract_boundary(pred)
    gb = _extract_boundary(gt)
    n_p = int(pb.sum())
    n_g = int(gb.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    if n_p == 0 or n_g == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    gb_dil = cv2.dilate(gb.astype(np.uint8), k, iterations=1) > 0
    pb_dil = cv2.dilate(pb.astype(np.uint8), k, iterations=1) > 0
    prec = float((pb & gb_dil).sum()) / float(n_p)
    rec = float((gb & pb_dil).sum()) / float(n_g)
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def jaw_gap_metrics(
    pred: np.ndarray,
    jaw_gap_mask: np.ndarray,
    input_size: int | tuple[int, int] | None = None,
) -> dict:
    """Measure open-jaw gap resolvability, false foreground inside the gap, and gap openness."""
    gap = (np.asarray(jaw_gap_mask) > 0).astype(np.uint8)
    gap_px = int(gap.sum())
    if gap_px == 0:
        return {
            "has_gap": False,
            "native_gap_width_px": None,
            "scaled_gap_width_px": None,
            "resolvable_at_input_size": None,
            "false_fg_in_gap": None,
            "gap_kept_open": None,
        }

    # Estimate characteristic gap width via 2 * max inscribed radius (distance transform)
    dist = cv2.distanceTransform(gap, cv2.DIST_L2, 5)
    native_width = float(2.0 * dist.max())
    h, w = gap.shape
    if input_size is not None:
        th, tw = (input_size, input_size) if isinstance(input_size, int) else input_size
        scale = min(tw / w, th / h)
        scaled_width = native_width * scale
        resolvable = scaled_width >= 1.0
    else:
        scaled_width = native_width
        resolvable = native_width >= 1.0

    p = np.asarray(pred) > 0
    g = gap > 0
    false_fg = float((p & g).sum()) / float(gap_px)
    open_pixels = (~p) & g
    # Gap stays open when >50% of gap pixels stay background and form a non-trivial open region
    kept_open = bool((false_fg < 0.5) and (open_pixels.sum() >= max(1, int(0.5 * gap_px))))
    return {
        "has_gap": True,
        "native_gap_width_px": native_width,
        "scaled_gap_width_px": float(scaled_width),
        "resolvable_at_input_size": bool(resolvable),
        "false_fg_in_gap": false_fg,
        "gap_kept_open": kept_open,
    }


def pause_stability(masks: Sequence[np.ndarray], pause_flags: np.ndarray) -> dict:
    """Measure temporal flicker and mask area loss during stationary pause runs."""
    arr = np.stack([np.asarray(m) > 0 for m in masks], axis=0)
    flags = np.asarray(pause_flags, dtype=bool)
    n = len(arr)
    runs = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= 2:
                runs.append((start, i))
            start = None
    if start is not None and n - start >= 2:
        runs.append((start, n))

    if not runs:
        return {
            "n_pause_runs": 0,
            "pause_frames": int(flags.sum()),
            "mean_pause_flicker_rate": None,
            "mean_pause_adj_iou": None,
            "max_pause_area_drop_frac": None,
        }

    flip_rates, adj_ious, area_drops = [], [], []
    for s, e in runs:
        run_masks = arr[s:e]
        for t in range(len(run_masks) - 1):
            a, b = run_masks[t], run_masks[t + 1]
            flip_rates.append(float((a ^ b).mean()))
            inter = int((a & b).sum())
            union = int((a | b).sum())
            adj_ious.append(inter / union if union > 0 else 1.0)
        areas = run_masks.sum(axis=(1, 2)).astype(np.float32)
        if areas[0] > 0:
            drop = float(max(0.0, (areas[0] - areas.min()) / areas[0]))
            area_drops.append(drop)

    return {
        "n_pause_runs": len(runs),
        "pause_frames": int(flags.sum()),
        "mean_pause_flicker_rate": float(np.mean(flip_rates)) if flip_rates else None,
        "mean_pause_adj_iou": float(np.mean(adj_ious)) if adj_ious else None,
        "max_pause_area_drop_frac": float(np.max(area_drops)) if area_drops else None,
    }


def annotation_repeatability(
    masks_a: Sequence[np.ndarray],
    masks_b: Sequence[np.ndarray],
    tolerance_px: int = 2,
) -> dict:
    """Estimate annotation uncertainty from re-marked evaluation frames (Step 1)."""
    if len(masks_a) != len(masks_b) or len(masks_a) == 0:
        raise ValueError("masks_a and masks_b must be non-empty and equal length")
    ious = [foreground_iou(a, b) for a, b in zip(masks_a, masks_b)]
    bf1s = [boundary_f1(a, b, tolerance_px=tolerance_px) for a, b in zip(masks_a, masks_b)]
    return {
        "n_frames": len(masks_a),
        "tolerance_px": tolerance_px,
        "mean_iou": float(np.mean(ious)),
        "min_iou": float(np.min(ious)),
        "mean_boundary_f1": float(np.mean(bf1s)),
        "min_boundary_f1": float(np.min(bf1s)),
    }


def parse_reference_mask(mask_path: str | Path) -> dict[str, np.ndarray | None]:
    """Load either a part-coded PNG (0..5) or a plain binary mask (0/1 or 0/255)."""
    raw = load_label_png(mask_path)
    uniq = set(np.unique(raw).tolist())
    if uniq.issubset({0, 1, 255}):
        robot = (raw > 0).astype(np.uint8)
        base = gripper = jaw_gap = shadow = None
    else:
        robot = np.isin(raw, [PART_ARM, PART_BASE, PART_GRIPPER]).astype(np.uint8)
        base = (raw == PART_BASE).astype(np.uint8)
        gripper = (raw == PART_GRIPPER).astype(np.uint8)
        jaw_gap = (raw == PART_JAW_GAP).astype(np.uint8)
        shadow = (raw == PART_SHADOW).astype(np.uint8)

    # Also allow optional sibling part masks: <stem>_base.png, _gripper.png, _jaw_gap.png, _shadow.png
    p = Path(mask_path)
    for key, cur in [("base", base), ("gripper", gripper), ("jaw_gap", jaw_gap), ("shadow", shadow)]:
        side = p.with_name(f"{p.stem}_{key}.png")
        if side.is_file():
            arr = (load_label_png(side) > 0).astype(np.uint8)
            if key == "base":
                base = arr
            elif key == "gripper":
                gripper = arr
            elif key == "jaw_gap":
                jaw_gap = arr
            elif key == "shadow":
                shadow = arr
    return {"robot": robot, "base": base, "gripper": gripper, "jaw_gap": jaw_gap, "shadow": shadow}


def render_comparison_overlay(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """TP=green, FP=red, FN=blue overlay on RGB uint8 image."""
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    out = rgb.astype(np.float32).copy()
    tp = p & g
    fp = p & (~g)
    fn = (~p) & g
    alpha = 0.5
    if tp.any():
        out[tp] = (1 - alpha) * out[tp] + alpha * np.array([30, 220, 70], dtype=np.float32)
    if fp.any():
        out[fp] = (1 - alpha) * out[fp] + alpha * np.array([235, 45, 45], dtype=np.float32)
    if fn.any():
        out[fn] = (1 - alpha) * out[fn] + alpha * np.array([45, 110, 245], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def extract_annotation_pack(
    inventory: dict,
    out_dir: str | Path,
    camera: str = "dev2",
) -> Path:
    """Export selected validation and sealed test frames for independent mask annotation."""
    root = Path(out_dir)
    pack_path = Path(inventory["pack_path"])
    manifest = {"pack_path": str(pack_path), "camera": camera, "val": [], "test_sealed": []}

    for part, subdir in [("val", "val"), ("test", "test_sealed")]:
        for ep, items in inventory["eval_candidates"][part].items():
            fids = [int(it["frame_idx"]) for it in items]
            ids, frames = read_video_frames(camera_video_path(pack_path, ep, camera), indices=fids)
            id_to_meta = {int(it["frame_idx"]): it for it in items}
            for fidx, rgb in zip(ids, frames):
                img_rel = f"{subdir}/{camera}/{ep}/frame_{fidx:06d}.png"
                mask_rel = f"{subdir}/{camera}/{ep}/frame_{fidx:06d}_mask.png"
                save_rgb_png(root / img_rel, rgb)
                manifest[subdir].append(
                    {
                        "episode": ep,
                        "camera": camera,
                        "frame_idx": int(fidx),
                        "image_rel": img_rel,
                        "mask_rel": mask_rel,
                        "is_pause": bool(id_to_meta[fidx]["is_pause"]),
                    }
                )
    (root / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root


def _mean_or_none(vals: Sequence[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def evaluate_predictor(
    predictor: Any,
    eval_manifest_path: str | Path,
    partition: str,
    out_dir: str | Path,
    tolerance_px: int = 2,
    input_size: int | tuple[int, int] | None = None,
    worst_k: int = 8,
    pack_path_for_pause: str | Path | None = None,
) -> dict:
    """Evaluate a predictor against independent reference masks, saving CSV, JSON, Markdown & worst overlays."""
    manifest_path = Path(eval_manifest_path)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if partition not in manifest:
        raise KeyError(f"Partition '{partition}' not in {manifest_path} (expected 'val' or 'test_sealed')")

    out = Path(out_dir)
    ovl_dir = out / "worst_overlays"
    ovl_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    overlays = []
    episodes_seen = {}

    for item in manifest[partition]:
        mask_path = root / item["mask_rel"]
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing reference mask: {mask_path}")
        img_bgr = cv2.imread(str(root / item["image_rel"]), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Missing eval image: {root / item['image_rel']}")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ref = parse_reference_mask(mask_path)
        pred = predictor.segment(rgb)

        iou = foreground_iou(pred, ref["robot"])
        bf1 = boundary_f1(pred, ref["robot"], tolerance_px=tolerance_px)
        b_rec = part_recall(pred, ref["base"]) if ref["base"] is not None else None
        g_rec = part_recall(pred, ref["gripper"]) if ref["gripper"] is not None else None
        s_leak = shadow_leakage(pred, ref["shadow"]) if ref["shadow"] is not None else None
        gap = (
            jaw_gap_metrics(pred, ref["jaw_gap"], input_size=input_size)
            if ref["jaw_gap"] is not None
            else {"has_gap": False, "resolvable_at_input_size": None, "false_fg_in_gap": None, "gap_kept_open": None}
        )
        row = {
            "episode": item["episode"],
            "camera": item["camera"],
            "frame_idx": int(item["frame_idx"]),
            "is_pause": bool(item.get("is_pause", False)),
            "iou": iou,
            "boundary_f1": bf1,
            "base_recall": b_rec,
            "gripper_recall": g_rec,
            "shadow_leakage": s_leak,
            "jaw_gap_resolvable": gap["resolvable_at_input_size"],
            "jaw_gap_false_fg": gap["false_fg_in_gap"] if gap["resolvable_at_input_size"] else None,
            "jaw_gap_kept_open": gap["gap_kept_open"] if gap["resolvable_at_input_size"] else None,
        }
        rows.append(row)
        overlays.append((iou, item, render_comparison_overlay(rgb, pred, ref["robot"])))
        episodes_seen[(item["episode"], item["camera"])] = True

    # Save worst-k overlays
    overlays.sort(key=lambda x: x[0])
    for rank, (iou, item, img) in enumerate(overlays[:worst_k], start=1):
        fname = f"worst_{rank:02d}_{item['camera']}_{item['episode']}_f{item['frame_idx']:06d}_iou{iou:.3f}.png"
        save_rgb_png(ovl_dir / fname, img)

    # Optional full-episode pause stability check
    pause_by_ep = {}
    if pack_path_for_pause is not None:
        for ep, cam in sorted(episodes_seen.keys()):
            vpath = camera_video_path(pack_path_for_pause, ep, cam)
            _, frames = read_video_frames(vpath)
            motion = frame_motion_scores(frames)
            pflags, _ = detect_pause_flags(motion)
            preds = [predictor.segment(f) for f in frames]
            pause_by_ep[f"{cam}/{ep}"] = pause_stability(preds, pflags)

    # Aggregate per episode and overall
    per_episode = {}
    for ep, cam in sorted(episodes_seen.keys()):
        sub = [r for r in rows if r["episode"] == ep and r["camera"] == cam]
        per_episode[f"{cam}/{ep}"] = {
            "n_frames": len(sub),
            "mean_iou": _mean_or_none([r["iou"] for r in sub]),
            "mean_boundary_f1": _mean_or_none([r["boundary_f1"] for r in sub]),
            "mean_base_recall": _mean_or_none([r["base_recall"] for r in sub]),
            "mean_gripper_recall": _mean_or_none([r["gripper_recall"] for r in sub]),
            "mean_shadow_leakage": _mean_or_none([r["shadow_leakage"] for r in sub]),
            "mean_jaw_gap_false_fg": _mean_or_none([r["jaw_gap_false_fg"] for r in sub]),
            "jaw_gap_open_rate": _mean_or_none(
                [1.0 if r["jaw_gap_kept_open"] else 0.0 for r in sub if r["jaw_gap_kept_open"] is not None]
            ),
        }

    summary = {
        "partition": partition,
        "tolerance_px": tolerance_px,
        "n_frames": len(rows),
        "overall": {
            "mean_iou": _mean_or_none([r["iou"] for r in rows]),
            "mean_boundary_f1": _mean_or_none([r["boundary_f1"] for r in rows]),
            "mean_base_recall": _mean_or_none([r["base_recall"] for r in rows]),
            "mean_gripper_recall": _mean_or_none([r["gripper_recall"] for r in rows]),
            "mean_shadow_leakage": _mean_or_none([r["shadow_leakage"] for r in rows]),
            "mean_jaw_gap_false_fg": _mean_or_none([r["jaw_gap_false_fg"] for r in rows]),
            "jaw_gap_open_rate": _mean_or_none(
                [1.0 if r["jaw_gap_kept_open"] else 0.0 for r in rows if r["jaw_gap_kept_open"] is not None]
            ),
        },
        "per_episode": per_episode,
        "pause_stability": pause_by_ep,
    }

    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_lines = [
        f"# Evaluation Report ({partition})",
        "",
        f"- Frames: {len(rows)}, Boundary tolerance: {tolerance_px} px",
        f"- Overall IoU: {summary['overall']['mean_iou']:.4f}" if summary["overall"]["mean_iou"] is not None else "- Overall IoU: n/a",
        f"- Overall Boundary F1: {summary['overall']['mean_boundary_f1']:.4f}" if summary["overall"]["mean_boundary_f1"] is not None else "- Overall Boundary F1: n/a",
        f"- Base recall: {summary['overall']['mean_base_recall']}",
        f"- Gripper recall: {summary['overall']['mean_gripper_recall']}",
        f"- Shadow leakage: {summary['overall']['mean_shadow_leakage']}",
        f"- Open jaw-gap rate: {summary['overall']['jaw_gap_open_rate']}",
    ]
    (out / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary
