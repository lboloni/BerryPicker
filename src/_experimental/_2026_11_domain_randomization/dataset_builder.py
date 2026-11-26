"""Standalone demonstration materializer, contact-sheet renderer, and CSV summary helpers.

All functions accept explicit file/directory paths so this module keeps zero
imports from BerryPicker core packages or ``ExpRunFlow``.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from .pipeline import DomainRandomizer
from .transforms import apply_op


def read_video_rgb(video_path: Path) -> tuple[float, list[np.ndarray]]:
    """Read all frames from an MP4 video as RGB uint8 arrays, returning (fps, frames)."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    frames: list[np.ndarray] = []
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return fps, frames


def write_video_rgb(video_path: Path, frames_rgb: Sequence[np.ndarray], fps: float) -> None:
    """Atomically write RGB uint8 frames to an MP4 video and verify frame count."""
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        raise ValueError(f"Cannot write empty video: {video_path}")
    h, w = frames_rgb[0].shape[:2]
    tmp_path = video_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {tmp_path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    tmp_path.replace(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Written video cannot be opened: {video_path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != len(frames_rgb):
            raise RuntimeError(
                f"Written video {video_path} has {count} frames, expected {len(frames_rgb)}"
            )
    finally:
        cap.release()


def cache_episode_masks(
    src_root: Path,
    mask_cache_dir: Path,
    episodes: Sequence[str],
    camera: str,
    segmenter: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Run ``segmenter`` once per episode and cache compressed uint8 binary masks."""
    src_root = Path(src_root)
    mask_cache_dir = Path(mask_cache_dir)
    mask_cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for ep in episodes:
        ep_dir = src_root / ep
        with open(ep_dir / "_metadata.yaml", encoding="utf-8") as fh:
            expected_steps = int(yaml.safe_load(fh)["maxsteps"])
        npz_path = mask_cache_dir / f"{ep}_{camera}.npz"
        if npz_path.is_file() and not overwrite:
            with np.load(npz_path) as data:
                if data["masks"].shape[0] == expected_steps:
                    out[ep] = npz_path
                    continue
        _, frames = read_video_rgb(ep_dir / f"video_{camera}.mp4")
        if len(frames) < expected_steps:
            raise RuntimeError(f"{ep}: video has {len(frames)} frames < maxsteps {expected_steps}")
        masks = np.stack(
            [(segmenter.segment(frames[i]) > 0).astype(np.uint8) for i in range(expected_steps)],
            axis=0,
        )
        np.savez_compressed(npz_path, masks=masks)
        out[ep] = npz_path
    return out


def _load_cached_masks(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        return data["masks"]


def materialize_randomized_demopack(
    src_yaml_path: Path,
    dst_yaml_path: Path,
    src_root: Path,
    dst_root: Path,
    dst_demo_run: str,
    episodes: Sequence[str],
    camera: str,
    randomizer: DomainRandomizer,
    *,
    mask_paths: dict[str, Path] | None = None,
    live_segmenter: Any | None = None,
    copies_per_episode: int | None = None,
    preview_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a sibling demonstration run with a YAML DR recipe applied.

    When ``copies_per_episode == 1``, output episode names match ``episodes``
    verbatim. When ``copies_per_episode > 1``, episode ``ep`` expands to
    ``ep__copy00``, ``ep__copy01``, ... and the returned ``episode_map`` maps
    each source episode name to its list of materialized episode directory names.
    """
    src_yaml_path = Path(src_yaml_path)
    dst_yaml_path = Path(dst_yaml_path)
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_yaml_path.is_file():
        raise FileNotFoundError(src_yaml_path)

    n_copies = randomizer.copies_per_episode if copies_per_episode is None else int(copies_per_episode)
    if n_copies < 1:
        raise ValueError(f"copies_per_episode must be >= 1, got {n_copies}")

    with open(src_yaml_path, encoding="utf-8") as fh:
        demo_cfg = yaml.safe_load(fh) or {}
    src_demo_run = demo_cfg.get("name", src_yaml_path.stem)
    demo_cfg["name"] = dst_demo_run
    demo_cfg["dr_source"] = src_demo_run
    demo_cfg["dr_recipe"] = randomizer.name
    demo_cfg["dr_copies_per_episode"] = n_copies
    dst_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(demo_cfg, fh)

    dst_root.mkdir(parents=True, exist_ok=True)

    episode_map: dict[str, list[str]] = {}
    samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    total_frames = 0
    cached_videos = 0

    for ep_idx, ep in enumerate(episodes):
        src_ep_dir = src_root / ep
        with open(src_ep_dir / "_metadata.yaml", encoding="utf-8") as fh:
            expected_steps = int(yaml.safe_load(fh)["maxsteps"])
        fps, raw_frames = read_video_rgb(src_ep_dir / f"video_{camera}.mp4")
        if len(raw_frames) < expected_steps:
            raise RuntimeError(f"{ep}: video has {len(raw_frames)} frames < maxsteps {expected_steps}")

        ep_masks: np.ndarray | None = None
        if randomizer.requires_mask:
            if mask_paths is not None and ep in mask_paths:
                ep_masks = _load_cached_masks(mask_paths[ep])
            elif live_segmenter is not None:
                ep_masks = np.stack(
                    [(live_segmenter.segment(raw_frames[i]) > 0).astype(np.uint8) for i in range(expected_steps)],
                    axis=0,
                )
            else:
                raise ValueError(
                    f"Recipe '{randomizer.name}' requires masks, but neither mask_paths['{ep}'] "
                    "nor live_segmenter was provided"
                )

        dst_names: list[str] = []
        for copy_idx in range(n_copies):
            dst_ep_name = ep if n_copies == 1 else f"{ep}__copy{copy_idx:02d}"
            dst_names.append(dst_ep_name)
            dst_ep_dir = dst_root / dst_ep_name
            dst_ep_dir.mkdir(parents=True, exist_ok=True)

            for meta_name in ("_action.yaml", "_annotation.yaml", "_metadata.yaml"):
                src_meta = src_ep_dir / meta_name
                if not src_meta.is_file():
                    raise FileNotFoundError(src_meta)
                dst_meta = dst_ep_dir / meta_name
                if overwrite or not dst_meta.is_file():
                    shutil.copy2(src_meta, dst_meta)

            dst_video = dst_ep_dir / f"video_{camera}.mp4"
            if dst_video.is_file() and not overwrite:
                _, existing_frames = read_video_rgb(dst_video)
                if len(existing_frames) != expected_steps:
                    raise RuntimeError(
                        f"Cached video {dst_video} has {len(existing_frames)} frames (expected {expected_steps})"
                    )
                cached_videos += 1
                total_frames += expected_steps
                if copy_idx == 0 and len(samples) < 6:
                    mid = expected_steps // 2
                    samples.append((dst_ep_name, raw_frames[mid], existing_frames[mid]))
                continue

            out_frames: list[np.ndarray] = []
            for frame_idx in range(expected_steps):
                rng = randomizer.frame_rng(ep_idx, frame_idx, copy_idx)
                m = None if ep_masks is None else ep_masks[frame_idx]
                out_frames.append(randomizer(raw_frames[frame_idx], mask=m, rng=rng))
            write_video_rgb(dst_video, out_frames, fps)
            total_frames += expected_steps
            if copy_idx == 0 and len(samples) < 6:
                mid = expected_steps // 2
                samples.append((dst_ep_name, raw_frames[mid], out_frames[mid]))

        episode_map[ep] = dst_names

    saved_preview = None
    if preview_path is not None and samples:
        saved_preview = save_recipe_preview(samples, randomizer.name, Path(preview_path))

    return {
        "src_demo_run": src_demo_run,
        "dst_demo_run": dst_demo_run,
        "recipe": randomizer.name,
        "dst_root": str(dst_root),
        "camera": camera,
        "copies_per_episode": n_copies,
        "episode_map": episode_map,
        "total_frames": total_frames,
        "cached_videos": cached_videos,
        "preview_path": str(saved_preview) if saved_preview is not None else None,
    }


def _labeled_tile(rgb: np.ndarray, title: str, banner_h: int = 26) -> np.ndarray:
    h, w = rgb.shape[:2]
    banner = np.full((banner_h, w, 3), 242, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (6, banner_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    tile = np.concatenate([banner, rgb], axis=0)
    # 1px subtle border for clean visual separation
    cv2.rectangle(tile, (0, 0), (w - 1, h + banner_h - 1), (180, 180, 180), 1)
    return tile


def save_recipe_preview(
    samples: Sequence[tuple[str, np.ndarray, np.ndarray]],
    recipe_name: str,
    out_path: Path,
) -> Path:
    """Save a (raw | randomized) contact sheet across episodes for one recipe."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        np.concatenate(
            [_labeled_tile(raw, f"{ep} | raw"), _labeled_tile(aug, f"{ep} | {recipe_name}")],
            axis=1,
        )
        for ep, raw, aug in samples
    ]
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def make_techniques_contact_sheet(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    recipes_dir: Path,
    out_path: Path,
    *,
    seed: int = 42,
    ncols: int = 4,
) -> Path:
    """Render a grid contact sheet showing every individual DR technique and YAML recipe."""
    recipes_dir = Path(recipes_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _rng(k: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([seed, k]))

    tiles: list[np.ndarray] = [
        _labeled_tile(frame_rgb, "00 original (raw)"),
        _labeled_tile(
            apply_op(
                "background_replace",
                frame_rgb,
                _rng(1),
                mask=mask,
                params={"sources": [{"kind": "solid_color", "mode": "fixed", "color": [255, 255, 255]}]},
            ),
            "01 seg_white",
        ),
        _labeled_tile(
            apply_op("color_jitter", frame_rgb, _rng(2), params={"brightness": 0.3, "contrast": 0.3, "saturation": 0.35, "hue": 0.08}),
            "02 color_jitter",
        ),
        _labeled_tile(
            apply_op("lighting_gradient", frame_rgb, _rng(3), params={"strength_range": [0.32, 0.38], "mode": "linear"}),
            "03 lighting_gradient",
        ),
        _labeled_tile(
            apply_op("white_balance", frame_rgb, _rng(4), params={"gain_range": [0.78, 1.22]}),
            "04 white_balance",
        ),
        _labeled_tile(
            apply_op("gamma", frame_rgb, _rng(5), params={"gamma_range": [1.35, 1.55]}),
            "05 gamma",
        ),
        _labeled_tile(
            apply_op("gaussian_noise", frame_rgb, _rng(6), params={"sigma_range": [14.0, 16.0]}),
            "06 gaussian_noise",
        ),
        _labeled_tile(
            apply_op("gaussian_blur", frame_rgb, _rng(7), params={"kernel_sizes": [5], "sigma_range": [1.3, 1.5]}),
            "07 gaussian_blur",
        ),
        _labeled_tile(
            apply_op("jpeg_compression", frame_rgb, _rng(8), params={"quality_range": [18, 22]}),
            "08 jpeg_compression",
        ),
        _labeled_tile(
            apply_op(
                "background_replace",
                frame_rgb,
                _rng(9),
                mask=mask,
                params={"sources": [{"kind": "smooth_field", "grid_size": 4}], "edge_feather_px": 1},
            ),
            "09 bg: smooth_field",
        ),
        _labeled_tile(
            apply_op(
                "background_replace",
                frame_rgb,
                _rng(10),
                mask=mask,
                params={"sources": [{"kind": "geometric_pattern", "style": "checker"}], "edge_feather_px": 1},
            ),
            "10 bg: geometric_pattern",
        ),
        _labeled_tile(
            apply_op(
                "background_replace",
                frame_rgb,
                _rng(11),
                mask=mask,
                params={"sources": [{"kind": "multi_scale_noise"}], "edge_feather_px": 1},
            ),
            "11 bg: multi_scale_noise",
        ),
        _labeled_tile(
            apply_op(
                "distractors",
                frame_rgb,
                _rng(12),
                mask=mask,
                params={"count_range": [4, 5], "placement": "background_only", "size_frac_range": [0.10, 0.22]},
            ),
            "12 distractors (bg_only)",
        ),
        _labeled_tile(
            apply_op(
                "random_erasing",
                frame_rgb,
                _rng(13),
                params={"count_range": [2, 2], "area_frac_range": [0.04, 0.08], "fill": "random_color"},
            ),
            "13 random_erasing",
        ),
        _labeled_tile(
            apply_op(
                "camera_jitter",
                frame_rgb,
                _rng(14),
                params={"max_shift_px": 4.0, "max_deg": 2.5, "scale_range": [0.97, 1.03]},
            ),
            "14 camera_jitter (opt-in)",
        ),
        _labeled_tile(
            DomainRandomizer.from_yaml(recipes_dir / "dr_photo.yaml")(frame_rgb, rng=_rng(15)),
            "15 recipe: dr_photo",
        ),
        _labeled_tile(
            DomainRandomizer.from_yaml(recipes_dir / "dr_bg_swap.yaml")(frame_rgb, mask=mask, rng=_rng(16)),
            "16 recipe: dr_bg_swap",
        ),
        _labeled_tile(
            DomainRandomizer.from_yaml(recipes_dir / "dr_full.yaml")(frame_rgb, mask=mask, rng=_rng(17)),
            "17 recipe: dr_full (s1)",
        ),
        _labeled_tile(
            DomainRandomizer.from_yaml(recipes_dir / "dr_full.yaml")(frame_rgb, mask=mask, rng=_rng(18)),
            "18 recipe: dr_full (s2)",
        ),
        _labeled_tile(
            DomainRandomizer.from_yaml(recipes_dir / "eval_ood_clutter.yaml")(frame_rgb, mask=mask, rng=_rng(19)),
            "19 eval_ood_clutter",
        ),
    ]

    th, tw = tiles[0].shape[:2]
    while len(tiles) % ncols != 0:
        tiles.append(np.full((th, tw, 3), 255, dtype=np.uint8))

    rows = [np.concatenate(tiles[i : i + ncols], axis=1) for i in range(0, len(tiles), ncols)]
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def _parse_compare_csv(csv_path: Path) -> dict[str, dict[str, float]]:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"Comparison CSV has no metric rows: {csv_path}")
    run_names = rows[0][1:]
    per_run: dict[str, dict[str, float]] = {name: {} for name in run_names}
    for row in rows[1:]:
        dof = row[0]
        for name, val in zip(run_names, row[1:]):
            per_run[name][dof] = float(val)
    return per_run


def summarize_dr_comparisons(
    clean_csv: Path,
    ood_csv: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Build a unified Suite A (clean) vs Suite B (cluttered OOD) RMSE markdown table."""
    clean_map = _parse_compare_csv(clean_csv)
    ood_map = _parse_compare_csv(ood_csv) if (ood_csv is not None and Path(ood_csv).is_file()) else {}

    records: list[dict[str, Any]] = []
    for run_name, dof_vals in clean_map.items():
        clean_mean = float(np.mean(list(dof_vals.values())))
        ood_key = f"{run_name}_ood" if f"{run_name}_ood" in ood_map else run_name
        ood_dofs = ood_map.get(ood_key)
        ood_mean = float(np.mean(list(ood_dofs.values()))) if ood_dofs else None
        ood_delta = (ood_mean - clean_mean) if ood_mean is not None else None
        records.append(
            {
                "run": run_name,
                "clean_mean_rmse": clean_mean,
                "ood_mean_rmse": ood_mean,
                "ood_delta": ood_delta,
                "clean_dofs": dof_vals,
                "ood_dofs": ood_dofs,
            }
        )

    if ood_map:
        lines = [
            "| Run | Suite A Clean RMSE | Suite B Cluttered OOD RMSE | OOD Degradation (B - A) |",
            "|---|---:|---:|---:|",
        ]
        for r in records:
            ood_str = f"{r['ood_mean_rmse']:.4f}" if r["ood_mean_rmse"] is not None else "—"
            delta_str = f"{r['ood_delta']:+.4f}" if r["ood_delta"] is not None else "—"
            lines.append(f"| `{r['run']}` | {r['clean_mean_rmse']:.4f} | {ood_str} | {delta_str} |")
    else:
        lines = ["| Run | Suite A Clean RMSE |", "|---|---:|"]
        for r in records:
            lines.append(f"| `{r['run']}` | {r['clean_mean_rmse']:.4f} |")

    return records, "\n".join(lines)
