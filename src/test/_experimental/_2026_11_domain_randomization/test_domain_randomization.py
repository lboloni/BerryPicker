"""Unit tests for the standalone domain randomization experimental package."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parents[3]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from _experimental._2026_11_domain_randomization import (
    OP_REGISTRY,
    DomainRandomizer,
    apply_op,
)
from _experimental._2026_11_domain_randomization.dataset_builder import (
    cache_episode_masks,
    make_techniques_contact_sheet,
    materialize_randomized_demopack,
    summarize_dr_comparisons,
    write_video_rgb,
)

_PKG_DIR = _SRC_DIR / "_experimental" / "_2026_11_domain_randomization"
_RECIPES_DIR = _PKG_DIR / "recipes"


def _synthetic_frame_and_mask(h: int = 64, w: int = 64) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.linspace(0, 255, h), np.linspace(0, 255, w), indexing="ij")
    rgb = np.stack([xx, yy, np.full((h, w), 120.0)], axis=2).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[18:48, 20:46] = 1
    rgb[mask == 1] = [40, 180, 90]
    return rgb, mask


def test_standalone_no_berrypicker_core_imports():
    """Verify no .py module in _2026_11_domain_randomization imports BerryPicker core or ExpRunFlow."""
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
    for py_file in _PKG_DIR.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert node.module.split(".")[0] not in forbidden, f"{py_file.name} imports {node.module}"


def test_all_ops_shape_dtype_and_determinism(tmp_path: Path):
    """Every registered op returns uint8 (H, W, 3) and is deterministic for a fixed RNG seed."""
    rgb, mask = _synthetic_frame_and_mask(64, 64)

    # Also test image_folder background generator
    bg_dir = tmp_path / "bgs"
    bg_dir.mkdir()
    cv2.imwrite(str(bg_dir / "bg0.png"), np.full((80, 80, 3), 200, dtype=np.uint8))

    extra_params = {
        "background_replace": {
            "sources": [
                {"kind": "solid_color", "mode": "random", "weight": 0.2},
                {"kind": "smooth_field", "weight": 0.2},
                {"kind": "multi_scale_noise", "weight": 0.2},
                {"kind": "geometric_pattern", "style": "random", "weight": 0.2},
                {"kind": "image_folder", "folder": str(bg_dir), "weight": 0.2},
            ],
            "edge_feather_px": 1,
            "mask_jitter_px": 1,
        }
    }

    for op_name in sorted(OP_REGISTRY):
        params = extra_params.get(op_name, {})
        out1 = apply_op(op_name, rgb, np.random.default_rng(123), mask=mask, params=params)
        out2 = apply_op(op_name, rgb, np.random.default_rng(123), mask=mask, params=params)
        assert out1.shape == rgb.shape and out1.dtype == np.uint8
        assert np.array_equal(out1, out2), f"Non-deterministic op: {op_name}"


def test_mask_preservation_and_fail_fast_validation():
    """Background replace (feather=0, jitter=0) and bg-only distractors preserve robot pixels verbatim."""
    rgb, mask = _synthetic_frame_and_mask(64, 64)

    bg_out = apply_op(
        "background_replace",
        rgb,
        np.random.default_rng(7),
        mask=mask,
        params={
            "sources": [{"kind": "solid_color", "mode": "fixed", "color": [255, 255, 255]}],
            "edge_feather_px": 0,
            "mask_jitter_px": 0,
        },
    )
    assert np.array_equal(bg_out[mask == 1], rgb[mask == 1])
    assert np.all(bg_out[mask == 0] == 255)

    dist_out = apply_op(
        "distractors",
        rgb,
        np.random.default_rng(8),
        mask=mask,
        params={"count_range": [5, 5], "placement": "background_only", "size_frac_range": [0.2, 0.4]},
    )
    assert np.array_equal(dist_out[mask == 1], rgb[mask == 1])

    # Fail-fast on unknown op or missing required mask
    with pytest.raises(KeyError):
        DomainRandomizer.from_dict({"name": "bad", "pipeline": [{"op": "nonexistent_op"}]})
    with pytest.raises(ValueError):
        apply_op("background_replace", rgb, np.random.default_rng(0), mask=None, params={"sources": [{"kind": "smooth_field"}]})


def test_checked_in_yaml_recipes_and_contact_sheet(tmp_path: Path):
    """All checked-in YAML recipes load cleanly and render the all-techniques contact sheet."""
    rgb, mask = _synthetic_frame_and_mask(64, 64)
    recipe_files = sorted(_RECIPES_DIR.glob("*.yaml"))
    assert len(recipe_files) >= 5

    for rpath in recipe_files:
        dr = DomainRandomizer.from_yaml(rpath)
        m = mask if dr.requires_mask else None
        out = dr(rgb, mask=m, rng=dr.frame_rng(0, 0, 0))
        assert out.shape == rgb.shape and out.dtype == np.uint8

    sheet_path = make_techniques_contact_sheet(rgb, mask, _RECIPES_DIR, tmp_path / "sheet.png")
    assert sheet_path.is_file() and sheet_path.stat().st_size > 0
    img = cv2.imread(str(sheet_path))
    assert img is not None and img.shape[0] > 64 and img.shape[1] == 64 * 4


def test_dataset_materialization_multi_copy_and_csv_summary(tmp_path: Path):
    """Verify mask caching, multi-copy materialization, metadata copying, and Suite A/B CSV summary."""
    rgb, mask = _synthetic_frame_and_mask(64, 64)
    src_yaml = tmp_path / "expruns" / "demonstration" / "demo_src.yaml"
    dst_yaml = tmp_path / "expruns" / "demonstration" / "demo_dr.yaml"
    src_yaml.parent.mkdir(parents=True)
    src_yaml.write_text("name: demo_src\n", encoding="utf-8")

    src_root = tmp_path / "results" / "demo_src"
    ep_dir = src_root / "ep0"
    ep_dir.mkdir(parents=True)
    write_video_rgb(ep_dir / "video_dev2.mp4", [rgb.copy() for _ in range(4)], fps=10.0)
    (ep_dir / "_metadata.yaml").write_text("maxsteps: 4\nstored_as_video: true\nstored_as_images: false\n", encoding="utf-8")
    (ep_dir / "_action.yaml").write_text("- rc-position-target: [0, 0, 0, 0, 0, 0]\n", encoding="utf-8")
    (ep_dir / "_annotation.yaml").write_text("- reward: 0\n", encoding="utf-8")

    class _ConstSeg:
        def segment(self, frame_rgb):
            return mask

    mask_paths = cache_episode_masks(src_root, tmp_path / "masks", ["ep0"], "dev2", _ConstSeg())
    assert mask_paths["ep0"].is_file()

    dr = DomainRandomizer.from_yaml(_RECIPES_DIR / "dr_bg_swap.yaml")
    dst_root = tmp_path / "results" / "demo_dr"
    res = materialize_randomized_demopack(
        src_yaml_path=src_yaml,
        dst_yaml_path=dst_yaml,
        src_root=src_root,
        dst_root=dst_root,
        dst_demo_run="demo_dr",
        episodes=["ep0"],
        camera="dev2",
        randomizer=dr,
        mask_paths=mask_paths,
        copies_per_episode=2,
        preview_path=tmp_path / "preview.png",
    )
    assert res["episode_map"]["ep0"] == ["ep0__copy00", "ep0__copy01"]
    assert res["total_frames"] == 8
    assert (dst_root / "ep0__copy00" / "video_dev2.mp4").is_file()
    assert (dst_root / "ep0__copy01" / "video_dev2.mp4").is_file()
    assert (dst_root / "ep0__copy01" / "_action.yaml").read_text(encoding="utf-8") == (ep_dir / "_action.yaml").read_text(encoding="utf-8")
    assert (tmp_path / "preview.png").is_file()

    # Second run with overwrite=False uses cached videos
    res2 = materialize_randomized_demopack(
        src_yaml_path=src_yaml,
        dst_yaml_path=dst_yaml,
        src_root=src_root,
        dst_root=dst_root,
        dst_demo_run="demo_dr",
        episodes=["ep0"],
        camera="dev2",
        randomizer=dr,
        mask_paths=mask_paths,
        copies_per_episode=2,
        overwrite=False,
    )
    assert res2["cached_videos"] == 2

    # Suite A vs Suite B CSV summarization
    clean_csv = tmp_path / "clean.csv"
    ood_csv = tmp_path / "ood.csv"
    clean_csv.write_text("Title,vp_vae_orig,vp_vae_dr_full\nheight,0.10,0.12\ndistance,0.20,0.18\n", encoding="utf-8")
    ood_csv.write_text("Title,vp_vae_orig_ood,vp_vae_dr_full_ood\nheight,0.30,0.14\ndistance,0.40,0.20\n", encoding="utf-8")
    records, md = summarize_dr_comparisons(clean_csv, ood_csv)
    assert len(records) == 2
    assert records[0]["clean_mean_rmse"] == pytest.approx(0.15)
    assert records[0]["ood_mean_rmse"] == pytest.approx(0.35)
    assert records[1]["ood_delta"] == pytest.approx(0.02)
    assert "OOD Degradation" in md
