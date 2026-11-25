"""Step 2 baselines: per-camera median background and static positional control."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .data import camera_video_path, iter_video_frames


class MedianBackgroundModel:
    """Standalone median-background segmenter matching ``transfer_data.py``.

    Fits per-pixel median RGB and typical variation (MAD) strictly from
    training frames of one fixed camera. Exposes ``segment(frame_rgb) -> mask``.
    """

    def __init__(
        self,
        background_rgb: np.ndarray,
        mad_rgb: np.ndarray,
        threshold: float = 0.12,
        dilate: int = 1,
        use_adaptive_scale: float = 0.0,
        camera_id: str = "dev2",
    ):
        self.background_rgb = np.asarray(background_rgb, dtype=np.uint8)
        self.mad_rgb = np.asarray(mad_rgb, dtype=np.float32)
        self.threshold = float(threshold)
        self.dilate = int(dilate)
        self.use_adaptive_scale = float(use_adaptive_scale)
        self.camera_id = camera_id
        self._bg_float = self.background_rgb.astype(np.float32) / 255.0
        self._kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.dilate + 1, 2 * self.dilate + 1)
            )
            if self.dilate > 0
            else None
        )

    @classmethod
    def fit_from_frames(
        cls,
        frames_rgb: np.ndarray,
        threshold: float = 0.12,
        dilate: int = 1,
        use_adaptive_scale: float = 0.0,
        camera_id: str = "dev2",
    ) -> "MedianBackgroundModel":
        arr = np.asarray(frames_rgb, dtype=np.uint8)
        bg = np.median(arr, axis=0).round().astype(np.uint8)
        mad = np.median(np.abs(arr.astype(np.float32) - bg.astype(np.float32)), axis=0) / 255.0
        return cls(
            background_rgb=bg,
            mad_rgb=mad.astype(np.float32),
            threshold=threshold,
            dilate=dilate,
            use_adaptive_scale=use_adaptive_scale,
            camera_id=camera_id,
        )

    @classmethod
    def fit_from_demopack(
        cls,
        pack_path: str | Path,
        episodes: Sequence[str],
        camera: str = "dev2",
        frame_step: int = 10,
        threshold: float = 0.12,
        dilate: int = 1,
        use_adaptive_scale: float = 0.0,
    ) -> "MedianBackgroundModel":
        sampled = []
        for ep in episodes:
            vpath = camera_video_path(pack_path, ep, camera)
            for _, _, rgb in iter_video_frames(vpath, step=frame_step):
                sampled.append(rgb)
        if not sampled:
            raise RuntimeError(f"No training frames found for camera {camera}")
        return cls.fit_from_frames(
            np.stack(sampled, axis=0),
            threshold=threshold,
            dilate=dilate,
            use_adaptive_scale=use_adaptive_scale,
            camera_id=camera,
        )

    def diff_map(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Max-channel absolute difference from the background in [0, 1]."""
        img = np.asarray(frame_rgb, dtype=np.float32) / 255.0
        return np.abs(img - self._bg_float).max(axis=2)

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return uint8 binary mask (0/1) at input resolution."""
        diff = self.diff_map(frame_rgb)
        if self.use_adaptive_scale > 0.0:
            thr = np.maximum(self.threshold, self.use_adaptive_scale * self.mad_rgb.max(axis=2))
            mask = (diff > thr).astype(np.uint8)
        else:
            mask = (diff > self.threshold).astype(np.uint8)
        if self._kernel is not None:
            mask = cv2.dilate(mask, self._kernel, iterations=1)
        return mask

    def segment_timed(
        self, frame_rgb: np.ndarray, frame_id: int | None = None, capture_ts: float | None = None
    ) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        mask = self.segment(frame_rgb)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        return mask, {
            "frame_id": frame_id,
            "capture_ts": capture_ts,
            "deliver_ts": t1,
            "preprocess_ms": 0.0,
            "inference_ms": ms,
            "transfer_ms": 0.0,
            "resize_ms": 0.0,
            "total_ms": ms,
        }

    def save(self, out_dir: str | Path) -> Path:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "background.npz", background_rgb=self.background_rgb, mad_rgb=self.mad_rgb)
        meta = {
            "kind": "median_background",
            "camera_id": self.camera_id,
            "threshold": self.threshold,
            "dilate": self.dilate,
            "use_adaptive_scale": self.use_adaptive_scale,
            "height": int(self.background_rgb.shape[0]),
            "width": int(self.background_rgb.shape[1]),
        }
        (root / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return root

    @classmethod
    def load(cls, out_dir: str | Path) -> "MedianBackgroundModel":
        root = Path(out_dir)
        meta = json.loads((root / "config.json").read_text(encoding="utf-8"))
        arrays = np.load(root / "background.npz")
        return cls(
            background_rgb=arrays["background_rgb"],
            mad_rgb=arrays["mad_rgb"],
            threshold=meta["threshold"],
            dilate=meta["dilate"],
            use_adaptive_scale=meta["use_adaptive_scale"],
            camera_id=meta["camera_id"],
        )


class StaticMaskControl:
    """Per-camera static positional control (Step 2 & review shortcut check).

    Outputs a fixed binary mask estimated solely from training occupancy,
    independent of the current frame, to detect fixed-camera shortcuts.
    """

    def __init__(self, occupancy: np.ndarray, threshold: float = 0.5, camera_id: str = "dev2"):
        self.occupancy = np.asarray(occupancy, dtype=np.float32)
        self.threshold = float(threshold)
        self.camera_id = camera_id
        self._mask = (self.occupancy >= self.threshold).astype(np.uint8)

    @classmethod
    def fit_from_masks(
        cls, masks: Sequence[np.ndarray], threshold: float = 0.5, camera_id: str = "dev2"
    ) -> "StaticMaskControl":
        stack = np.stack([np.asarray(m, dtype=np.float32) > 0 for m in masks], axis=0)
        return cls(occupancy=stack.mean(axis=0), threshold=threshold, camera_id=camera_id)

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        if self._mask.shape != (h, w):
            return cv2.resize(self._mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return self._mask.copy()

    def segment_timed(
        self, frame_rgb: np.ndarray, frame_id: int | None = None, capture_ts: float | None = None
    ) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        mask = self.segment(frame_rgb)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        return mask, {
            "frame_id": frame_id,
            "capture_ts": capture_ts,
            "deliver_ts": t1,
            "preprocess_ms": 0.0,
            "inference_ms": ms,
            "transfer_ms": 0.0,
            "resize_ms": 0.0,
            "total_ms": ms,
        }

    def save(self, out_dir: str | Path) -> Path:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "static_mask.npz", occupancy=self.occupancy)
        meta = {
            "kind": "static_control",
            "camera_id": self.camera_id,
            "threshold": self.threshold,
        }
        (root / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return root

    @classmethod
    def load(cls, out_dir: str | Path) -> "StaticMaskControl":
        root = Path(out_dir)
        meta = json.loads((root / "config.json").read_text(encoding="utf-8"))
        arrays = np.load(root / "static_mask.npz")
        return cls(
            occupancy=arrays["occupancy"],
            threshold=meta["threshold"],
            camera_id=meta["camera_id"],
        )
