"""Step 3 offline teachers: motion partial labeling, SAM 2.1, and SAM 3.

All teachers save raw outputs separately from filtered 0/1/255 training labels
and report retained frame coverage broken down by moving vs. paused frames.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .baselines import MedianBackgroundModel
from .data import (
    BG,
    FG,
    UNKNOWN,
    detect_pause_flags,
    frame_motion_scores,
    render_mask_overlay,
    save_label_png,
    save_rgb_png,
)


@dataclass
class TeacherEpisodeResult:
    episode: str
    camera: str
    teacher_name: str
    frame_indices: list[int]
    raw_masks: np.ndarray  # uint8 [N, H, W] (0/1 or 0/1/255)
    labels: np.ndarray  # uint8 [N, H, W] (BG=0, FG=1, UNKNOWN=255)
    retained: np.ndarray  # bool [N]
    pause_flags: np.ndarray  # bool [N]
    metadata: dict


def _farneback_flow(gray_a: np.ndarray, gray_b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=13,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )


def _edge_strength(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _propagate_seed_with_flow(
    src_seed: np.ndarray,
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    dst_edges: np.ndarray,
    roundtrip_max_px: float = 1.5,
    edge_barrier: float = 90.0,
) -> np.ndarray:
    """Propagate ``src_seed`` onto ``dst`` frame, rejecting round-trip/edge/disocclusion failures."""
    h, w = src_gray.shape
    flow_dst_to_src = _farneback_flow(dst_gray, src_gray)
    flow_src_to_dst = _farneback_flow(src_gray, dst_gray)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow_dst_to_src[..., 0]
    map_y = yy + flow_dst_to_src[..., 1]

    in_bounds = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    # Sample reverse flow at mapped source coordinates for round-trip consistency
    rev_x = cv2.remap(flow_src_to_dst[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rev_y = cv2.remap(flow_src_to_dst[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rt_err = np.hypot(flow_dst_to_src[..., 0] + rev_x, flow_dst_to_src[..., 1] + rev_y)

    # Disocclusion check via divergence of backward flow
    du_dx = np.gradient(flow_dst_to_src[..., 0], axis=1)
    dv_dy = np.gradient(flow_dst_to_src[..., 1], axis=0)
    div = du_dx + dv_dy

    warped_seed = cv2.remap(
        src_seed.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ) > 0.5

    # Check whether the correspondence crosses a strong image edge at destination
    valid = in_bounds & (rt_err <= roundtrip_max_px) & (div > -0.5) & (dst_edges < edge_barrier)
    return warped_seed & valid


class MotionLabelTeacher:
    """Offline partial-label generator from multi-offset differences + bidirectional optical flow.

    Produces ternary labels:
      - BG (0): stable background outside the robot's swept motion region
      - FG (1): locally coherent moving foreground agreeing with background deviation
        (plus round-trip verified propagation from neighboring frames)
      - UNKNOWN (255): ambiguous regions (including stationary parts inside swept area;
        absence of motion alone never produces a BG label inside the swept area).
    """

    def __init__(
        self,
        bg_model: MedianBackgroundModel,
        offsets: Sequence[int] = (1, 3, 8),
        diff_threshold: float = 12.0,
        flow_min_px: float = 0.6,
        bg_diff_threshold: float = 0.10,
        bg_safe_threshold: float = 0.05,
        propagate_steps: Sequence[int] = (-2, -1, 1, 2),
        roundtrip_max_px: float = 1.5,
        edge_barrier: float = 95.0,
        swept_dilate_px: int = 9,
        min_fg_pixels: int = 40,
    ):
        self.bg_model = bg_model
        self.offsets = tuple(int(o) for o in offsets)
        self.diff_threshold = float(diff_threshold)
        self.flow_min_px = float(flow_min_px)
        self.bg_diff_threshold = float(bg_diff_threshold)
        self.bg_safe_threshold = float(bg_safe_threshold)
        self.propagate_steps = tuple(int(s) for s in propagate_steps)
        self.roundtrip_max_px = float(roundtrip_max_px)
        self.edge_barrier = float(edge_barrier)
        self.swept_dilate_px = int(swept_dilate_px)
        self.min_fg_pixels = int(min_fg_pixels)

    def label_episode(
        self,
        frames_rgb: np.ndarray,
        frame_indices: Sequence[int],
        episode: str,
        camera: str,
    ) -> TeacherEpisodeResult:
        n, h, w = frames_rgb.shape[:3]
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb]
        edges = [_edge_strength(g) for g in grays]
        bg_diffs = np.stack([self.bg_model.diff_map(f) for f in frames_rgb], axis=0)

        # 1. Seed moving foreground where multi-offset differences + coherent optical flow + bg diff agree
        seeds = np.zeros((n, h, w), dtype=bool)
        morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for t in range(n):
            multi_diff = np.zeros((h, w), dtype=np.float32)
            for off in self.offsets:
                if t - off >= 0:
                    multi_diff = np.maximum(multi_diff, np.abs(grays[t].astype(np.float32) - grays[t - off].astype(np.float32)))
                if t + off < n:
                    multi_diff = np.maximum(multi_diff, np.abs(grays[t].astype(np.float32) - grays[t + off].astype(np.float32)))

            flow_mag = np.zeros((h, w), dtype=np.float32)
            if t + 1 < n:
                f_fwd = _farneback_flow(grays[t], grays[t + 1])
                flow_mag = np.maximum(flow_mag, np.hypot(f_fwd[..., 0], f_fwd[..., 1]))
            if t - 1 >= 0:
                f_bwd = _farneback_flow(grays[t], grays[t - 1])
                flow_mag = np.maximum(flow_mag, np.hypot(f_bwd[..., 0], f_bwd[..., 1]))

            # Local coherence: smooth motion magnitude slightly so isolated noise drops out
            coherent_flow = cv2.GaussianBlur(flow_mag, (5, 5), 0) >= self.flow_min_px
            raw_seed = (
                (multi_diff >= self.diff_threshold)
                & coherent_flow
                & (bg_diffs[t] >= self.bg_diff_threshold)
            )
            cleaned = cv2.morphologyEx(raw_seed.astype(np.uint8), cv2.MORPH_OPEN, morph) > 0
            seeds[t] = cleaned

        # 2. Propagate seeds from neighboring frames onto current frame's visible surfaces
        propagated = seeds.copy()
        for t in range(n):
            votes = seeds[t].astype(np.uint8) * 2
            for dt in self.propagate_steps:
                src = t + dt
                if 0 <= src < n and seeds[src].any():
                    prop = _propagate_seed_with_flow(
                        seeds[src],
                        grays[src],
                        grays[t],
                        edges[t],
                        roundtrip_max_px=self.roundtrip_max_px,
                        edge_barrier=self.edge_barrier,
                    )
                    # Require propagated pixel to still disagree with background
                    votes += (prop & (bg_diffs[t] >= self.bg_safe_threshold)).astype(np.uint8)
            propagated[t] = votes >= 2

        # 3. Swept motion envelope: outside this + matching median bg -> confident BG; inside ambiguous -> UNKNOWN
        swept = propagated.any(axis=0).astype(np.uint8)
        if self.swept_dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.swept_dilate_px + 1, 2 * self.swept_dilate_px + 1)
            )
            swept_zone = cv2.dilate(swept, k, iterations=1) > 0
        else:
            swept_zone = swept > 0

        boundary_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        labels = np.full((n, h, w), UNKNOWN, dtype=np.uint8)
        raw_masks = propagated.astype(np.uint8)
        retained = np.zeros(n, dtype=bool)

        for t in range(n):
            fg = propagated[t]
            # Confident BG outside swept zone where RGB agrees with median background
            confident_bg = (~swept_zone) & (bg_diffs[t] < self.bg_safe_threshold)
            # Also inside swept zone, pixels far from current FG AND far from background deviation
            # are only BG if clearly unchanged across the whole neighborhood and matching bg_model;
            # keep a safety halo around FG and around any bg_diff > bg_safe_threshold as UNKNOWN.
            fg_halo = cv2.dilate(fg.astype(np.uint8), boundary_k, iterations=2) > 0
            bg_halo = cv2.dilate((bg_diffs[t] >= self.bg_safe_threshold).astype(np.uint8), boundary_k, iterations=2) > 0
            # Note: stationary robot base inside swept_zone is NOT marked BG if it deviates or could be static;
            # keep swept_zone & ~fg as UNKNOWN unless strictly clean background outside swept_zone.
            lbl = np.full((h, w), UNKNOWN, dtype=np.uint8)
            lbl[confident_bg & (~fg_halo) & (~bg_halo)] = BG
            lbl[fg] = FG
            # Thin boundary ring around FG marked UNKNOWN to avoid bleeding shadow/blur into loss
            eroded = cv2.erode(fg.astype(np.uint8), boundary_k, iterations=1) > 0
            fg_border = fg & (~eroded)
            # Keep thin jaws (where erosion would erase small structures) by only marking outer halo unknown
            outer_halo = fg_halo & (~fg)
            lbl[outer_halo] = UNKNOWN
            _ = fg_border  # preserved as FG so thin jaws are not erased
            labels[t] = lbl
            retained[t] = int(fg.sum()) >= self.min_fg_pixels

        motion = frame_motion_scores(frames_rgb)
        pauses, pause_thr = detect_pause_flags(motion)
        return TeacherEpisodeResult(
            episode=episode,
            camera=camera,
            teacher_name="motion",
            frame_indices=list(frame_indices),
            raw_masks=raw_masks,
            labels=labels,
            retained=retained,
            pause_flags=pauses,
            metadata={
                "offsets": list(self.offsets),
                "diff_threshold": self.diff_threshold,
                "flow_min_px": self.flow_min_px,
                "bg_diff_threshold": self.bg_diff_threshold,
                "bg_safe_threshold": self.bg_safe_threshold,
                "pause_threshold": pause_thr,
            },
        )


def select_motion_anchors(
    motion_labels: np.ndarray,
    max_anchors: int = 4,
    min_fg_pixels: int = 80,
    n_pos_points: int = 5,
    n_neg_points: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Choose anchor frames with clean motion foreground and sample positive/negative prompts + box."""
    rng = np.random.default_rng(seed)
    n = len(motion_labels)
    fg_counts = (motion_labels == FG).sum(axis=(1, 2))
    eligible = np.flatnonzero(fg_counts >= min_fg_pixels)
    if len(eligible) == 0:
        raise ValueError("No frames have enough motion foreground pixels to anchor SAM 2")

    # Spread anchors across the episode
    pick_pos = np.linspace(0, len(eligible) - 1, min(max_anchors, len(eligible))).round().astype(int)
    anchor_frames = sorted(set(int(eligible[p]) for p in pick_pos))

    swept = (motion_labels == FG).any(axis=0)
    prompts = []
    for fidx in anchor_frames:
        fg = motion_labels[fidx] == FG
        bg = (motion_labels[fidx] == BG) & (~swept)
        if not bg.any():
            bg = ~swept
        # Prefer interior pixels via distance transform for positive prompts
        dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.where(fg)
        order = np.argsort(dist[ys, xs])[::-1]
        top = order[: max(n_pos_points * 4, len(order))]
        pos_sel = rng.choice(top, size=min(n_pos_points, len(top)), replace=False)
        pos_pts = [[float(xs[i]), float(ys[i])] for i in pos_sel]

        bys, bxs = np.where(bg)
        neg_sel = rng.choice(len(bxs), size=min(n_neg_points, len(bxs)), replace=False)
        neg_pts = [[float(bxs[i]), float(bys[i])] for i in neg_sel]

        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        prompts.append(
            {
                "frame": int(fidx),
                "points": pos_pts + neg_pts,
                "labels": [1] * len(pos_pts) + [0] * len(neg_pts),
                "box": box,
            }
        )
    return prompts


class SAM2Teacher:
    """Offline SAM 2.1 video teacher prompted by motion anchors (forward + backward agreement)."""

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str | Path,
        device: str = "cpu",
        use_box: bool = True,
        max_anchors: int = 4,
        min_fg_pixels: int = 50,
    ):
        self.model_cfg = model_cfg
        self.checkpoint = str(checkpoint)
        self.device = device
        self.use_box = bool(use_box)
        self.max_anchors = int(max_anchors)
        self.min_fg_pixels = int(min_fg_pixels)

    def label_episode(
        self,
        frames_rgb: np.ndarray,
        frame_indices: Sequence[int],
        motion_result: TeacherEpisodeResult,
        episode: str,
        camera: str,
        seed: int = 0,
    ) -> TeacherEpisodeResult:
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        prompts = select_motion_anchors(
            motion_result.labels, max_anchors=self.max_anchors, seed=seed
        )
        predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)
        n, h, w = frames_rgb.shape[:3]
        fwd_masks = np.zeros((n, h, w), dtype=bool)
        bwd_masks = np.zeros((n, h, w), dtype=bool)

        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmpdir:
            for t, rgb in enumerate(frames_rgb):
                cv2.imwrite(str(Path(tmpdir) / f"{t:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            with torch.inference_mode():
                state = predictor.init_state(video_path=tmpdir)
                for p in prompts:
                    kwargs = dict(
                        inference_state=state,
                        frame_idx=p["frame"],
                        obj_id=1,
                        points=np.array(p["points"], dtype=np.float32),
                        labels=np.array(p["labels"], dtype=np.int32),
                    )
                    if self.use_box:
                        kwargs["box"] = np.array(p["box"], dtype=np.float32)
                    predictor.add_new_points_or_box(**kwargs)

                for t, _, logits in predictor.propagate_in_video(state, reverse=False):
                    fwd_masks[t] = logits[0, 0].cpu().numpy() > 0
                for t, _, logits in predictor.propagate_in_video(state, reverse=True):
                    bwd_masks[t] = logits[0, 0].cpu().numpy() > 0

        agree_fg = fwd_masks & bwd_masks
        agree_bg = (~fwd_masks) & (~bwd_masks)
        labels = np.full((n, h, w), UNKNOWN, dtype=np.uint8)
        labels[agree_bg] = BG
        labels[agree_fg] = FG
        raw_masks = (fwd_masks | bwd_masks).astype(np.uint8)
        retained = np.array([int(agree_fg[t].sum()) >= self.min_fg_pixels for t in range(n)], dtype=bool)

        motion = frame_motion_scores(frames_rgb)
        pauses, pause_thr = detect_pause_flags(motion)
        return TeacherEpisodeResult(
            episode=episode,
            camera=camera,
            teacher_name="sam2",
            frame_indices=list(frame_indices),
            raw_masks=raw_masks,
            labels=labels,
            retained=retained,
            pause_flags=pauses,
            metadata={
                "model_cfg": self.model_cfg,
                "checkpoint": self.checkpoint,
                "use_box": self.use_box,
                "prompts": prompts,
                "pause_threshold": pause_thr,
            },
        )


class SAM3DirectSegmenter:
    """Single-frame SAM 3 / SAM 3.1 predictor implementing ``segment(frame_rgb) -> mask``.

    Defaults to Ultralytics ``SAM3SemanticPredictor`` (which runs ``sam3.1_multiplex.pt``
    and ``sam3.pt`` on CUDA, Apple MPS, or CPU). Also supports ``backend='meta_sam3'``
    if the upstream ``facebookresearch/sam3`` package is installed.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        prompt: str = "robot arm",
        conf: float = 0.25,
        imgsz: int = 644,
        device: str = "auto",
        camera_id: str = "dev2",
        backend: str = "ultralytics",
    ):
        import torch

        self.checkpoint = str(checkpoint)
        self.prompt = prompt
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.camera_id = camera_id
        self.backend = backend
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = str(device)

        if backend == "ultralytics":
            from ultralytics.models.sam.predict import SAM3SemanticPredictor

            overrides = dict(
                conf=self.conf,
                task="segment",
                mode="predict",
                model=self.checkpoint,
                imgsz=self.imgsz,
                device=self.device,
                half=False,
                save=False,
                verbose=False,
            )
            self._predictor = SAM3SemanticPredictor(overrides=overrides)
        elif backend == "meta_sam3":
            from PIL import Image
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model

            self._pil_image = Image
            self._model = build_sam3_image_model(checkpoint_path=self.checkpoint)
            self._processor = Sam3Processor(self._model)
        else:
            raise ValueError(f"Unknown SAM3 backend: {backend}")

    def _sync(self) -> None:
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        mask, _ = self.segment_timed(frame_rgb)
        return mask

    def segment_timed(
        self, frame_rgb: np.ndarray, frame_id: int | None = None, capture_ts: float | None = None
    ) -> tuple[np.ndarray, dict]:
        import torch

        h, w = frame_rgb.shape[:2]
        t0 = time.perf_counter()
        if self.backend == "ultralytics":
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            t1 = time.perf_counter()
            self._predictor.set_image(bgr)
            results = self._predictor(text=[self.prompt])
            self._sync()
            t2 = time.perf_counter()
            res = results[0]
            if res.masks is None or len(res.masks.data) == 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                t3 = t2
                t4 = time.perf_counter()
            else:
                raw = (res.masks.data.amax(dim=0) > 0.5).cpu().numpy().astype(np.uint8)
                t3 = time.perf_counter()
                mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST) if raw.shape != (h, w) else raw
                t4 = time.perf_counter()
        else:
            pil = self._pil_image.fromarray(frame_rgb)
            t1 = time.perf_counter()
            with torch.inference_mode():
                state = self._processor.set_image(pil)
                out = self._processor.set_text_prompt(state=state, prompt=self.prompt)
            self._sync()
            t2 = time.perf_counter()
            masks = out.get("masks", [])
            if len(masks) == 0:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                combined = np.any(
                    np.asarray([m.cpu().numpy() if hasattr(m, "cpu") else m for m in masks]) > 0, axis=0
                )
                if combined.ndim == 3:
                    combined = combined.any(axis=0)
                mask = combined.astype(np.uint8)
            t3 = time.perf_counter()
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            t4 = time.perf_counter()

        return mask, {
            "frame_id": frame_id,
            "capture_ts": capture_ts,
            "deliver_ts": t4,
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "transfer_ms": (t3 - t2) * 1000.0,
            "resize_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
        }

    def save(self, out_dir: str | Path) -> Path:
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": "sam3_direct",
            "checkpoint": self.checkpoint,
            "prompt": self.prompt,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "camera_id": self.camera_id,
            "backend": self.backend,
        }
        (root / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return root


class SAM3Teacher:
    """Offline text-prompted SAM 3 / SAM 3.1 teacher with motion identity cross-check.

    Uses Ultralytics ``SAM3SemanticPredictor`` by default (compatible with
    ``sam3.1_multiplex.pt`` and ``sam3.pt`` on CUDA, MPS, or CPU) and falls back
    to the upstream ``sam3`` video predictor when ``backend='meta_sam3'``.
    """

    def __init__(
        self,
        prompt: str = "robot arm",
        checkpoint: str | Path | None = None,
        conf: float = 0.25,
        imgsz: int = 644,
        device: str = "auto",
        min_motion_overlap: float = 0.05,
        min_fg_pixels: int = 50,
        backend: str = "ultralytics",
        segmenter: SAM3DirectSegmenter | None = None,
    ):
        self.prompt = prompt
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.min_motion_overlap = float(min_motion_overlap)
        self.min_fg_pixels = int(min_fg_pixels)
        self.backend = backend
        self._segmenter = segmenter

    def _get_segmenter(self, camera: str) -> SAM3DirectSegmenter:
        if self._segmenter is None:
            if self.checkpoint is None:
                raise ValueError("SAM3Teacher requires a checkpoint path (e.g. sam3.1_multiplex.pt)")
            self._segmenter = SAM3DirectSegmenter(
                checkpoint=self.checkpoint,
                prompt=self.prompt,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                camera_id=camera,
                backend=self.backend,
            )
        return self._segmenter

    def label_episode(
        self,
        frames_rgb: np.ndarray,
        frame_indices: Sequence[int],
        motion_result: TeacherEpisodeResult | None,
        episode: str,
        camera: str,
    ) -> TeacherEpisodeResult:
        n, h, w = frames_rgb.shape[:3]
        raw_masks = np.zeros((n, h, w), dtype=np.uint8)

        if self.backend == "meta_sam3_video":
            import torch
            from sam3.model_builder import build_sam3_video_predictor

            predictor = build_sam3_video_predictor(checkpoint_path=self.checkpoint)
            with tempfile.TemporaryDirectory(prefix="sam3_frames_") as tmpdir:
                for t, rgb in enumerate(frames_rgb):
                    cv2.imwrite(str(Path(tmpdir) / f"{t:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                with torch.inference_mode():
                    session = predictor.handle_request({"type": "start_session", "resource_path": tmpdir})
                    sid = session["session_id"]
                    predictor.handle_request(
                        {"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": self.prompt}
                    )
                    for resp in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
                        t = int(resp["frame_index"])
                        masks = resp["outputs"]["out_binary_masks"]
                        if len(masks) > 0:
                            raw_masks[t] = (np.any(np.asarray(masks) > 0, axis=0)).astype(np.uint8)
                    predictor.handle_request({"type": "close_session", "session_id": sid})
        else:
            seg = self._get_segmenter(camera)
            for t in range(n):
                raw_masks[t] = seg.segment(frames_rgb[t])

        # Cross-check identity against motion while allowing static parts/pauses to survive
        motion = frame_motion_scores(frames_rgb)
        pauses, pause_thr = detect_pause_flags(motion)
        labels = np.where(raw_masks > 0, FG, BG).astype(np.uint8)
        retained = np.zeros(n, dtype=bool)
        motion_by_fidx = (
            {int(fi): i for i, fi in enumerate(motion_result.frame_indices)} if motion_result is not None else {}
        )
        for t in range(n):
            fg = raw_masks[t] > 0
            if int(fg.sum()) < self.min_fg_pixels:
                labels[t] = UNKNOWN
                continue
            fidx = int(frame_indices[t])
            if motion_result is not None and (not pauses[t]) and (fidx in motion_by_fidx):
                mfg = motion_result.labels[motion_by_fidx[fidx]] == FG
                if mfg.any():
                    overlap = float((fg & mfg).sum()) / float(mfg.sum())
                    if overlap < self.min_motion_overlap:
                        # Identity likely jumped to table/other object
                        labels[t] = UNKNOWN
                        continue
            retained[t] = True

        return TeacherEpisodeResult(
            episode=episode,
            camera=camera,
            teacher_name="sam3",
            frame_indices=list(frame_indices),
            raw_masks=raw_masks,
            labels=labels,
            retained=retained,
            pause_flags=pauses,
            metadata={
                "prompt": self.prompt,
                "checkpoint": self.checkpoint,
                "conf": self.conf,
                "imgsz": self.imgsz,
                "backend": self.backend,
                "min_motion_overlap": self.min_motion_overlap,
                "pause_threshold": pause_thr,
            },
        )


def save_teacher_outputs(
    result: TeacherEpisodeResult,
    frames_rgb: np.ndarray,
    out_dir: str | Path,
    save_overlays_every: int = 25,
) -> dict:
    """Save raw teacher masks separately from filtered 0/1/255 training labels + coverage summary."""
    root = Path(out_dir) / result.teacher_name / result.camera / result.episode
    raw_dir = root / "raw_masks"
    lbl_dir = root / "training_labels"
    ovl_dir = root / "overlays"
    raw_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ovl_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, fidx in enumerate(result.frame_indices):
        stem = f"frame_{fidx:06d}.png"
        save_label_png(raw_dir / stem, result.raw_masks[i])
        save_label_png(lbl_dir / stem, result.labels[i])
        if save_overlays_every > 0 and (i % save_overlays_every == 0):
            save_rgb_png(ovl_dir / stem, render_mask_overlay(frames_rgb[i], result.labels[i]))
        lbl = result.labels[i]
        records.append(
            {
                "frame_idx": int(fidx),
                "retained": bool(result.retained[i]),
                "is_pause": bool(result.pause_flags[i]),
                "fg_pixels": int((lbl == FG).sum()),
                "bg_pixels": int((lbl == BG).sum()),
                "unknown_pixels": int((lbl == UNKNOWN).sum()),
            }
        )

    pauses = result.pause_flags
    moving = ~pauses
    summary = {
        "episode": result.episode,
        "camera": result.camera,
        "teacher": result.teacher_name,
        "total_frames": len(result.frame_indices),
        "retained_frames": int(result.retained.sum()),
        "retained_fraction": float(result.retained.mean()) if len(result.retained) else 0.0,
        "moving_total": int(moving.sum()),
        "moving_retained": int((result.retained & moving).sum()),
        "pause_total": int(pauses.sum()),
        "pause_retained": int((result.retained & pauses).sum()),
        "metadata": result.metadata,
        "frames": records,
    }
    (root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
