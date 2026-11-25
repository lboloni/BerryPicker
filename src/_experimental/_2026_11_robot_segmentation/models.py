"""Single-frame segmentation models, YOLO polygon round-trip audit, and runtime predictor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import MedianBackgroundModel, StaticMaskControl


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def letterbox_rgb(
    frame_rgb: np.ndarray,
    input_size: int | tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, dict]:
    """Aspect-ratio preserving resize + pad to ``(target_h, target_w)``."""
    h, w = frame_rgb.shape[:2]
    if isinstance(input_size, int):
        th, tw = input_size, input_size
    else:
        th, tw = int(input_size[0]), int(input_size[1])

    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=interp)

    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized
    meta = {
        "orig_h": h,
        "orig_w": w,
        "target_h": th,
        "target_w": tw,
        "new_h": nh,
        "new_w": nw,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return canvas, meta


def unletterbox_mask(mask_2d: np.ndarray, meta: dict) -> np.ndarray:
    """Remove letterbox padding and resize back to original camera coordinates."""
    top, left = meta["pad_top"], meta["pad_left"]
    cropped = mask_2d[top : top + meta["new_h"], left : left + meta["new_w"]]
    if cropped.dtype == np.uint8:
        return cv2.resize(cropped, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(cropped.astype(np.float32), (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallSegUNet(nn.Module):
    """Compact from-scratch encoder-decoder with skip connections (1 RGB in -> 1 logit out)."""

    def __init__(self, base_channels: int = 24):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.enc1 = _ConvBlock(3, c1)
        self.enc2 = _ConvBlock(c1, c2)
        self.enc3 = _ConvBlock(c2, c3)
        self.bottleneck = _ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.dec3 = _ConvBlock(c4 + c3, c3)
        self.dec2 = _ConvBlock(c3 + c2, c2)
        self.dec1 = _ConvBlock(c2 + c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return self.head(d1)


class YOLO26SemanticNet(nn.Module):
    """YOLO26 nano backbone/neck + lightweight high-res skip decoder for 1-channel semantic logits.

    Supports both pretrained init (e.g. ``yolo26n-seg.pt``) and random init
    (``yolo26n-seg.yaml``) while training directly on ``0/1/255`` PNG masks so
    unknown pixels are ignored and thin jaw gaps/holes are preserved.
    """

    TAP_LAYERS = (0, 2, 16, 19, 22)  # P1/2, P2/4, head P3/8, head P4/16, head P5/32

    def __init__(self, weights_or_cfg: str = "yolo26n-seg.yaml", decoder_channels: int = 32):
        super().__init__()
        from ultralytics import YOLO

        yolo = YOLO(weights_or_cfg)
        full_model = yolo.model
        # Keep backbone + PAN neck (layers 0..22), drop final Segment26/Detect head
        self.layers = nn.ModuleList(list(full_model.model)[:23])
        self.save_set = set(full_model.save) | set(self.TAP_LAYERS)
        self.weights_or_cfg = weights_or_cfg
        self.decoder_channels = int(decoder_channels)

        # Probe tap channel widths dynamically
        with torch.no_grad():
            taps = self._extract_taps(torch.zeros(1, 3, 128, 128))
        ch = [t.shape[1] for t in taps]
        dc = self.decoder_channels
        self.lat = nn.ModuleList([nn.Conv2d(c, dc, kernel_size=1, bias=False) for c in ch])
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(dc, dc, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(dc),
                    nn.SiLU(inplace=True),
                )
                for _ in range(len(ch))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(dc, dc // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dc // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(dc // 2, 1, kernel_size=1),
        )

    def _extract_taps(self, x: torch.Tensor) -> list[torch.Tensor]:
        y: list[torch.Tensor | None] = []
        for idx, m in enumerate(self.layers):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if idx in self.save_set else None)
        return [y[i] for i in self.TAP_LAYERS]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_hw = x.shape[-2:]
        taps = self._extract_taps(x)
        # Top-down FPN fusion from P5/32 down to P1/2
        p = self.smooth[-1](self.lat[-1](taps[-1]))
        for i in range(len(taps) - 2, -1, -1):
            p = F.interpolate(p, size=taps[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p + self.lat[i](taps[i]))
        logits = self.head(p)
        return F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)


def build_semantic_model(arch: str, **kwargs: Any) -> nn.Module:
    if arch == "unet":
        return SmallSegUNet(base_channels=int(kwargs.get("base_channels", 24)))
    if arch == "yolo26_semantic":
        return YOLO26SemanticNet(
            weights_or_cfg=kwargs.get("weights_or_cfg", "yolo26n-seg.yaml"),
            decoder_channels=int(kwargs.get("decoder_channels", 32)),
        )
    raise ValueError(f"Unknown semantic architecture: {arch}")


# ---------------------------------------------------------------------------
# Ultralytics YOLO26 instance segmentation helpers + polygon round-trip audit
# ---------------------------------------------------------------------------

def mask_to_yolo_polygons(
    binary_mask: np.ndarray,
    class_id: int = 0,
    approx_eps_px: float = 1.0,
    min_area_px: float = 10.0,
) -> list[str]:
    """Convert a binary robot mask to YOLO polygon label lines (external contours)."""
    h, w = binary_mask.shape[:2]
    contours, _ = cv2.findContours(
        (binary_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        if approx_eps_px > 0:
            cnt = cv2.approxPolyDP(cnt, approx_eps_px, closed=True)
        pts = cnt.reshape(-1, 2)
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f"{x / w:.6f}")
            coords.append(f"{y / h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    return lines


def rasterize_yolo_polygons(lines: list[str], height: int, width: int) -> np.ndarray:
    """Rasterize YOLO polygon label lines back into a uint8 binary mask."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        vals = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        pts = np.stack([np.rint(vals[:, 0] * width), np.rint(vals[:, 1] * height)], axis=1).astype(np.int32)
        cv2.fillPoly(canvas, [pts], 1)
    return canvas


def audit_polygon_roundtrip(masks: list[np.ndarray], approx_eps_px: float = 1.0) -> dict:
    """Measure lost holes and thin-part distortions caused by YOLO polygon conversion."""
    ious = []
    hole_fill_px_list = []
    lost_fg_px_list = []
    frames_with_holes = 0
    for m in masks:
        gt = (m > 0).astype(np.uint8)
        h, w = gt.shape
        lines = mask_to_yolo_polygons(gt, approx_eps_px=approx_eps_px)
        rt = rasterize_yolo_polygons(lines, h, w)
        inter = int((gt & rt).sum())
        union = int((gt | rt).sum())
        ious.append(inter / union if union > 0 else 1.0)
        # Pixels that were 0 inside external contour (e.g. jaw holes) and got filled to 1
        filled_holes = int(((gt == 0) & (rt == 1)).sum())
        lost_fg = int(((gt == 1) & (rt == 0)).sum())
        if filled_holes > 0:
            frames_with_holes += 1
        hole_fill_px_list.append(filled_holes)
        lost_fg_px_list.append(lost_fg)
    return {
        "n_masks": len(masks),
        "approx_eps_px": approx_eps_px,
        "mean_roundtrip_iou": float(np.mean(ious)) if ious else 1.0,
        "min_roundtrip_iou": float(np.min(ious)) if ious else 1.0,
        "frames_with_filled_holes": frames_with_holes,
        "mean_filled_hole_px": float(np.mean(hole_fill_px_list)) if hole_fill_px_list else 0.0,
        "max_filled_hole_px": int(np.max(hole_fill_px_list)) if hole_fill_px_list else 0,
        "mean_lost_fg_px": float(np.mean(lost_fg_px_list)) if lost_fg_px_list else 0.0,
    }


class YOLO26InstancePredictor:
    """Runtime wrapper around a fine-tuned Ultralytics ``yolo26n-seg`` model."""

    def __init__(
        self,
        weights: str | Path,
        input_size: int = 256,
        conf: float = 0.25,
        device: str = "auto",
        camera_id: str = "dev2",
    ):
        from ultralytics import YOLO

        self.weights = str(weights)
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.device = resolve_device(device)
        self.camera_id = camera_id
        self.model = YOLO(self.weights)

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        mask, _ = self.segment_timed(frame_rgb)
        return mask

    def segment_timed(
        self, frame_rgb: np.ndarray, frame_id: int | None = None, capture_ts: float | None = None
    ) -> tuple[np.ndarray, dict]:
        h, w = frame_rgb.shape[:2]
        t0 = time.perf_counter()
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        t1 = time.perf_counter()
        results = self.model.predict(
            source=bgr,
            imgsz=self.input_size,
            conf=self.conf,
            device=str(self.device),
            verbose=False,
        )
        sync_device(self.device)
        t2 = time.perf_counter()
        res = results[0]
        if res.masks is None or len(res.masks.data) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)
            t3 = t2
            t4 = time.perf_counter()
        else:
            raw = (res.masks.data.amax(dim=0) > 0.5).cpu().numpy().astype(np.uint8)
            t3 = time.perf_counter()
            mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
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


# ---------------------------------------------------------------------------
# Unified single-frame runtime predictor & checkpoint persistence
# ---------------------------------------------------------------------------

class RobotSegmenter:
    """Single-frame predictor exposing ``segment(frame_rgb) -> mask`` at original resolution."""

    def __init__(
        self,
        model: nn.Module,
        input_size: int | tuple[int, int] = 256,
        threshold: float = 0.5,
        device: str | torch.device = "auto",
        camera_id: str = "dev2",
        metadata: dict | None = None,
    ):
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self.input_size = input_size
        self.threshold = float(threshold)
        self.camera_id = camera_id
        self.metadata = metadata or {}

    @torch.inference_mode()
    def segment_prob(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return float32 foreground probability map in [0, 1] at original resolution."""
        canvas, meta = letterbox_rgb(frame_rgb, self.input_size)
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(self.device)
        logits = self.model(tensor)[0, 0]
        probs = torch.sigmoid(logits).cpu().numpy()
        return unletterbox_mask(probs, meta)

    @torch.inference_mode()
    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return uint8 binary mask (0=bg, 1=robot) at original ``frame_rgb`` resolution."""
        mask, _ = self.segment_timed(frame_rgb)
        return mask

    @torch.inference_mode()
    def segment_timed(
        self,
        frame_rgb: np.ndarray,
        frame_id: int | None = None,
        capture_ts: float | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run single-frame segmentation with synchronized per-stage timing breakdown."""
        t0 = time.perf_counter()
        canvas, meta = letterbox_rgb(frame_rgb, self.input_size)
        tensor_cpu = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        t1 = time.perf_counter()

        tensor = tensor_cpu.to(self.device)
        sync_device(self.device)
        t2 = time.perf_counter()

        logits = self.model(tensor)[0, 0]
        pred_small = (torch.sigmoid(logits) >= self.threshold).to(torch.uint8)
        sync_device(self.device)
        t3 = time.perf_counter()

        pred_np = pred_small.cpu().numpy()
        t4 = time.perf_counter()

        mask = unletterbox_mask(pred_np, meta)
        t5 = time.perf_counter()

        return mask, {
            "frame_id": frame_id,
            "capture_ts": capture_ts,
            "deliver_ts": t5,
            "preprocess_ms": (t1 - t0) * 1000.0,
            "transfer_ms": ((t2 - t1) + (t4 - t3)) * 1000.0,
            "inference_ms": (t3 - t2) * 1000.0,
            "resize_ms": (t5 - t4) * 1000.0,
            "total_ms": (t5 - t0) * 1000.0,
        }


def save_semantic_checkpoint(
    out_dir: str | Path,
    model: nn.Module,
    arch: str,
    arch_kwargs: dict,
    camera_id: str,
    input_size: int | tuple[int, int],
    threshold: float,
    split: dict | None = None,
    training_settings: dict | None = None,
) -> Path:
    """Save weights together with camera ID, resizing/RGB conventions, threshold, split, and config."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / "weights.pt")
    meta = {
        "kind": "semantic_torch",
        "arch": arch,
        "arch_kwargs": arch_kwargs,
        "camera_id": camera_id,
        "input_size": input_size,
        "rgb_convention": "RGB_uint8_div_255_letterbox_114",
        "threshold": float(threshold),
        "split": split,
        "training_settings": training_settings or {},
    }
    (root / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


def load_segmenter(artifact_dir: str | Path, device: str = "auto") -> Any:
    """Load any saved predictor artifact (semantic model, YOLO instance, median bg, static control)."""
    root = Path(artifact_dir)
    meta = json.loads((root / "config.json").read_text(encoding="utf-8"))
    kind = meta["kind"]
    if kind == "semantic_torch":
        arch_kwargs = dict(meta["arch_kwargs"])
        # Always instantiate structure without re-downloading external weights when loading a trained checkpoint
        if meta["arch"] == "yolo26_semantic" and str(arch_kwargs.get("weights_or_cfg", "")).endswith(".pt"):
            arch_kwargs["weights_or_cfg"] = "yolo26n-seg.yaml"
        model = build_semantic_model(meta["arch"], **arch_kwargs)
        state = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return RobotSegmenter(
            model=model,
            input_size=meta["input_size"],
            threshold=meta["threshold"],
            device=device,
            camera_id=meta["camera_id"],
            metadata=meta,
        )
    if kind == "yolo26_instance":
        return YOLO26InstancePredictor(
            weights=root / meta["weights_rel"],
            input_size=meta["input_size"],
            conf=meta["conf"],
            device=device,
            camera_id=meta["camera_id"],
        )
    if kind == "median_background":
        return MedianBackgroundModel.load(root)
    if kind == "static_control":
        return StaticMaskControl.load(root)
    if kind == "sam3_direct":
        from .teachers import SAM3DirectSegmenter

        return SAM3DirectSegmenter(
            checkpoint=meta["checkpoint"],
            prompt=meta["prompt"],
            conf=meta["conf"],
            imgsz=meta["imgsz"],
            device=device,
            camera_id=meta["camera_id"],
            backend=meta["backend"],
        )
    raise ValueError(f"Unknown artifact kind: {kind}")
