"""Composable uint8 RGB image domain randomization operators.

Every operator takes ``frame_rgb`` of shape ``(H, W, 3)`` (uint8), an explicit
``np.random.Generator``, optional binary ``mask`` (``uint8 (H, W)``, ``1=robot``,
``0=background``), and keyword parameters, returning a contiguous ``uint8`` RGB
array of the exact same shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np


def _ensure_rgb_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB frame, got {frame_rgb.shape}")
    if frame_rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame_rgb.dtype}")
    return frame_rgb


def _ensure_mask(mask: np.ndarray | None, hw: tuple[int, int], op_name: str) -> np.ndarray:
    if mask is None:
        raise ValueError(f"Operator '{op_name}' requires a binary robot mask, got None")
    if mask.shape != hw:
        raise ValueError(f"Mask shape {mask.shape} != frame shape {hw} for '{op_name}'")
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Photometric & lighting transforms
# ---------------------------------------------------------------------------


def op_color_jitter(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    brightness: float = 0.25,
    contrast: float = 0.25,
    saturation: float = 0.25,
    hue: float = 0.05,
) -> np.ndarray:
    """Random multiplicative brightness, contrast, saturation, and HSV hue shift."""
    _ensure_rgb_uint8(frame_rgb)
    x = frame_rgb.astype(np.float32) / 255.0

    if brightness > 0:
        factor = float(rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness))
        x = x * factor
    if contrast > 0:
        factor = float(rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast))
        mean = float(x.mean())
        x = (x - mean) * factor + mean
    if saturation > 0:
        factor = float(rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation))
        gray = x.mean(axis=2, keepdims=True)
        x = (x - gray) * factor + gray

    out = np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)

    if hue > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
        delta_h = int(round(float(rng.uniform(-hue, hue)) * 180.0))
        hsv[:, :, 0] = (hsv[:, :, 0] + delta_h) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return np.ascontiguousarray(out)


def op_lighting_gradient(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    strength_range: Sequence[float] = (0.10, 0.35),
    mode: str = "random",
) -> np.ndarray:
    """Apply a smooth directional linear or radial illumination gradient."""
    _ensure_rgb_uint8(frame_rgb)
    h, w = frame_rgb.shape[:2]
    chosen = mode if mode != "random" else ("linear" if rng.random() < 0.6 else "radial")
    strength = float(rng.uniform(float(strength_range[0]), float(strength_range[1])))

    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )
    if chosen == "linear":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        grad = np.cos(angle) * xx + np.sin(angle) * yy
        gain = 1.0 + strength * grad
    elif chosen == "radial":
        cx = float(rng.uniform(-0.8, 0.8))
        cy = float(rng.uniform(-0.8, 0.8))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        gain = 1.0 + sign * strength * (1.0 - np.clip(dist, 0.0, 1.5) / 1.5)
    else:
        raise ValueError(f"Unknown lighting_gradient mode: {chosen}")

    out = np.clip(frame_rgb.astype(np.float32) * gain[..., None], 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(out)


def op_white_balance(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    gain_range: Sequence[float] = (0.85, 1.15),
) -> np.ndarray:
    """Per-channel RGB gain drift simulating ambient color temperature shifts."""
    _ensure_rgb_uint8(frame_rgb)
    gains = rng.uniform(float(gain_range[0]), float(gain_range[1]), size=(1, 1, 3)).astype(np.float32)
    out = np.clip(frame_rgb.astype(np.float32) * gains, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(out)


def op_gamma(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    gamma_range: Sequence[float] = (0.7, 1.4),
) -> np.ndarray:
    """Random power-law exposure adjustment."""
    _ensure_rgb_uint8(frame_rgb)
    g = float(rng.uniform(float(gamma_range[0]), float(gamma_range[1])))
    lut = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** g) * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(cv2.LUT(frame_rgb, lut))


# ---------------------------------------------------------------------------
# Sensor & optics transforms
# ---------------------------------------------------------------------------


def op_gaussian_noise(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    sigma_range: Sequence[float] = (2.0, 12.0),
) -> np.ndarray:
    """Additive zero-mean Gaussian sensor noise (in uint8 pixel units)."""
    _ensure_rgb_uint8(frame_rgb)
    sigma = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
    noise = rng.normal(0.0, sigma, size=frame_rgb.shape).astype(np.float32)
    out = np.clip(frame_rgb.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(out)


def op_gaussian_blur(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    kernel_sizes: Sequence[int] = (3, 5),
    sigma_range: Sequence[float] = (0.4, 1.4),
) -> np.ndarray:
    """Mild Gaussian defocus blur."""
    _ensure_rgb_uint8(frame_rgb)
    k = int(kernel_sizes[int(rng.integers(0, len(kernel_sizes)))])
    if k % 2 == 0 or k < 1:
        raise ValueError(f"gaussian_blur kernel size must be positive odd, got {k}")
    sigma = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
    return np.ascontiguousarray(cv2.GaussianBlur(frame_rgb, (k, k), sigma))


def op_jpeg_compression(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    quality_range: Sequence[int] = (35, 95),
) -> np.ndarray:
    """Simulate lossy JPEG/webcam compression artifacts."""
    _ensure_rgb_uint8(frame_rgb)
    q = int(rng.integers(int(quality_range[0]), int(quality_range[1]) + 1))
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if dec is None:
        raise RuntimeError("cv2.imdecode failed")
    return np.ascontiguousarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Background generators & mask-aware clutter transforms
# ---------------------------------------------------------------------------


def _gen_solid_color(h: int, w: int, rng: np.random.Generator, spec: dict[str, Any]) -> np.ndarray:
    mode = spec.get("mode", "random")
    if mode == "fixed":
        color = np.asarray(spec["color"], dtype=np.uint8)
        if color.shape != (3,):
            raise ValueError(f"solid_color fixed color must have 3 RGB values, got {color}")
    elif mode == "random":
        color = rng.integers(0, 256, size=(3,), dtype=np.uint8)
    else:
        raise ValueError(f"Unknown solid_color mode: {mode}")
    return np.broadcast_to(color[None, None, :], (h, w, 3)).copy()


def _gen_smooth_field(h: int, w: int, rng: np.random.Generator, spec: dict[str, Any]) -> np.ndarray:
    grid = int(spec.get("grid_size", 4))
    coarse = rng.uniform(0.0, 255.0, size=(grid, grid, 3)).astype(np.float32)
    up = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0.0, 255.0).astype(np.uint8)


def _gen_multi_scale_noise(h: int, w: int, rng: np.random.Generator, spec: dict[str, Any]) -> np.ndarray:
    scales = spec.get("scales", [4, 16, 64])
    weights = spec.get("scale_weights", [0.45, 0.35, 0.20])
    acc = np.zeros((h, w, 3), dtype=np.float32)
    w_sum = 0.0
    for sc, wt in zip(scales, weights):
        sc_int = max(2, int(sc))
        patch = rng.uniform(0.0, 255.0, size=(sc_int, sc_int, 3)).astype(np.float32)
        up = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        acc += float(wt) * up
        w_sum += float(wt)
    return np.clip(acc / max(w_sum, 1e-6), 0.0, 255.0).astype(np.uint8)


def _gen_geometric_pattern(h: int, w: int, rng: np.random.Generator, spec: dict[str, Any]) -> np.ndarray:
    style = spec.get("style", "random")
    if style == "random":
        style = ("stripes", "checker", "gradient", "voronoi")[int(rng.integers(0, 4))]

    c1 = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
    c2 = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")

    if style == "stripes":
        angle = float(rng.uniform(0.0, np.pi))
        period = float(rng.uniform(12.0, max(24.0, min(h, w) * 0.35)))
        phase = float(rng.uniform(0.0, period))
        coord = xx * np.cos(angle) + yy * np.sin(angle) + phase
        t = ((coord % period) < (period * 0.5)).astype(np.float32)[..., None]
        bg = c1 * t + c2 * (1.0 - t)
    elif style == "checker":
        angle = float(rng.uniform(0.0, np.pi * 0.5))
        cell = float(rng.uniform(14.0, max(24.0, min(h, w) * 0.28)))
        u = np.floor((xx * np.cos(angle) + yy * np.sin(angle)) / cell)
        v = np.floor((-xx * np.sin(angle) + yy * np.cos(angle)) / cell)
        t = (((u + v) % 2) == 0).astype(np.float32)[..., None]
        bg = c1 * t + c2 * (1.0 - t)
    elif style == "gradient":
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        proj = (xx / max(w - 1, 1) - 0.5) * np.cos(angle) + (yy / max(h - 1, 1) - 0.5) * np.sin(angle)
        t = np.clip(proj + 0.5, 0.0, 1.0)[..., None]
        bg = c1 * t + c2 * (1.0 - t)
    elif style == "voronoi":
        n_cells = int(rng.integers(6, 18))
        pts_y = rng.uniform(0, h, size=(n_cells,)).astype(np.float32)
        pts_x = rng.uniform(0, w, size=(n_cells,)).astype(np.float32)
        colors = rng.integers(0, 256, size=(n_cells, 3), dtype=np.uint8)
        d2 = (yy[:, :, None] - pts_y[None, None, :]) ** 2 + (xx[:, :, None] - pts_x[None, None, :]) ** 2
        nearest = np.argmin(d2, axis=2)
        return np.ascontiguousarray(colors[nearest])
    else:
        raise ValueError(f"Unknown geometric_pattern style: {style}")

    return np.clip(bg, 0.0, 255.0).astype(np.uint8)


def _gen_image_folder(h: int, w: int, rng: np.random.Generator, spec: dict[str, Any]) -> np.ndarray:
    folder = Path(spec["folder"]).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"background_replace image_folder not found: {folder}")
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if not files:
        raise FileNotFoundError(f"No images found in background folder: {folder}")
    chosen = files[int(rng.integers(0, len(files)))]
    bgr = cv2.imread(str(chosen), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read background image: {chosen}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]
    if ih >= h and iw >= w:
        y0 = int(rng.integers(0, ih - h + 1))
        x0 = int(rng.integers(0, iw - w + 1))
        return np.ascontiguousarray(rgb[y0 : y0 + h, x0 : x0 + w])
    return np.ascontiguousarray(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR))


_BG_GENERATORS: dict[str, Callable[[int, int, np.random.Generator, dict[str, Any]], np.ndarray]] = {
    "solid_color": _gen_solid_color,
    "smooth_field": _gen_smooth_field,
    "multi_scale_noise": _gen_multi_scale_noise,
    "geometric_pattern": _gen_geometric_pattern,
    "image_folder": _gen_image_folder,
}


def op_background_replace(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    sources: Sequence[dict[str, Any]],
    edge_feather_px: int = 0,
    mask_jitter_px: int = 0,
) -> np.ndarray:
    """Replace background pixels (``mask == 0``) using a weighted background generator."""
    _ensure_rgb_uint8(frame_rgb)
    h, w = frame_rgb.shape[:2]
    fg_mask = _ensure_mask(mask, (h, w), "background_replace")

    if not sources:
        raise ValueError("background_replace requires at least one entry in 'sources'")
    weights = np.array([float(s.get("weight", 1.0)) for s in sources], dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError(f"All background_replace source weights must be > 0, got {weights}")
    probs = weights / weights.sum()
    spec = sources[int(rng.choice(len(sources), p=probs))]
    kind = spec["kind"]
    if kind not in _BG_GENERATORS:
        raise ValueError(f"Unknown background_replace source kind: {kind}")
    bg = _BG_GENERATORS[kind](h, w, rng, spec)

    if mask_jitter_px > 0:
        delta = int(rng.integers(-mask_jitter_px, mask_jitter_px + 1))
        if delta != 0:
            k = 2 * abs(delta) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            if delta > 0:
                fg_mask = cv2.dilate(fg_mask, kernel)
            else:
                fg_mask = cv2.erode(fg_mask, kernel)

    if edge_feather_px > 0:
        k = 2 * int(edge_feather_px) + 1
        alpha = cv2.GaussianBlur(fg_mask.astype(np.float32), (k, k), 0.0)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        comp = frame_rgb.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
        return np.ascontiguousarray(np.clip(comp, 0.0, 255.0).astype(np.uint8))

    return np.ascontiguousarray(np.where(fg_mask[..., None] > 0, frame_rgb, bg).astype(np.uint8))


def op_distractors(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    count_range: Sequence[int] = (1, 4),
    placement: str = "background_only",
    size_frac_range: Sequence[float] = (0.06, 0.22),
    shapes: Sequence[str] = ("rect", "circle", "polygon", "line"),
) -> np.ndarray:
    """Draw synthetic distractor objects/cables onto background or anywhere."""
    _ensure_rgb_uint8(frame_rgb)
    h, w = frame_rgb.shape[:2]
    if placement not in ("background_only", "anywhere"):
        raise ValueError(f"Unknown distractors placement: {placement}")
    fg_mask = _ensure_mask(mask, (h, w), "distractors") if placement == "background_only" else None

    canvas = frame_rgb.copy()
    n_obj = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1))
    min_dim = float(min(h, w))

    for _ in range(n_obj):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        color = tuple(int(c) for c in rng.integers(0, 256, size=(3,)))
        size = max(4, int(round(float(rng.uniform(float(size_frac_range[0]), float(size_frac_range[1]))) * min_dim)))
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        if shape == "rect":
            rw = max(4, int(round(size * float(rng.uniform(0.5, 1.6)))))
            rh = max(4, int(round(size * float(rng.uniform(0.5, 1.6)))))
            cv2.rectangle(canvas, (cx - rw // 2, cy - rh // 2), (cx + rw // 2, cy + rh // 2), color, -1)
        elif shape == "circle":
            cv2.circle(canvas, (cx, cy), max(2, size // 2), color, -1)
        elif shape == "polygon":
            n_pts = int(rng.integers(3, 7))
            angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=(n_pts,)))
            radii = rng.uniform(0.35 * size, 0.75 * size, size=(n_pts,))
            pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1).astype(np.int32)
            cv2.fillPoly(canvas, [pts], color)
        elif shape == "line":
            x2 = int(rng.integers(0, w))
            y2 = int(rng.integers(0, h))
            thick = int(rng.integers(2, max(4, size // 5)))
            cv2.line(canvas, (cx, cy), (x2, y2), color, thick, cv2.LINE_AA)
        else:
            raise ValueError(f"Unknown distractor shape: {shape}")

    if fg_mask is not None:
        canvas = np.where(fg_mask[..., None] > 0, frame_rgb, canvas)
    return np.ascontiguousarray(canvas.astype(np.uint8))


def op_random_erasing(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    count_range: Sequence[int] = (1, 2),
    area_frac_range: Sequence[float] = (0.02, 0.12),
    fill: str = "random_color",
) -> np.ndarray:
    """Cutout / random rectangular occlusion patches."""
    _ensure_rgb_uint8(frame_rgb)
    h, w = frame_rgb.shape[:2]
    out = frame_rgb.copy()
    n_patches = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1))

    for _ in range(n_patches):
        area = float(rng.uniform(float(area_frac_range[0]), float(area_frac_range[1]))) * h * w
        aspect = float(rng.uniform(0.4, 2.5))
        ph = max(2, min(h - 1, int(round(np.sqrt(area * aspect)))))
        pw = max(2, min(w - 1, int(round(np.sqrt(area / aspect)))))
        y0 = int(rng.integers(0, h - ph + 1))
        x0 = int(rng.integers(0, w - pw + 1))

        if fill == "random_color":
            out[y0 : y0 + ph, x0 : x0 + pw] = rng.integers(0, 256, size=(3,), dtype=np.uint8)
        elif fill == "gray":
            out[y0 : y0 + ph, x0 : x0 + pw] = 128
        elif fill == "noise":
            out[y0 : y0 + ph, x0 : x0 + pw] = rng.integers(0, 256, size=(ph, pw, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown random_erasing fill: {fill}")

    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Small camera-mount jitter (opt-in only)
# ---------------------------------------------------------------------------


def op_camera_jitter(
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    max_shift_px: float = 3.0,
    max_deg: float = 2.0,
    scale_range: Sequence[float] = (0.98, 1.02),
    border_mode: str = "reflect",
) -> np.ndarray:
    """Small affine camera-mount perturbation (opt-in; keep tiny for VP)."""
    _ensure_rgb_uint8(frame_rgb)
    h, w = frame_rgb.shape[:2]
    angle = float(rng.uniform(-max_deg, max_deg))
    scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    tx = float(rng.uniform(-max_shift_px, max_shift_px))
    ty = float(rng.uniform(-max_shift_px, max_shift_px))
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty

    bmodes = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "constant": cv2.BORDER_CONSTANT,
    }
    if border_mode not in bmodes:
        raise ValueError(f"Unknown camera_jitter border_mode: {border_mode}")
    out = cv2.warpAffine(frame_rgb, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=bmodes[border_mode])
    return np.ascontiguousarray(out)


OP_REGISTRY: dict[str, Callable[..., np.ndarray]] = {
    "color_jitter": op_color_jitter,
    "lighting_gradient": op_lighting_gradient,
    "white_balance": op_white_balance,
    "gamma": op_gamma,
    "gaussian_noise": op_gaussian_noise,
    "gaussian_blur": op_gaussian_blur,
    "jpeg_compression": op_jpeg_compression,
    "background_replace": op_background_replace,
    "distractors": op_distractors,
    "random_erasing": op_random_erasing,
    "camera_jitter": op_camera_jitter,
}

MASK_REQUIRED_OPS = {"background_replace"}


def step_requires_mask(step: dict[str, Any]) -> bool:
    """Return True if a configured pipeline step requires a robot mask."""
    op_name = step["op"]
    if op_name in MASK_REQUIRED_OPS:
        return True
    if op_name == "distractors" and step.get("placement", "background_only") == "background_only":
        return True
    return False


def apply_op(
    op_name: str,
    frame_rgb: np.ndarray,
    rng: np.random.Generator,
    *,
    mask: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Invoke a named domain randomization operator."""
    if op_name not in OP_REGISTRY:
        raise KeyError(f"Unknown domain randomization op '{op_name}'. Known ops: {sorted(OP_REGISTRY)}")
    kwargs = dict(params or {})
    return OP_REGISTRY[op_name](frame_rgb, rng, mask=mask, **kwargs)
