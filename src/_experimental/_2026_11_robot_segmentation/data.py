"""Demopack discovery, 6/2/2 episode split, video reading, and label I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

# Single-channel PNG semantic label convention (PLAN-Robot-Segmentation.md)
BG = 0
FG = 1
UNKNOWN = 255


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class EpisodeSplit:
    pack_path: str
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSplit":
        return cls(
            pack_path=data["pack_path"],
            train=list(data["train"]),
            val=list(data["val"]),
            test=list(data["test"]),
        )


def camera_video_path(pack_path: str | Path, episode: str, camera: str) -> Path:
    """Return ``<pack>/<episode>/video_<camera>.mp4`` (or ``<camera>`` if it ends with .mp4)."""
    name = camera if camera.endswith(".mp4") else f"video_{camera}.mp4"
    return Path(pack_path) / episode / name


def list_demopack_episodes(pack_path: str | Path, camera: str = "dev2") -> list[str]:
    """Return sorted episode folder names containing ``video_<camera>.mp4``."""
    root = Path(pack_path)
    episodes = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and camera_video_path(root, p.name, camera).is_file()
    )
    if not episodes:
        raise FileNotFoundError(f"No episodes with video_{camera}.mp4 under {root}")
    return episodes


def make_episode_split(
    pack_path: str | Path,
    camera: str = "dev2",
    n_train: int = 6,
    n_val: int = 2,
    n_test: int = 2,
) -> EpisodeSplit:
    """Freeze the sorted 6/2/2 episode split matching the AL5D comparison protocol."""
    episodes = list_demopack_episodes(pack_path, camera=camera)
    required = n_train + n_val + n_test
    if len(episodes) < required:
        raise ValueError(f"Need at least {required} episodes in {pack_path}, found {len(episodes)}")
    return EpisodeSplit(
        pack_path=str(Path(pack_path).resolve()),
        train=episodes[:n_train],
        val=episodes[n_train : n_train + n_val],
        test=episodes[n_train + n_val : required],
    )


def save_split(split: EpisodeSplit, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def load_split(path: str | Path) -> EpisodeSplit:
    return EpisodeSplit.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def probe_video(video_path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    info = VideoInfo(
        path=str(video_path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return info


def iter_video_frames(
    video_path: str | Path,
    step: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(frame_idx, timestamp_s, rgb_uint8)`` from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    idx = 0
    yielded = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield idx, idx / fps, rgb
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def read_video_frames(
    video_path: str | Path,
    indices: Sequence[int] | None = None,
    step: int = 1,
    max_frames: int | None = None,
) -> tuple[list[int], np.ndarray]:
    """Read RGB uint8 frames ``[N, H, W, 3]`` and their original frame indices."""
    if indices is None:
        ids, frames = [], []
        for idx, _, rgb in iter_video_frames(video_path, step=step, max_frames=max_frames):
            ids.append(idx)
            frames.append(rgb)
        if not frames:
            raise RuntimeError(f"No frames read from {video_path}")
        return ids, np.stack(frames, axis=0)

    wanted = sorted(set(int(i) for i in indices))
    wanted_set = set(wanted)
    max_idx = wanted[-1]
    found: dict[int, np.ndarray] = {}
    for idx, _, rgb in iter_video_frames(video_path, step=1):
        if idx in wanted_set:
            found[idx] = rgb
        if idx >= max_idx:
            break
    missing = [i for i in wanted if i not in found]
    if missing:
        raise IndexError(f"Frame indices {missing[:5]} out of range for {video_path}")
    return wanted, np.stack([found[i] for i in wanted], axis=0)


def frame_motion_scores(frames_rgb: np.ndarray) -> np.ndarray:
    """Mean absolute grayscale frame-to-frame difference in [0, 255] for each frame."""
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_rgb], dtype=np.float32)
    scores = np.zeros(len(gray), dtype=np.float32)
    if len(gray) > 1:
        diff = np.abs(gray[1:] - gray[:-1]).mean(axis=(1, 2))
        scores[1:] = diff
        scores[0] = diff[0]
    return scores


def detect_pause_flags(
    motion_scores: np.ndarray,
    pause_threshold: float | None = None,
    min_run_len: int = 3,
) -> tuple[np.ndarray, float]:
    """Classify frames belonging to stationary pauses (sustained low inter-frame motion)."""
    if pause_threshold is None:
        p15 = float(np.percentile(motion_scores, 15))
        med = float(np.median(motion_scores))
        pause_threshold = max(0.35, min(1.5, 0.5 * (p15 + 0.35 * med)))
    low = motion_scores <= pause_threshold
    flags = np.zeros(len(motion_scores), dtype=bool)
    start = None
    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = i
        elif not is_low and start is not None:
            if i - start >= min_run_len:
                flags[start:i] = True
            start = None
    if start is not None and len(low) - start >= min_run_len:
        flags[start:] = True
    return flags, float(pause_threshold)


def sample_pose_balanced_indices(
    frames_rgb: np.ndarray,
    count: int,
    pause_fraction: float = 0.2,
    seed: int = 0,
    pause_threshold: float | None = None,
) -> list[int]:
    """Sample diverse frames spanning poses, speeds, and pauses.

    Uses greedy farthest-point selection in downsampled grayscale space inside
    moving and paused pools so neighboring frames from a long take do not dominate.
    """
    n = len(frames_rgb)
    if count >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)
    motion = frame_motion_scores(frames_rgb)
    pauses, _ = detect_pause_flags(motion, pause_threshold=pause_threshold)
    feats = np.stack(
        [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (32, 32)).reshape(-1) for f in frames_rgb],
        axis=0,
    ).astype(np.float32)

    def _farthest_subset(pool: np.ndarray, k: int) -> list[int]:
        if k <= 0 or len(pool) == 0:
            return []
        if k >= len(pool):
            return pool.tolist()
        first = int(pool[rng.integers(0, len(pool))])
        chosen = [first]
        dmin = np.linalg.norm(feats[pool] - feats[first], axis=1)
        for _ in range(1, k):
            best_pos = int(np.argmax(dmin))
            nxt = int(pool[best_pos])
            chosen.append(nxt)
            dist = np.linalg.norm(feats[pool] - feats[nxt], axis=1)
            dmin = np.minimum(dmin, dist)
        return chosen

    pause_pool = np.flatnonzero(pauses)
    move_pool = np.flatnonzero(~pauses)
    n_pause = min(len(pause_pool), int(round(count * pause_fraction)))
    n_move = min(len(move_pool), count - n_pause)
    rem = count - (n_pause + n_move)
    if rem > 0:
        if len(move_pool) - n_move >= rem:
            n_move += rem
        else:
            n_pause = min(len(pause_pool), n_pause + rem)

    selected = _farthest_subset(pause_pool, n_pause) + _farthest_subset(move_pool, n_move)
    return sorted(set(selected))


def inventory_demopack(
    pack_path: str | Path,
    cameras: Sequence[str] = ("dev2", "dev3"),
    eval_frames_per_episode: int = 12,
    seed: int = 0,
) -> dict:
    """Step 1 helper: inventory episodes, freeze 6/2/2 split, and select disjoint eval frame IDs."""
    split = make_episode_split(pack_path, camera=cameras[0])
    episodes_info: dict[str, dict] = {}
    eval_candidates: dict[str, dict[str, list[dict]]] = {"val": {}, "test": {}}

    all_eps = split.train + split.val + split.test
    for ep in all_eps:
        ep_entry: dict[str, dict] = {}
        for cam in cameras:
            vpath = camera_video_path(pack_path, ep, cam)
            info = probe_video(vpath)
            ids, frames = read_video_frames(vpath)
            motion = frame_motion_scores(frames)
            pauses, pause_thr = detect_pause_flags(motion)
            ep_entry[cam] = {
                "width": info.width,
                "height": info.height,
                "frame_count": len(ids),
                "fps": info.fps,
                "pause_threshold": pause_thr,
                "pause_frames": int(pauses.sum()),
                "moving_frames": int((~pauses).sum()),
                "motion_median": float(np.median(motion)),
                "motion_p95": float(np.percentile(motion, 95)),
            }
            if cam == cameras[0] and (ep in split.val or ep in split.test):
                part = "val" if ep in split.val else "test"
                chosen = sample_pose_balanced_indices(
                    frames, count=eval_frames_per_episode, pause_fraction=0.33, seed=seed
                )
                eval_candidates[part][ep] = [
                    {
                        "frame_idx": int(ids[i]),
                        "is_pause": bool(pauses[i]),
                        "motion_score": float(motion[i]),
                    }
                    for i in chosen
                ]
        episodes_info[ep] = ep_entry

    return {
        "pack_path": split.pack_path,
        "cameras": list(cameras),
        "split": split.to_dict(),
        "episodes": episodes_info,
        "eval_candidates": eval_candidates,
    }


def save_label_png(path: str | Path, label: np.ndarray) -> Path:
    """Save a single-channel uint8 semantic label PNG (0=bg, 1=robot, 255=unknown)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(label, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got shape {arr.shape}")
    if not cv2.imwrite(str(out), arr):
        raise RuntimeError(f"Failed to write label PNG: {out}")
    return out


def load_label_png(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Cannot read label PNG: {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8)


def render_mask_overlay(
    rgb: np.ndarray,
    label: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay semantic label (1=green robot, 255=amber unknown) onto an RGB uint8 frame."""
    out = rgb.astype(np.float32).copy()
    fg = label == FG
    unk = label == UNKNOWN
    if fg.any():
        out[fg] = (1.0 - alpha) * out[fg] + alpha * np.array([30.0, 220.0, 70.0], dtype=np.float32)
    if unk.any():
        out[unk] = (1.0 - alpha) * out[unk] + alpha * np.array([240.0, 170.0, 20.0], dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_rgb_png(path: str | Path, rgb: np.ndarray) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out), bgr):
        raise RuntimeError(f"Failed to write image: {out}")
    return out
