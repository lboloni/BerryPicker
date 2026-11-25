"""Offline clips use BGR uint8 in memory and paths relative to a JSON manifest."""
from pathlib import Path
import json
import shutil
import subprocess

import cv2
import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def still(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise OSError(f"Cannot read {path}")
    return image


def frames(path):
    path = Path(path)
    if path.is_dir():
        names = sorted(p for p in path.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        return np.stack([still(p) for p in names])
    capture = cv2.VideoCapture(str(path))
    try:
        images = []
        while capture.isOpened():
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected > 0 and expected != len(images):
            raise OSError(f"Decoded {len(images)} of {expected} frames from {path}")
        return np.stack(images)
    finally:
        capture.release()


def pngs(path, images):
    path.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        if not cv2.imwrite(str(path / f"{i:06}.png"), image):
            raise OSError(f"Cannot write frame {i} to {path}")


def video(path, images, fps):
    h, w = images[0].shape[:2]
    executable = shutil.which("ffmpeg")
    if executable:
        args = [executable, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            for image in images:
                encoder.stdin.write(image.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise OSError(f"ffmpeg could not encode {path}")
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError(f"Cannot write {path}")
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()


class Clip:
    def __init__(self, path):
        path = Path(path)
        self.root = path.parent
        self.spec = json.loads(path.read_text())
        if "occluder_rgb" in self.spec:
            clean = frames(self.root/self.spec["clean"])
            foreground = frames(self.root/self.spec["occluder_rgb"])
            alpha = frames(self.root/self.spec["occluder_alpha"])[...,0]
            self.rgb = synthesize_occlusions(clean, foreground, alpha)
        else:
            self.rgb = frames(self.root/self.spec["rgb"])
        self.fps = self.spec["fps"]
        self.times = np.asarray(self.spec.get("timestamps", np.arange(len(self.rgb)) / self.fps))
        self.ids = self.spec.get("frame_ids", list(range(len(self.rgb))))
        if len(self.times) != len(self.rgb) or len(self.ids) != len(self.rgb):
            raise ValueError("Clip timestamps and IDs must align with decoded frames")
        self.plate = still(self.root / self.spec["plate"])

    def series(self, field):
        return frames(self.root / self.spec[field])


def synthesize_occlusions(clean, foreground, alpha):
    """Make paired supervised clips from moving clean RGB and an aligned hand layer."""
    opacity = np.asarray(alpha, np.float32)[...,None]/255
    return np.rint(clean*(1-opacity) + foreground*opacity).astype(np.uint8)
