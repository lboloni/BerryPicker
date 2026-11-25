"""Training loops for partial-label semantic students and YOLO26 instance comparison."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import BG, FG, UNKNOWN, camera_video_path, load_label_png, read_video_frames, save_rgb_png
from .models import (
    audit_polygon_roundtrip,
    build_semantic_model,
    letterbox_rgb,
    mask_to_yolo_polygons,
    resolve_device,
    save_semantic_checkpoint,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def letterbox_label(label: np.ndarray, input_size: int | tuple[int, int]) -> np.ndarray:
    """Apply matching aspect-preserving resize + pad (with UNKNOWN=255) to a 0/1/255 mask."""
    h, w = label.shape[:2]
    if isinstance(input_size, int):
        th, tw = input_size, input_size
    else:
        th, tw = int(input_size[0]), int(input_size[1])
    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)
    pad_left = (tw - nw) // 2
    pad_top = (th - nh) // 2
    canvas = np.full((th, tw), UNKNOWN, dtype=np.uint8)
    canvas[pad_top : pad_top + nh, pad_left : pad_left + nw] = resized
    return canvas


def augment_pair(
    rgb: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    allow_hflip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical geometric augmentation to RGB and partial 0/1/255 label."""
    h, w = rgb.shape[:2]
    if allow_hflip and rng.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        label = np.ascontiguousarray(label[:, ::-1])

    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.90, 1.10))
    tx = float(rng.uniform(-0.06, 0.06) * w)
    ty = float(rng.uniform(-0.06, 0.06) * h)
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    rgb_aug = cv2.warpAffine(
        rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)
    )
    lbl_aug = cv2.warpAffine(
        label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=int(UNKNOWN)
    )

    # Mild photometric jitter on RGB only
    alpha = float(rng.uniform(0.88, 1.12))
    beta = float(rng.uniform(-12.0, 12.0))
    rgb_aug = np.clip(rgb_aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return rgb_aug, lbl_aug


def partial_balanced_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Foreground/background-balanced BCE strictly over confident pixels (labels != UNKNOWN)."""
    valid = labels != UNKNOWN
    if not valid.any():
        return logits.sum() * 0.0

    pos = labels == FG
    neg = labels == BG
    target = pos.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    n_pos = pos.sum().clamp_min(1).float()
    n_neg = neg.sum().clamp_min(1).float()
    loss_pos = (bce * pos.float()).sum() / n_pos
    loss_neg = (bce * neg.float()).sum() / n_neg
    if pos.any() and neg.any():
        return 0.5 * (loss_pos + loss_neg)
    if pos.any():
        return loss_pos
    return loss_neg


class PartialLabelDataset(Dataset):
    """In-memory paired RGB + 0/1/255 partial label dataset."""

    def __init__(
        self,
        images_rgb: Sequence[np.ndarray],
        labels: Sequence[np.ndarray],
        input_size: int | tuple[int, int] = 256,
        augment: bool = False,
        seed: int = 0,
    ):
        self.images_rgb = list(images_rgb)
        self.labels = list(labels)
        self.input_size = input_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images_rgb)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, _ = letterbox_rgb(self.images_rgb[idx], self.input_size)
        lbl = letterbox_label(self.labels[idx], self.input_size)
        if self.augment:
            rgb, lbl = augment_pair(rgb, lbl, self.rng)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        y = torch.from_numpy(lbl).unsqueeze(0).long()
        return x, y


def load_teacher_samples(
    pack_path: str | Path,
    teacher_root: str | Path,
    teacher_name: str,
    camera: str,
    episodes: Sequence[str],
    frame_filter: dict[str, Sequence[int]] | None = None,
    only_retained: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Load RGB frames and teacher 0/1/255 labels for selected episodes (optionally matched frame IDs)."""
    images, labels, records = [], [], []
    for ep in episodes:
        ep_dir = Path(teacher_root) / teacher_name / camera / ep
        manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
        allowed = set(int(i) for i in frame_filter[ep]) if (frame_filter and ep in frame_filter) else None

        wanted = []
        for rec in manifest["frames"]:
            fidx = int(rec["frame_idx"])
            if only_retained and not rec["retained"]:
                continue
            if allowed is not None and fidx not in allowed:
                continue
            wanted.append(fidx)
        if not wanted:
            continue

        ids, frames = read_video_frames(camera_video_path(pack_path, ep, camera), indices=wanted)
        lbl_dir = ep_dir / "training_labels"
        for fidx, rgb in zip(ids, frames):
            lbl = load_label_png(lbl_dir / f"frame_{fidx:06d}.png")
            images.append(rgb)
            labels.append(lbl)
            records.append({"episode": ep, "camera": camera, "frame_idx": int(fidx)})
    if not images:
        raise RuntimeError(f"No training samples loaded for teacher={teacher_name}, camera={camera}")
    return images, labels, records


def _confident_iou(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    valid = labels != UNKNOWN
    if not valid.any():
        return 0.0
    pred = (torch.sigmoid(logits) >= threshold) & valid
    gt = (labels == FG) & valid
    inter = int((pred & gt).sum().item())
    union = int((pred | gt).sum().item())
    return inter / union if union > 0 else 1.0


def train_semantic_student(
    train_images: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    val_images: Sequence[np.ndarray],
    val_labels: Sequence[np.ndarray],
    out_dir: str | Path,
    arch: str = "unet",
    arch_kwargs: dict | None = None,
    camera_id: str = "dev2",
    input_size: int = 256,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    seed: int = 0,
    device: str = "auto",
    split: dict | None = None,
    train_records: list[dict] | None = None,
) -> dict:
    """Train a single-frame semantic student on 0/1/255 partial labels and save best checkpoint."""
    set_seed(seed)
    dev = resolve_device(device)
    arch_kwargs = dict(arch_kwargs or {})
    model = build_semantic_model(arch, **arch_kwargs).to(dev)

    train_ds = PartialLabelDataset(train_images, train_labels, input_size=input_size, augment=True, seed=seed)
    val_ds = PartialLabelDataset(val_images, val_labels, input_size=input_size, augment=False, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_iou = -1.0

    training_settings = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(dev),
        "n_train": len(train_images),
        "n_val": len(val_images),
    }

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss_sum = 0.0
        tr_steps = 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = partial_balanced_bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss_sum += float(loss.item())
            tr_steps += 1
        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_steps = 0
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = model(x)
                val_loss_sum += float(partial_balanced_bce_loss(logits, y).item())
                val_iou_sum += _confident_iou(logits, y, threshold=threshold)
                val_steps += 1

        row = {
            "epoch": epoch,
            "train_loss": tr_loss_sum / max(1, tr_steps),
            "val_loss": val_loss_sum / max(1, val_steps),
            "val_confident_iou": val_iou_sum / max(1, val_steps),
        }
        history.append(row)
        if row["val_confident_iou"] >= best_val_iou:
            best_val_iou = row["val_confident_iou"]
            save_semantic_checkpoint(
                out / "best",
                model=model,
                arch=arch,
                arch_kwargs=arch_kwargs,
                camera_id=camera_id,
                input_size=input_size,
                threshold=threshold,
                split=split,
                training_settings=training_settings,
            )

    save_semantic_checkpoint(
        out / "last",
        model=model,
        arch=arch,
        arch_kwargs=arch_kwargs,
        camera_id=camera_id,
        input_size=input_size,
        threshold=threshold,
        split=split,
        training_settings=training_settings,
    )
    summary = {
        "arch": arch,
        "arch_kwargs": arch_kwargs,
        "camera_id": camera_id,
        "input_size": input_size,
        "best_val_confident_iou": best_val_iou,
        "history": history,
        "train_records": train_records or [],
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def export_yolo_instance_dataset(
    train_images: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    val_images: Sequence[np.ndarray],
    val_labels: Sequence[np.ndarray],
    dataset_dir: str | Path,
    approx_eps_px: float = 1.0,
) -> tuple[Path, dict]:
    """Export binary robot masks as a single-class ``al5d_robot`` YOLO polygon dataset + audit."""
    root = Path(dataset_dir)
    for part, imgs, lbls in [("train", train_images, train_labels), ("val", val_images, val_labels)]:
        (root / "images" / part).mkdir(parents=True, exist_ok=True)
        (root / "labels" / part).mkdir(parents=True, exist_ok=True)
        for i, (rgb, lbl) in enumerate(zip(imgs, lbls)):
            stem = f"{part}_{i:05d}"
            save_rgb_png(root / "images" / part / f"{stem}.png", rgb)
            lines = mask_to_yolo_polygons((lbl == FG).astype(np.uint8), class_id=0, approx_eps_px=approx_eps_px)
            (root / "labels" / part / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    audit = {
        "train": audit_polygon_roundtrip([(m == FG).astype(np.uint8) for m in train_labels], approx_eps_px=approx_eps_px),
        "val": audit_polygon_roundtrip([(m == FG).astype(np.uint8) for m in val_labels], approx_eps_px=approx_eps_px),
    }
    (root / "polygon_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        f"path: {root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: al5d_robot\n",
        encoding="utf-8",
    )
    return yaml_path, audit


def train_yolo_instance_student(
    train_images: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    val_images: Sequence[np.ndarray],
    val_labels: Sequence[np.ndarray],
    out_dir: str | Path,
    weights_or_cfg: str = "yolo26n-seg.pt",
    camera_id: str = "dev2",
    input_size: int = 256,
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 0,
    device: str = "auto",
    approx_eps_px: float = 1.0,
) -> dict:
    """Audit polygon conversion, then fine-tune Ultralytics ``yolo26n-seg`` on ``al5d_robot``."""
    from ultralytics import YOLO

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path, audit = export_yolo_instance_dataset(
        train_images, train_labels, val_images, val_labels, out / "yolo_dataset", approx_eps_px=approx_eps_px
    )
    dev = resolve_device(device)
    model = YOLO(weights_or_cfg)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=input_size,
        batch=batch_size,
        seed=seed,
        device=str(dev),
        project=str(out / "runs"),
        name="train",
        exist_ok=True,
        verbose=False,
    )
    best_pt = out / "runs" / "train" / "weights" / "best.pt"
    artifact_dir = out / "best"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target_pt = artifact_dir / "best.pt"
    target_pt.write_bytes(best_pt.read_bytes())
    meta = {
        "kind": "yolo26_instance",
        "weights_rel": "best.pt",
        "camera_id": camera_id,
        "input_size": int(input_size),
        "conf": 0.25,
        "polygon_audit": audit,
    }
    (artifact_dir / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"artifact_dir": str(artifact_dir), "polygon_audit": audit}
