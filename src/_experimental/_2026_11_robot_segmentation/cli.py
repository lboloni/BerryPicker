"""Standalone CLI for the 7-step robot segmentation experiment.

Runs directly from a demopack path or video path into a caller-supplied result
directory with no imports from BerryPicker core packages or ExpRunFlow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import MedianBackgroundModel, StaticMaskControl
from .benchmark import run_paced_benchmark
from .data import (
    camera_video_path,
    inventory_demopack,
    load_split,
    make_episode_split,
    read_video_frames,
    render_mask_overlay,
    sample_pose_balanced_indices,
    save_rgb_png,
    save_split,
)
from .evaluate import evaluate_predictor, extract_annotation_pack
from .models import load_segmenter
from .teachers import (
    MotionLabelTeacher,
    SAM2Teacher,
    SAM3DirectSegmenter,
    SAM3Teacher,
    apply_static_consensus_gating,
    estimate_static_pcb_mask,
    save_teacher_outputs,
)
from .train import load_teacher_samples, train_semantic_student, train_yolo_instance_student


def cmd_inventory(args: argparse.Namespace) -> None:
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = inventory_demopack(
        args.pack_path,
        cameras=tuple(args.cameras),
        eval_frames_per_episode=args.eval_frames_per_episode,
        seed=args.seed,
    )
    (out / "inventory.json").write_text(json.dumps(inv, indent=2) + "\n", encoding="utf-8")
    save_split(make_episode_split(args.pack_path, camera=args.cameras[0]), out / "split.json")
    if args.extract_eval:
        for cam in args.cameras:
            extract_annotation_pack(inv, out / "eval_pack", camera=cam)
    print(f"Wrote inventory and frozen split to {out}")


def cmd_baseline(args: argparse.Namespace) -> None:
    out = Path(args.result_dir)
    split = load_split(args.split) if args.split else make_episode_split(args.pack_path, camera=args.camera)
    bg = MedianBackgroundModel.fit_from_demopack(
        args.pack_path,
        split.train,
        camera=args.camera,
        frame_step=args.frame_step,
        threshold=args.threshold,
        dilate=args.dilate,
    )
    bg.save(out / "median_background" / args.camera)

    # Fit static positional control from training baseline masks
    train_masks = []
    for ep in split.train:
        _, frames = read_video_frames(camera_video_path(args.pack_path, ep, args.camera), step=args.frame_step)
        train_masks.extend(bg.segment(f) for f in frames)
    static = StaticMaskControl.fit_from_masks(train_masks, threshold=0.5, camera_id=args.camera)
    static.save(out / "static_control" / args.camera)

    # Save sample validation overlays
    ovl_dir = out / "baseline_val_overlays" / args.camera
    for ep in split.val:
        ids, frames = read_video_frames(camera_video_path(args.pack_path, ep, args.camera), step=args.overlay_step)
        for fidx, rgb in zip(ids, frames):
            mask = bg.segment(rgb)
            save_rgb_png(ovl_dir / ep / f"frame_{fidx:06d}.png", render_mask_overlay(rgb, mask))
    print(f"Saved median_background and static_control for {args.camera} under {out}")


def cmd_label(args: argparse.Namespace) -> None:
    out = Path(args.result_dir)
    split = load_split(args.split) if args.split else make_episode_split(args.pack_path, camera=args.camera)
    bg_path = Path(args.bg_dir) if args.bg_dir else (out / "median_background" / args.camera)
    if (bg_path / "config.json").is_file():
        bg = MedianBackgroundModel.load(bg_path)
    else:
        bg = MedianBackgroundModel.fit_from_demopack(args.pack_path, split.train, camera=args.camera)
        bg.save(bg_path)

    episodes = list(args.episodes) if args.episodes else (split.train + split.val)
    motion_teacher = MotionLabelTeacher(bg)
    sam2_teacher = (
        SAM2Teacher(args.sam2_cfg, args.sam2_ckpt, device=args.device, use_box=not args.no_box)
        if args.teacher == "sam2"
        else None
    )
    sam3_seg = None
    pcb_mask = None
    sam3_teacher = None
    if args.teacher == "sam3":
        sam3_seg = SAM3DirectSegmenter(
            checkpoint=args.sam3_ckpt,
            prompt=args.sam3_prompt,
            conf=args.sam3_conf,
            imgsz=args.sam3_imgsz,
            device=args.device,
            camera_id=args.camera,
            selection_mode=args.sam3_selection_mode,
        )
        if not args.no_sam3_stabilize:
            pcb_mask = estimate_static_pcb_mask(
                sam3_seg, bg, pcb_prompts=tuple(args.sam3_pcb_prompts), conf=args.sam3_pcb_conf
            )
            sam3_seg.bg_model = bg
            sam3_seg.pcb_mask = pcb_mask
        sam3_seg.save(out / "models" / f"sam3_direct_{args.camera}")
        sam3_teacher = SAM3Teacher(
            prompt=args.sam3_prompt,
            checkpoint=args.sam3_ckpt,
            conf=args.sam3_conf,
            imgsz=args.sam3_imgsz,
            device=args.device,
            segmenter=sam3_seg,
        )

    results_by_ep = {}
    frames_by_ep = {}
    for ep in episodes:
        ids, frames = read_video_frames(
            camera_video_path(args.pack_path, ep, args.camera), step=args.step, max_frames=args.max_frames
        )
        mres = motion_teacher.label_episode(frames, ids, episode=ep, camera=args.camera)
        if args.teacher == "motion":
            res = mres
        elif args.teacher == "sam2":
            assert sam2_teacher is not None
            res = sam2_teacher.label_episode(frames, ids, mres, episode=ep, camera=args.camera, seed=args.seed)
        else:
            assert sam3_teacher is not None
            res = sam3_teacher.label_episode(frames, ids, mres, episode=ep, camera=args.camera)
        results_by_ep[ep] = res
        frames_by_ep[ep] = frames

    # Stage B: static consensus gating across training episodes for SAM 3.1 (PCB included as FG)
    if args.teacher == "sam3" and not args.no_sam3_stabilize:
        train_eps = [ep for ep in episodes if ep in split.train]
        if train_eps:
            train_res = {ep: results_by_ep[ep] for ep in train_eps}
            train_frms = {ep: frames_by_ep[ep] for ep in train_eps}
            gated_train, freq = apply_static_consensus_gating(
                train_res, train_frms, bg, pcb_mask=pcb_mask
            )
            results_by_ep.update(gated_train)
            np.savez_compressed(
                out / "teachers" / f"sam3_static_consensus_{args.camera}.npz",
                static_freq=freq,
                pcb_mask=pcb_mask if pcb_mask is not None else np.zeros(freq.shape, dtype=np.uint8),
            )

    summaries = []
    for ep in episodes:
        summary = save_teacher_outputs(
            results_by_ep[ep], frames_by_ep[ep], out / "teachers", save_overlays_every=args.overlay_every
        )
        summaries.append({k: v for k, v in summary.items() if k != "frames"})
    (out / "teachers" / f"{args.teacher}_{args.camera}_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Finished {args.teacher} labeling for {len(episodes)} episodes on {args.camera}")


def cmd_train(args: argparse.Namespace) -> None:
    out = Path(args.result_dir)
    split = load_split(args.split) if args.split else make_episode_split(args.pack_path, camera=args.camera)
    teacher_root = Path(args.teacher_root) if args.teacher_root else (out / "teachers")

    train_eps = list(args.train_episodes) if args.train_episodes else split.train
    val_eps = list(args.val_episodes) if args.val_episodes else split.val

    # Optional pose-balanced subsampling per training episode
    frame_filter = None
    if args.frames_per_episode is not None:
        frame_filter = {}
        for ep in train_eps:
            ids, frames = read_video_frames(camera_video_path(args.pack_path, ep, args.camera))
            chosen = sample_pose_balanced_indices(
                frames, count=args.frames_per_episode, pause_fraction=args.pause_fraction, seed=args.seed
            )
            frame_filter[ep] = [ids[i] for i in chosen]

    tr_imgs, tr_lbls, tr_recs = load_teacher_samples(
        args.pack_path, teacher_root, args.teacher, args.camera, train_eps, frame_filter=frame_filter
    )
    va_imgs, va_lbls, _ = load_teacher_samples(
        args.pack_path, teacher_root, args.teacher, args.camera, val_eps
    )

    run_dir = out / "models" / f"{args.arch}_{args.teacher}_{args.camera}"
    if args.arch == "yolo26_instance":
        res = train_yolo_instance_student(
            tr_imgs,
            tr_lbls,
            va_imgs,
            va_lbls,
            out_dir=run_dir,
            weights_or_cfg=args.weights,
            camera_id=args.camera,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
    else:
        arch_kwargs = (
            {"base_channels": args.base_channels}
            if args.arch == "unet"
            else {"weights_or_cfg": args.weights, "decoder_channels": args.decoder_channels}
        )
        res = train_semantic_student(
            tr_imgs,
            tr_lbls,
            va_imgs,
            va_lbls,
            out_dir=run_dir,
            arch=args.arch,
            arch_kwargs=arch_kwargs,
            camera_id=args.camera,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            split=split.to_dict(),
            train_records=tr_recs,
        )
    print(f"Training finished -> {run_dir} ({json.dumps({k: v for k, v in res.items() if k not in ('history', 'train_records')})})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    pred = load_segmenter(args.artifact_dir, device=args.device)
    summary = evaluate_predictor(
        predictor=pred,
        eval_manifest_path=args.eval_manifest,
        partition=args.partition,
        out_dir=args.result_dir,
        tolerance_px=args.tolerance_px,
        input_size=args.input_size,
        worst_k=args.worst_k,
        pack_path_for_pause=args.pack_path,
    )
    print(json.dumps(summary["overall"], indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    preds = [load_segmenter(a, device=args.device) for a in args.artifacts]
    report = run_paced_benchmark(
        predictors=preds,
        video_paths=args.videos,
        duration_s=args.duration_s,
        target_hz=args.hz,
        warmup_ticks=args.warmup_ticks,
        downstream_ms=args.downstream_ms,
        out_path=args.out_json,
    )
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone AL5D robot segmentation experiment CLI")
    sub = p.add_subparsers(dest="command", required=True)

    s_inv = sub.add_parser("inventory", help="Step 1: inventory episodes, freeze 6/2/2 split, select eval frames")
    s_inv.add_argument("--pack-path", required=True)
    s_inv.add_argument("--result-dir", required=True)
    s_inv.add_argument("--cameras", nargs="+", default=["dev2", "dev3"])
    s_inv.add_argument("--eval-frames-per-episode", type=int, default=12)
    s_inv.add_argument("--extract-eval", action="store_true")
    s_inv.add_argument("--seed", type=int, default=0)
    s_inv.set_defaults(func=cmd_inventory)

    s_base = sub.add_parser("baseline", help="Step 2: fit median-background and static control")
    s_base.add_argument("--pack-path", required=True)
    s_base.add_argument("--result-dir", required=True)
    s_base.add_argument("--camera", default="dev2")
    s_base.add_argument("--split", default=None)
    s_base.add_argument("--frame-step", type=int, default=10)
    s_base.add_argument("--threshold", type=float, default=0.12)
    s_base.add_argument("--dilate", type=int, default=1)
    s_base.add_argument("--overlay-step", type=int, default=20)
    s_base.set_defaults(func=cmd_baseline)

    s_lbl = sub.add_parser("label", help="Step 3: run offline teacher (motion, sam2, sam3)")
    s_lbl.add_argument("--pack-path", required=True)
    s_lbl.add_argument("--result-dir", required=True)
    s_lbl.add_argument("--teacher", choices=["motion", "sam2", "sam3"], default="motion")
    s_lbl.add_argument("--camera", default="dev2")
    s_lbl.add_argument("--split", default=None)
    s_lbl.add_argument("--bg-dir", default=None)
    s_lbl.add_argument("--episodes", nargs="*", default=None)
    s_lbl.add_argument("--step", type=int, default=1)
    s_lbl.add_argument("--max-frames", type=int, default=None)
    s_lbl.add_argument("--overlay-every", type=int, default=25)
    s_lbl.add_argument("--device", default="auto")
    s_lbl.add_argument("--seed", type=int, default=0)
    s_lbl.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    s_lbl.add_argument("--sam2-ckpt", default="sam2.1_hiera_tiny.pt")
    s_lbl.add_argument("--no-box", action="store_true")
    s_lbl.add_argument("--sam3-prompt", default="robot arm")
    s_lbl.add_argument("--sam3-ckpt", default=None)
    s_lbl.add_argument("--sam3-conf", type=float, default=0.25)
    s_lbl.add_argument("--sam3-imgsz", type=int, default=644)
    s_lbl.add_argument("--sam3-selection-mode", choices=["union", "consistent"], default="union")
    s_lbl.add_argument("--sam3-pcb-prompts", nargs="+", default=["circuit board"])
    s_lbl.add_argument("--sam3-pcb-conf", type=float, default=0.50)
    s_lbl.add_argument("--no-sam3-stabilize", action="store_true")
    s_lbl.set_defaults(func=cmd_label)

    s_tr = sub.add_parser("train", help="Steps 4-5: train single-frame student")
    s_tr.add_argument("--pack-path", required=True)
    s_tr.add_argument("--result-dir", required=True)
    s_tr.add_argument("--teacher", choices=["motion", "sam2", "sam3"], default="motion")
    s_tr.add_argument("--teacher-root", default=None)
    s_tr.add_argument("--arch", choices=["unet", "yolo26_semantic", "yolo26_instance"], default="unet")
    s_tr.add_argument("--weights", default="yolo26n-seg.yaml")
    s_tr.add_argument("--camera", default="dev2")
    s_tr.add_argument("--split", default=None)
    s_tr.add_argument("--train-episodes", nargs="*", default=None)
    s_tr.add_argument("--val-episodes", nargs="*", default=None)
    s_tr.add_argument("--input-size", type=int, default=256)
    s_tr.add_argument("--epochs", type=int, default=15)
    s_tr.add_argument("--batch-size", type=int, default=16)
    s_tr.add_argument("--lr", type=float, default=1e-3)
    s_tr.add_argument("--base-channels", type=int, default=24)
    s_tr.add_argument("--decoder-channels", type=int, default=32)
    s_tr.add_argument("--frames-per-episode", type=int, default=None)
    s_tr.add_argument("--pause-fraction", type=float, default=0.2)
    s_tr.add_argument("--seed", type=int, default=0)
    s_tr.add_argument("--device", default="auto")
    s_tr.set_defaults(func=cmd_train)

    s_ev = sub.add_parser("evaluate", help="Steps 4-7: score saved predictor against reference masks")
    s_ev.add_argument("--artifact-dir", required=True)
    s_ev.add_argument("--eval-manifest", required=True)
    s_ev.add_argument("--partition", choices=["val", "test_sealed"], default="val")
    s_ev.add_argument("--result-dir", required=True)
    s_ev.add_argument("--tolerance-px", type=int, default=2)
    s_ev.add_argument("--input-size", type=int, default=None)
    s_ev.add_argument("--worst-k", type=int, default=8)
    s_ev.add_argument("--pack-path", default=None)
    s_ev.add_argument("--device", default="auto")
    s_ev.set_defaults(func=cmd_evaluate)

    s_bm = sub.add_parser("benchmark", help="Steps 2 & 7: paced 10 Hz latency & freshness benchmark")
    s_bm.add_argument("--artifacts", nargs="+", required=True)
    s_bm.add_argument("--videos", nargs="+", required=True)
    s_bm.add_argument("--duration-s", type=float, default=10.0)
    s_bm.add_argument("--hz", type=float, default=10.0)
    s_bm.add_argument("--warmup-ticks", type=int, default=10)
    s_bm.add_argument("--downstream-ms", type=float, default=0.0)
    s_bm.add_argument("--device", default="auto")
    s_bm.add_argument("--out-json", default=None)
    s_bm.set_defaults(func=cmd_benchmark)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
