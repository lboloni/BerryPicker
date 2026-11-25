"""Paced 10 Hz latency, freshness, drop, and queue-growth benchmark (Steps 2 & 7)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


def _stats(vals: Sequence[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) == 0:
        return {"median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "median": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _open_sources(video_paths: Sequence[str | Path]) -> list[cv2.VideoCapture]:
    caps = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            for c in caps:
                c.release()
            raise RuntimeError(f"Cannot open benchmark video source: {vp}")
        caps.append(cap)
    return caps


def _read_next_rgb(cap: cv2.VideoCapture) -> np.ndarray:
    ok, bgr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from benchmark video source")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_paced_benchmark(
    predictors: Sequence[Any],
    video_paths: Sequence[str | Path],
    duration_s: float = 10.0,
    target_hz: float = 10.0,
    warmup_ticks: int = 10,
    downstream_ms: float = 0.0,
    out_path: str | Path | None = None,
) -> dict:
    """Benchmark 1 or 2 camera streams paced at ``target_hz`` (default 10 Hz / 100 ms tick).

    Carries ``frame_id`` and ``capture_ts`` through mask delivery so stale frames,
    lateness, queue growth, dropped ticks, and deadline misses are measured explicitly.
    """
    if len(predictors) != len(video_paths) or len(predictors) == 0:
        raise ValueError("predictors and video_paths must be non-empty and have matching lengths")

    period_s = 1.0 / float(target_hz)
    deadline_ms = period_s * 1000.0
    caps = _open_sources(video_paths)
    try:
        # Measure cold start (first tick before warm-up)
        t_cold0 = time.perf_counter()
        first_frames = [_read_next_rgb(c) for c in caps]
        for pred, rgb in zip(predictors, first_frames):
            pred.segment_timed(rgb, frame_id=0, capture_ts=t_cold0)
        cold_start_ms = (time.perf_counter() - t_cold0) * 1000.0

        # Warm-up (not included in paced stats)
        for w in range(warmup_ticks):
            for pred, cap in zip(predictors, caps):
                rgb = _read_next_rgb(cap)
                pred.segment_timed(rgb, frame_id=w + 1, capture_ts=time.perf_counter())

        total_ticks = max(1, int(round(duration_s * target_hz)))
        start_time = time.perf_counter()

        capture_ms_list = []
        preprocess_ms_list = []
        transfer_ms_list = []
        inference_ms_list = []
        resize_ms_list = []
        segment_total_ms_list = []
        tick_total_ms_list = []
        age_ms_list = []
        lateness_ms_list = []
        queue_depths = []

        deadline_misses = 0
        dropped_frames = 0
        next_tick = 0

        while next_tick < total_ticks:
            scheduled_ts = start_time + next_tick * period_s
            now = time.perf_counter()
            if now < scheduled_ts:
                time.sleep(scheduled_ts - now)
                now = time.perf_counter()

            # Compute backlog (how many ticks have arrived while waiting/processing)
            late_ticks = int((now - scheduled_ts) // period_s)
            queue_depths.append(late_ticks)
            if late_ticks > 0:
                dropped_frames += late_ticks
                deadline_misses += late_ticks
                next_tick += late_ticks
                if next_tick >= total_ticks:
                    break
                scheduled_ts = start_time + next_tick * period_s

            tick_t0 = time.perf_counter()
            frames = [_read_next_rgb(c) for c in caps]
            cap_ms = (time.perf_counter() - tick_t0) * 1000.0

            pre_sum = tr_sum = inf_sum = res_sum = seg_sum = 0.0
            for pred, rgb in zip(predictors, frames):
                _, timing = pred.segment_timed(rgb, frame_id=next_tick, capture_ts=scheduled_ts)
                pre_sum += timing["preprocess_ms"]
                tr_sum += timing["transfer_ms"]
                inf_sum += timing["inference_ms"]
                res_sum += timing["resize_ms"]
                seg_sum += timing["total_ms"]

            if downstream_ms > 0.0:
                time.sleep(downstream_ms / 1000.0)

            deliver_ts = time.perf_counter()
            tick_ms = (deliver_ts - tick_t0) * 1000.0
            age_ms = (deliver_ts - scheduled_ts) * 1000.0
            lateness_ms = max(0.0, age_ms - deadline_ms)

            capture_ms_list.append(cap_ms)
            preprocess_ms_list.append(pre_sum)
            transfer_ms_list.append(tr_sum)
            inference_ms_list.append(inf_sum)
            resize_ms_list.append(res_sum)
            segment_total_ms_list.append(seg_sum)
            tick_total_ms_list.append(tick_ms)
            age_ms_list.append(age_ms)
            lateness_ms_list.append(lateness_ms)

            if age_ms > deadline_ms:
                deadline_misses += 1
            next_tick += 1

        max_queue = int(max(queue_depths)) if queue_depths else 0
        queue_growth = int(queue_depths[-1] - queue_depths[0]) if len(queue_depths) >= 2 else 0
        observed_pass = (deadline_misses == 0) and (dropped_frames == 0) and (max_queue == 0)

        report = {
            "camera_count": len(predictors),
            "target_hz": float(target_hz),
            "deadline_ms": deadline_ms,
            "duration_s": float(duration_s),
            "scheduled_ticks": total_ticks,
            "completed_ticks": len(tick_total_ms_list),
            "cold_start_ms": cold_start_ms,
            "downstream_simulated_ms": float(downstream_ms),
            "capture_ms": _stats(capture_ms_list),
            "preprocess_ms": _stats(preprocess_ms_list),
            "transfer_ms": _stats(transfer_ms_list),
            "inference_ms": _stats(inference_ms_list),
            "resize_ms": _stats(resize_ms_list),
            "segment_total_ms": _stats(segment_total_ms_list),
            "tick_total_ms": _stats(tick_total_ms_list),
            "frame_age_ms": _stats(age_ms_list),
            "max_lateness_ms": float(max(lateness_ms_list)) if lateness_ms_list else 0.0,
            "deadline_misses": int(deadline_misses),
            "dropped_frames": int(dropped_frames),
            "max_queue_depth": max_queue,
            "queue_growth": queue_growth,
            "observed_10hz_pass": bool(observed_pass),
        }
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        for c in caps:
            c.release()
