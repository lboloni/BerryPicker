"""Import a partial Aloha WidowX-AI recording into BerryPicker format.

This intentionally simple importer keeps RGB, leader positions, and follower
positions at 10 Hz. It discards depth, IR, and detailed source timing. See
DESIGN-Aloha-WidowX-AI-SimpleImport.md for the format and its limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import pathlib
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np
import yaml


SOURCE_FPS = 60
OUTPUT_FPS = 10
FRAME_STRIDE = 6
ROBOT_WIDTH = 7
RGB_VIDEO_PATTERN = re.compile(
    r"^camera_(?P<serial>[A-Za-z0-9.-]+)_rgb\.mp4$"
)


class AlohaImportError(ValueError):
    """Raised when an Aloha recording cannot be imported safely."""


@dataclass(frozen=True)
class SourceVideo:
    """Validated source RGB stream."""

    serial: str
    path: pathlib.Path
    frame_count: int
    width: int
    height: int
    fps: float

    @property
    def camera_name(self):
        return f"aloha_{self.serial}_rgb"


@dataclass(frozen=True)
class SourceRecording:
    """The subset of an Aloha recording used by the simple importer."""

    source: pathlib.Path
    leader: np.ndarray
    follower: np.ndarray
    selected_indices: np.ndarray
    videos: tuple[SourceVideo, ...]

    @property
    def source_frame_count(self):
        return int(self.leader.shape[0])

    @property
    def output_frame_count(self):
        return int(self.selected_indices.size)


class _ImportedDemonstrationExperiment(dict):
    """Minimal experiment object needed to load a staged Demonstration."""

    def __init__(self, data_directory):
        super().__init__()
        self._data_directory = pathlib.Path(data_directory)

    def data_dir(self):
        return self._data_directory


def _is_within(path, directory):
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _resolve_paths(source, target):
    source = pathlib.Path(source).expanduser().resolve(strict=True)
    target = pathlib.Path(target).expanduser().resolve(strict=False)

    if not source.is_dir():
        raise AlohaImportError(f"Aloha source is not a directory: {source}")
    if target == source or _is_within(target, source):
        raise AlohaImportError(
            "Target must be separate from, and not inside, the Aloha source"
        )
    if target.exists():
        raise AlohaImportError(f"Target already exists: {target}")
    if not target.parent.is_dir():
        raise AlohaImportError(
            f"Target parent directory does not exist: {target.parent}"
        )
    return source, target


def _load_robot_data(source):
    positions_path = source / "robot_positions.npz"
    if not positions_path.is_file():
        raise AlohaImportError(f"Missing robot positions: {positions_path}")

    try:
        with np.load(positions_path, allow_pickle=False) as positions:
            missing = {
                "pair_1_leader", "pair_1_follower"
            } - set(positions.files)
            if missing:
                raise AlohaImportError(
                    f"Robot positions are missing arrays: {sorted(missing)}"
                )
            leader = np.asarray(positions["pair_1_leader"]).copy()
            follower = np.asarray(positions["pair_1_follower"]).copy()
            synced = (
                np.asarray(positions["synced"]).copy()
                if "synced" in positions.files
                else None
            )
            recorded_fps = (
                np.asarray(positions["fps"])
                if "fps" in positions.files
                else None
            )
    except (OSError, ValueError) as error:
        if isinstance(error, AlohaImportError):
            raise
        raise AlohaImportError(
            f"Could not read robot positions: {positions_path}"
        ) from error

    if leader.ndim != 2 or leader.shape[1] != ROBOT_WIDTH:
        raise AlohaImportError(
            f"pair_1_leader must have shape (N, {ROBOT_WIDTH}); "
            f"found {leader.shape}"
        )
    if follower.shape != leader.shape:
        raise AlohaImportError(
            f"pair_1_follower must have shape {leader.shape}; "
            f"found {follower.shape}"
        )
    if leader.shape[0] == 0:
        raise AlohaImportError("Aloha recording contains no robot rows")
    if not np.issubdtype(leader.dtype, np.number) or not np.issubdtype(
        follower.dtype, np.number
    ):
        raise AlohaImportError("Leader and follower arrays must be numeric")
    if not np.isfinite(leader).all() or not np.isfinite(follower).all():
        raise AlohaImportError("Leader and follower arrays must be finite")

    if recorded_fps is not None:
        if recorded_fps.shape != ():
            raise AlohaImportError("fps must be a scalar")
        if not math.isclose(float(recorded_fps), SOURCE_FPS, abs_tol=0.01):
            raise AlohaImportError(
                f"Expected {SOURCE_FPS} Hz robot data; found "
                f"{float(recorded_fps):g} Hz"
            )

    selected_indices = np.arange(0, leader.shape[0], FRAME_STRIDE, dtype=int)
    if synced is not None:
        if synced.shape != (leader.shape[0],):
            raise AlohaImportError(
                f"synced must have shape ({leader.shape[0]},); "
                f"found {synced.shape}"
            )
        unsynchronized = selected_indices[~synced[selected_indices].astype(bool)]
        if unsynchronized.size:
            raise AlohaImportError(
                "Selected source rows are not synchronized: "
                f"{unsynchronized.tolist()}"
            )

    return leader, follower, selected_indices


def _probe_video(path, serial, expected_frames):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise AlohaImportError(f"Could not open RGB video: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        readable, frame = capture.read()
    finally:
        capture.release()

    if frame_count != expected_frames:
        raise AlohaImportError(
            f"RGB video {path.name} has {frame_count} frames; "
            f"expected {expected_frames}"
        )
    if width <= 0 or height <= 0 or not readable or frame is None:
        raise AlohaImportError(f"RGB video has no readable frames: {path}")
    if frame.shape[:2] != (height, width):
        raise AlohaImportError(
            f"RGB video {path.name} reports size {(width, height)} but its "
            f"first frame has size {(frame.shape[1], frame.shape[0])}"
        )
    if not math.isfinite(fps) or not math.isclose(
        fps, SOURCE_FPS, abs_tol=0.1
    ):
        raise AlohaImportError(
            f"RGB video {path.name} must be {SOURCE_FPS} FPS; found {fps:g}"
        )
    return SourceVideo(serial, path, frame_count, width, height, fps)


def _discover_videos(source, expected_frames):
    matches = []
    for path in source.iterdir():
        if not path.is_file():
            continue
        match = RGB_VIDEO_PATTERN.fullmatch(path.name)
        if match:
            matches.append((match.group("serial"), path))
    if not matches:
        raise AlohaImportError(f"No camera_<serial>_rgb.mp4 files in {source}")

    matches.sort(key=lambda item: item[0])
    serials = [serial for serial, _ in matches]
    if len(serials) != len(set(serials)):
        raise AlohaImportError("Aloha source contains duplicate RGB camera serials")
    return tuple(
        _probe_video(path, serial, expected_frames) for serial, path in matches
    )


def inspect_source(source):
    """Validate and return the source data used by the simple importer."""
    source = pathlib.Path(source).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise AlohaImportError(f"Aloha source is not a directory: {source}")
    leader, follower, selected_indices = _load_robot_data(source)
    videos = _discover_videos(source, leader.shape[0])
    return SourceRecording(
        source=source,
        leader=leader,
        follower=follower,
        selected_indices=selected_indices,
        videos=videos,
    )


def _write_video(source_video, target_path, selected_indices):
    capture = cv2.VideoCapture(str(source_video.path))
    if not capture.isOpened():
        capture.release()
        raise AlohaImportError(f"Could not reopen RGB video: {source_video.path}")

    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (source_video.width, source_video.height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise AlohaImportError(f"Could not create output video: {target_path}")

    selected = set(int(index) for index in selected_indices)
    written = 0
    try:
        for source_index in range(source_video.frame_count):
            readable, frame = capture.read()
            if not readable or frame is None:
                raise AlohaImportError(
                    f"Could not decode frame {source_index} from "
                    f"{source_video.path.name}"
                )
            if frame.shape[:2] != (source_video.height, source_video.width):
                raise AlohaImportError(
                    f"Frame {source_index} in {source_video.path.name} changed "
                    "dimensions"
                )
            if source_index in selected:
                writer.write(frame)
                written += 1
    finally:
        capture.release()
        writer.release()

    if written != len(selected_indices):
        raise AlohaImportError(
            f"Wrote {written} frames from {source_video.path.name}; "
            f"expected {len(selected_indices)}"
        )


def _make_actions(recording):
    return [
        {
            "aloha-pair-1-leader": [
                float(value) for value in recording.leader[index]
            ]
        }
        for index in recording.selected_indices
    ]


def _make_annotations(recording):
    return [
        {
            "aloha": {
                "pair_1_follower": [
                    float(value) for value in recording.follower[index]
                ]
            }
        }
        for index in recording.selected_indices
    ]


def _make_metadata(recording):
    return {
        "cameras": [video.camera_name for video in recording.videos],
        "maxsteps": recording.output_frame_count,
        "stored_as_images": False,
        "stored_as_video": True,
        "import": {
            "source_format": "aloha-widowx-ai",
            "importer": "simple-partial-v1",
            "partial": True,
            "temporary": True,
            "source_fps": SOURCE_FPS,
            "output_fps": OUTPUT_FPS,
            "source_frame_stride": FRAME_STRIDE,
            "source_frame_count": recording.source_frame_count,
            "retained_modalities": [
                "rgb", "pair_1_leader", "pair_1_follower"
            ],
            "discarded_modalities": [
                "depth", "ir1", "ir2", "timestamps", "frame_numbers"
            ],
        },
    }


def _write_yaml(path, value):
    with path.open("w", encoding="utf-8") as output:
        yaml.safe_dump(value, output, sort_keys=False)


def _load_demonstration_class():
    if __package__:
        from demonstration.demonstration import Demonstration
    else:
        source_root = str(pathlib.Path(__file__).resolve().parents[1])
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from demonstration import Demonstration
    return Demonstration


def _verify_output(staging, recording, actions, annotations):
    Demonstration = _load_demonstration_class()
    experiment = _ImportedDemonstrationExperiment(staging.parent)
    demonstration = Demonstration(experiment, staging.name)

    expected_count = recording.output_frame_count
    if demonstration.metadata.get("maxsteps") != expected_count:
        raise AlohaImportError("Imported maxsteps does not match selected rows")
    if demonstration.metadata.get("cameras") != [
        video.camera_name for video in recording.videos
    ]:
        raise AlohaImportError("Imported camera metadata is incorrect")
    if demonstration.actions != actions:
        raise AlohaImportError("Imported actions do not match selected leader rows")
    if demonstration.annotations != annotations:
        raise AlohaImportError(
            "Imported annotations do not match selected follower rows"
        )

    for video in recording.videos:
        output_path = demonstration.get_video_path(video.camera_name)
        capture = cv2.VideoCapture(str(output_path))
        try:
            if not capture.isOpened():
                raise AlohaImportError(f"Could not open imported video: {output_path}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            first_readable, first = capture.read()
            capture.set(cv2.CAP_PROP_POS_FRAMES, expected_count - 1)
            last_readable, last = capture.read()
        finally:
            capture.release()

        if frame_count != expected_count:
            raise AlohaImportError(
                f"Imported video {output_path.name} has {frame_count} frames; "
                f"expected {expected_count}"
            )
        if not math.isclose(fps, OUTPUT_FPS, abs_tol=0.1):
            raise AlohaImportError(
                f"Imported video {output_path.name} is {fps:g} FPS; "
                f"expected {OUTPUT_FPS}"
            )
        if (
            not first_readable or first is None
            or not last_readable or last is None
        ):
            raise AlohaImportError(
                f"Could not read first and last frames from {output_path.name}"
            )


def import_aloha(source, target):
    """Copy and partially convert an Aloha recording into a new target."""
    source, target = _resolve_paths(source, target)
    recording = inspect_source(source)
    actions = _make_actions(recording)
    annotations = _make_annotations(recording)
    metadata = _make_metadata(recording)

    staging = pathlib.Path(tempfile.mkdtemp(
        prefix=f".{target.name}.import-", dir=target.parent
    ))
    published = False
    try:
        for video in recording.videos:
            output_path = staging / f"video_{video.camera_name}.mp4"
            _write_video(video, output_path, recording.selected_indices)
        _write_yaml(staging / "_metadata.yaml", metadata)
        _write_yaml(staging / "_action.yaml", actions)
        _write_yaml(staging / "_annotation.yaml", annotations)
        _verify_output(staging, recording, actions, annotations)

        if target.exists():
            raise AlohaImportError(f"Target appeared during import: {target}")
        staging.rename(target)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target


def _argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Partially import a 60 Hz Aloha WidowX-AI recording as a new "
            "10 Hz BerryPicker demonstration. The source is never modified."
        )
    )
    parser.add_argument("source", help="Existing Aloha demonstration directory")
    parser.add_argument("target", help="New BerryPicker demonstration directory")
    return parser


def main(argv=None):
    args = _argument_parser().parse_args(argv)
    try:
        target = import_aloha(args.source, args.target)
    except (AlohaImportError, FileNotFoundError) as error:
        raise SystemExit(f"Import failed: {error}") from error
    print(f"Imported partial Aloha demonstration to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
