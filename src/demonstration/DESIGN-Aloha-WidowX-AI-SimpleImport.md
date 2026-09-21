# Simple Aloha WidowX-AI Demonstration Import

## Status and intent

This document specifies a deliberately partial and temporary importer for
Aloha WidowX-AI demonstrations. Its purpose is to make the immediately useful
RGB and robot data available through the current BerryPicker
`Demonstration` format with as little new machinery as possible.

The importer is not a complete conversion of the Aloha recording. It discards
data that the current BerryPicker readers do not need. A later importer is
expected to add lossless multimodal data, richer timing, and an explicit Aloha
robot model.

The import is a copy-and-convert operation. It reads an existing Aloha source
directory and creates a new BerryPicker demonstration in a separate target
directory. It never converts, renames, deletes, or adds files in the source
directory.

## Supported source layout

The initial importer supports one Aloha demonstration directory containing:

- `robot_positions.npz`;
- one `camera_<serial>_rgb.mp4` file per camera; and
- optionally present IR videos and depth directories, which are ignored.

The source is expected to contain synchronized 60 Hz video and robot rows. The
known sample contains four RGB videos and 640 rows.

## Retained data

The simple import retains only:

- the RGB stream from every camera;
- every sixth RGB frame, starting with source frame zero;
- the `pair_1_leader` robot vector from the same selected source row, stored as
  the action; and
- the `pair_1_follower` robot vector from the same selected source row, stored
  as observed robot telemetry in the annotation.

Leader data is treated as the commanded action and follower data as the
observed robot state. The importer preserves the seven numeric values without
renaming joints, changing units, normalizing them, or converting them into an
AL5D `rc-position-target` or a Cartesian `widowx-command`.

The proposed action and annotation records are:

```yaml
# _action.yaml
- aloha-pair-1-leader: [value_0, value_1, value_2, value_3, value_4, value_5, value_6]

# _annotation.yaml
- aloha:
    pair_1_follower: [value_0, value_1, value_2, value_3, value_4, value_5, value_6]
```

All values must be converted from NumPy scalars to ordinary Python floats
before YAML serialization.

## Discarded data

This version does not copy or convert:

- depth frames or depth chunk indices;
- IR1 or IR2 video;
- the five out of every six RGB frames omitted during downsampling;
- robot and camera timestamps;
- source camera frame numbers;
- the `synced` array after it has been checked during validation;
- the original `robot_positions.npz`; or
- any source files retained only for provenance.

The source directory therefore remains the authoritative copy of everything
not represented by the partial import.

## Downsampling and synchronization

Let `N` be the common source row and RGB frame count. Output row `i` uses source
row and source video frame:

```text
source_index = 6 * i
```

The selected indices are `range(0, N, 6)`. This converts the nominal rate from
60 Hz to 10 Hz without interpolation or averaging. For the known 640-frame
sample, the selected indices are `0, 6, ..., 636`, producing 107 output rows.
Source frames 637 through 639 are not included.

The four selected RGB frames, leader vector, and follower vector for an output
row must all come from the same source index. The importer must fail instead of
silently shifting or duplicating data when their lengths differ. It must also
fail if a selected row is marked unsynchronized.

## BerryPicker output

The importer takes distinct source and target paths:

```text
python src/demonstration/import_aloha.py \
    <aloha-source-directory> <berrypicker-target-directory>
```

The resolved target path must not equal the source path or be located inside
the source directory. The importer rejects such a request as an attempted
in-place import. The target must not already exist.

The importer creates one standard BerryPicker demonstration at the target:

```text
<demonstration-name>/
    _metadata.yaml
    _action.yaml
    _annotation.yaml
    video_aloha_<serial-1>_rgb.mp4
    video_aloha_<serial-2>_rgb.mp4
    ...
```

Each output video contains only the selected frames, in source order, encoded
at 10 FPS with the source width and height. Camera names are stable and derived
from the hardware serial, for example `aloha_409122272325_rgb`.

The metadata uses the fields already consumed by `Demonstration` and adds a
small warning/provenance section:

```yaml
cameras:
  - aloha_409122272325_rgb
  - aloha_409122273431_rgb
  - aloha_409122274550_rgb
  - aloha_409122274697_rgb
maxsteps: 107
stored_as_images: false
stored_as_video: true

import:
  source_format: aloha-widowx-ai
  importer: simple-partial-v1
  partial: true
  temporary: true
  source_fps: 60
  output_fps: 10
  source_frame_stride: 6
  source_frame_count: 640
  retained_modalities: [rgb, pair_1_leader, pair_1_follower]
  discarded_modalities: [depth, ir1, ir2, timestamps, frame_numbers]
```

The counts in this example are derived from the known sample. The importer
must calculate them from each source rather than hard-code them.

## Minimal import procedure

1. Resolve separate source and target paths, and reject an in-place or nested
   target.
2. Verify that the source exists and the target does not.
3. Load the source `robot_positions.npz` without allowing pickled objects.
4. Verify that `pair_1_leader` and `pair_1_follower` both have shape `(N, 7)`.
5. Discover and sort the source RGB videos by camera serial.
6. Verify that every RGB video is readable and contains exactly `N` frames.
7. If `synced` is present, verify that it has length `N` and that every
   selected row is true.
8. Select source indices `0, 6, 12, ...`.
9. Create a temporary directory alongside the requested target.
10. Write one 10 FPS output video per camera into the temporary directory using
    exactly the selected source frames.
11. Write the selected leader rows to `_action.yaml` and follower rows to
   `_annotation.yaml`.
12. Write metadata that explicitly marks the import partial and temporary.
13. Load the temporary result through `Demonstration` and verify the first and
    last output frames, action count, annotation count, and `maxsteps`.
14. Rename the validated temporary directory to the requested target.

If conversion or validation fails, the incomplete temporary output is removed
and the source remains unchanged. The importer does not overwrite an existing
target demonstration.

## Explicit limitations

- The imported data is lossy in time and modality.
- Original timestamps cannot be reconstructed from the imported directory.
- Depth and IR cannot be recovered unless the original Aloha directory is
  retained separately.
- The seven robot dimensions remain unnamed and unnormalized.
- Existing training code that requires `rc-position-target` or
  `widowx-command` will need an Aloha-specific action adapter before it can use
  these robot vectors.
- The import assumes the source video's frame index corresponds directly to
  the same row in `robot_positions.npz`.
- The source directory must remain available until the copy-and-convert
  operation completes, but the completed target is independent of it.

## Planned extension

A future full importer may preserve original timestamps and frame numbers,
support raw 16-bit depth and IR, define named WidowX-AI joints and limits,
provide normalization and training adapters, and offer configurable temporal
resampling. Those extensions are intentionally outside the simple importer.
