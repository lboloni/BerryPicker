# Offline hand removal

Open `Flow_HandRemoval.ipynb` with the BerryPicker environment and run all cells.
The default flow makes separate train, validation and test episodes, trains a
small U-Net, evaluates a fork and spoon, and writes a report, CSV, executed
notebooks, lossless cleaned frames, MP4 previews and pixel provenance. The
fixtures are layered synthetic RGB scenes with known landmarks. The auxiliary
camera fixture is a calibrated planar disparity test. A real camera benchmark
requires recordings, calibration and independent annotations.

## Run on an ordinary video

Use a locked target camera. Take a clean background image at the video resolution
and a cutout photograph of the same object on a transparent background. Record
two or more recognizable landmarks in the same order in the reference and on
video keyframes. Three or more well-separated points give the estimator a chance
to reject a bad match. Add another keyframe when the grip or viewing angle changes.
A reference bank can contain several views. Each reference includes its own
coordinates for the same landmarks. Coordinates are native-resolution (x,y).
Use null for an invisible landmark. The notebook can also read supplied per-frame
landmark coordinates in `tracks` in place of sparse `keyframes`.

For example, save the following as clip.json next to the assets. The coordinates
and HSV bounds are illustrative and should be measured on the recording.

```json
{
  "episode": "fork_take_01", "split": "test", "view": "front",
  "rgb": "front.mp4", "fps": 30, "plate": "table.png",
  "references": [{"image": "fork_rgba.png", "points": [[30,48],[76,48],[99,40],[99,56]]}],
  "keyframes": [{"frame": 0, "points": [null,[96,76],[119,68],[119,84]]}],
  "hsv": [[38,110,80],[85,255,255]]
}
```

Set `manifests` in the flow notebook to the JSON paths, `learn=False` and
`multiview=False` for a single test clip. The clean reference can be cropped;
only the target RGB, edit masks and plate must share image size. All image
directories use lexically sorted filenames. Supply frame_ids and timestamps
when available; otherwise output times are index/fps. MP4s are constant-rate
review copies. The lossless frames and JSON preserve source timing.

For bare hands, replace `hsv` with `hand_masks` pointing at a numbered frame
directory (white means edit), or install SAM 2 and supply its config, checkpoint,
device and point prompts as `sam_config`, `sam_checkpoint`, `sam_device`,
`sam_prompts`. Each prompt has frame, points and binary labels. Use SAM-compatible
JPEG frame directories for that backend. Optional `protect_masks` preserve
visible subject pixels. Shadows may be included in a corrected edit mask.
For articulated figures supply `parts`, ordered back to front, each with its
own references and keyframes. Layers use independent similarity transforms.
Unobserved motion and out-of-plane turns without a matching view can be unresolved.

## Learning and synchronized views

Add manifests with split train and validation and paired `rgb` (occluded) and
`clean` (ground truth) sequences to train. Keep all versions of a take in one
split. An explicit subject silhouette is optional for the neural path: use
`methods=[]`, `learn=True` for a learned-only single-view run without references.
The geometric baseline uses the alpha channel of a clean object reference.

An optional `auxiliary` mapping contains RGB source, fps, timestamps (optional),
offset_s (additive, optional), max_skew_s, source K and target_K (3×3),
target_from_source (4×4 rigid transformation), and depth_m (a NumPy array
in metres with dimensions frames × height × width). Use rectified images.
Supply auxiliary hand_masks or HSV bounds. Optional target_depth_m describes
the scene behind the hand; use depth_tolerance_m to gate occlusion conflicts.
plane_depth_m is for known planar scenes only. See the generated clip manifests
for a numerical camera example. Depth holes, reflective surfaces and unresolved
contact must be reviewed. The registered-reference method is available when
depth coverage is poor.

## Reading outputs

Each method directory contains frames/, cleaned.mp4, review.mp4, frames.json
and provenance.npz. In the verification notebook inspect_pixel reports the
origin of an edit. Original, plate, reference, temporal, auxiliary, learned
and unresolved have integer codes 0–6. The donor value for a learned fill refers
to its conditioning candidate. Unresolved pixels use a plate-colored preview;
frames with unresolved edits are excluded by the demonstration exporter.
The comparison's coverage is the fraction of edited pixels assigned an estimate,
including plate fills. Inspect hidden-subject IoU alongside it. The synthetic
silhouette scores use foreground/background color difference and are intended
for controlled scenes or paired evaluation where that model holds.

## Hand-guided robot collection

The demonstration collector recipes are hand_guided_wxai_cameras and
hand_guided_widowx_cameras. Select the matching task-arm binding in the
machine-local settings (see the machine template). The WXAI reader uses its
existing gravity-compensated external-effort teaching mode. For a wx250s
start the manufacturer's supported gravity-compensation session separately
with the arm supported during transitions. The ROS reader subscribes only
to measured joints; verify the joint names against the installed gripper.
Press g/r in the camera preview to annotate grasp/release intent and q to
finish. These events annotate the take. They do not actuate the gripper.

Camera times are host read-completion times. Robot annotations store the samples
between ticks and ROS source stamps where available. Measure clock offset and
USB latency before interpreting contact timing. Use demonstrations.import_view
to create a clip and demonstrations.export_view to publish consecutive accepted
intervals in a new demonstration exp/run. Supply reviewed rejected frame IDs
for ambiguous contact. JPEG in this adapter follows the existing recorder
convention; the method's PNG frames retain the lossless reconstruction.

To train from a clean video and a separate moving occluder, a train/validation
manifest can specify clean, occluder_rgb and occluder_alpha instead of rgb.
Use matched frame sizes and lengths. Supply the alpha sequence as hand_masks
or run the same glove segmentation as for test clips. This produces paired
training images in memory. Separate clean motion clips, real hand overlays and
whole held-out recordings are needed to measure generalization.

Calibration helpers in calibration.py fit intrinsics and the source-to-target
transform from paired chessboards, estimate a clock offset from common LED
events, and triangulate corresponding landmarks with reprojection residuals.
Undistort footage and corresponding masks/depth into the specified pinhole
camera coordinates before projection. A single planar calibration fixture
does not measure performance on a reflective three-dimensional utensil.

For RGB-only synchronized views of a locally planar rigid patch, auxiliary
landmarks can supply source and target lists (one array of 2D points per frame)
and triangles (triplets of landmark indices on the same material). Set
max_reprojection_px to the measured alignment tolerance. The fusion stage
triangulates the matches and transfers visible texture only inside accepted
triangles. Exclude gaps between tines and triangles spanning independently
moving parts. For RGB-D, max_depth_step_m rejects pixels near depth jumps.
Inspect registration.json for the per-frame reference choice, transform and
visible-point residual. The optional learned stage also measures optical-flow
round trips on the clean training clips to constrain temporal inconsistency
on correspondences supported by both frames.

For a hand-guided take, joints_at_exposure(demo, camera, capture_delay_s,
max_gap_s) aligns the measured joints to the camera clock using the measured
capture delay. Pass its output as aligned_joints to export_view. Missing time
brackets and excessive state-sample gaps exclude the frame. Export keeps the
original command and adds taught-joints-at-image. To learn the next waypoint,
take that channel from the subsequent row of the same accepted interval.
