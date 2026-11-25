"""Mask, register, and fill a moving subject in its target camera coordinates."""
from copy import copy

import cv2
import numpy as np

from .clip import frames, still

ORIGINAL, PLATE, REFERENCE, TEMPORAL, AUXILIARY, LEARNED, UNKNOWN = range(7)


def hsv_masks(images, bounds):
    return np.stack([cv2.inRange(cv2.cvtColor(im, cv2.COLOR_BGR2HSV),
                                 np.array(bounds[0]), np.array(bounds[1])) > 0 for im in images])


def masks(clip):
    spec = clip.spec
    if "hand_masks" in spec:
        result = clip.series("hand_masks")[..., 0] > 0
    elif "sam_prompts" in spec:
        from sam2.build_sam import build_sam2_video_predictor
        import torch
        predictor = build_sam2_video_predictor(spec["sam_config"], str(clip.root/spec["sam_checkpoint"]),
                                               device=spec["sam_device"])
        result = np.zeros(clip.rgb.shape[:3], bool)
        with torch.inference_mode():
            state = predictor.init_state(video_path=str(clip.root/spec["rgb"]))
            for prompt in spec["sam_prompts"]:
                predictor.add_new_points_or_box(inference_state=state, frame_idx=prompt["frame"],
                                                obj_id=1, points=np.array(prompt["points"]),
                                                labels=np.array(prompt["labels"]))
            for reverse in (False, True):
                for t, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                    result[t] |= logits[0, 0].cpu().numpy() > 0
    else:
        result = hsv_masks(clip.rgb, spec["hsv"])
    if "protect_masks" in spec:
        result &= ~(clip.series("protect_masks")[..., 0] > 0)
    return result


def tracked_points(clip, hand):
    """Track sparse manual keyframes in both directions, retaining visible correspondences."""
    if "tracks" in clip.spec:
        return np.asarray([[p if p is not None else [np.nan]*2 for p in row]
                           for row in clip.spec["tracks"]], np.float32)
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in clip.rgb]
    size = len(clip.spec["references"][0]["points"])
    tracks = np.full((len(gray), size, 2), np.nan, np.float32)
    distance = np.full(tracks.shape[:2], np.inf)
    for key in clip.spec["keyframes"]:
        start = key["frame"]
        for step in (1, -1):
            pts = np.float32([p if p is not None else [np.nan, np.nan] for p in key["points"]]).reshape(size, 2)
            valid = np.isfinite(pts).all(1)
            pts = np.nan_to_num(pts)
            for i in range(start, len(gray) if step == 1 else -1, step):
                if i != start:
                    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(gray[i-step], gray[i], pts, None)
                    back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i-step], nxt, None)
                    valid &= ok.ravel().astype(bool) & ok2.ravel().astype(bool)
                    valid &= np.linalg.norm(back-pts, axis=1) < 1.5
                    pts = nxt
                x, y = np.rint(pts).astype(int).T
                h, w = gray[i].shape
                valid &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
                valid &= ~hand[i, y.clip(0,h-1), x.clip(0,w-1)]
                use = np.logical_and(valid, abs(i-start) < distance[i])
                tracks[i,use] = pts[use]
                distance[i,use] = abs(i-start)
                if not valid.any():
                    break
    return tracks


def register(clip, hand):
    tracks = tracked_points(clip, hand)
    references = [(still(clip.root/r["image"], cv2.IMREAD_UNCHANGED),
                   np.float32(r["points"])) for r in clip.spec["references"]]
    aligned = []
    for points in tracks:
        valid = np.isfinite(points).all(1)
        best = None
        for k, (rgba, template) in enumerate(references):
            if valid.sum() < 2:
                continue
            transform, inliers = cv2.estimateAffinePartial2D(template[valid], points[valid])
            if transform is None or inliers.sum() < max(2, valid.sum()*.6):
                continue
            residual = cv2.transform(template[None,valid], transform)[0] - points[valid]
            error = float(np.mean(np.linalg.norm(residual, axis=1)))
            if error <= clip.spec.get("max_registration_px", 3) and (best is None or error < best[0]):
                best = (error, k, transform, rgba)
        aligned.append(best)
    return aligned


def interpolate_pose(poses, times, index):
    """Interpolate a rigid 2-D preview between observations of the same reference."""
    before = next((j for j in range(index-1, -1, -1) if poses[j] is not None), None)
    after = next((j for j in range(index+1, len(poses)) if poses[j] is not None), None)
    if before is None or after is None or poses[before][1] != poses[after][1]:
        return None
    fraction = (times[index]-times[before])/(times[after]-times[before])
    _, k, a, ref = poses[before]
    b = poses[after][2]
    angles = np.unwrap([np.arctan2(m[1,0], m[0,0]) for m in (a,b)])
    angle = np.interp(fraction, [0,1], angles)
    scale = (1-fraction)*np.linalg.norm(a[:,0]) + fraction*np.linalg.norm(b[:,0])
    linear = scale*np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    matrix = np.column_stack((linear, (1-fraction)*a[:,2]+fraction*b[:,2]))
    return None, k, matrix, ref


def project_auxiliary(clip):
    """Z-buffer calibrated RGB-D into the target camera; depth is optical-axis Z in metres."""
    spec = clip.spec["auxiliary"]
    images = frames(clip.root/spec["rgb"])
    blocked = (frames(clip.root/spec["hand_masks"])[...,0] > 0 if "hand_masks" in spec
               else hsv_masks(images, spec["hsv"]))
    times = np.asarray(spec.get("timestamps", np.arange(len(images))/spec["fps"])) + spec.get("offset_s", 0)
    if "landmarks" in spec:
        from .calibration import triangulate_landmarks
        # Transfer only the supported triangles of an observed rigid surface.
        # Each triangle must lie on one material patch (exclude tine gaps).
        h,w = clip.rgb.shape[1:3]
        for i, stamp in enumerate(clip.times):
            j = int(np.argmin(abs(times-stamp)))
            out = np.zeros((h,w,3), np.uint8)
            valid = np.zeros((h,w), bool)
            if abs(times[j]-stamp) <= spec["max_skew_s"]:
                a = np.float32(spec["landmarks"]["source"][j])
                b = np.float32(spec["landmarks"]["target"][i])
                for triangle in spec["landmarks"]["triangles"]:
                    sa,tb = a[triangle],b[triangle]
                    if not np.isfinite(sa).all() or not np.isfinite(tb).all():
                        continue
                    xyz,ea,eb = triangulate_landmarks(sa,tb,spec)
                    if (xyz[:,2] <= 0).any() or max(ea.max(),eb.max()) > spec["max_reprojection_px"]:
                        continue
                    warp = cv2.getAffineTransform(sa,tb)
                    footprint = np.zeros(images[j].shape[:2], np.uint8)
                    cv2.fillConvexPoly(footprint,np.rint(sa).astype(np.int32),1)
                    footprint[blocked[j]] = 0
                    use = cv2.warpAffine(footprint, warp, (w,h), flags=cv2.INTER_NEAREST)>0
                    projected = cv2.warpAffine(images[j],warp,(w,h))
                    out[use] = projected[use]
                    valid[use] = True
            yield out,valid,j
        return
    depth = np.load(clip.root/spec["depth_m"]) if "depth_m" in spec else None
    target_depth = np.load(clip.root/spec["target_depth_m"]) if "target_depth_m" in spec else None
    h,w = clip.rgb.shape[1:3]
    y,x = np.indices(images.shape[1:3])
    rays = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    rays = np.linalg.solve(spec["K"], rays)
    extrinsic = np.asarray(spec["target_from_source"])
    for i, stamp in enumerate(clip.times):
        j = int(np.argmin(abs(times-stamp)))
        out = np.zeros((h,w,3), np.uint8)
        valid = np.zeros((h,w), bool)
        if abs(times[j]-stamp) <= spec["max_skew_s"]:
            z = depth[j].ravel() if depth is not None else np.full(x.size, spec["plane_depth_m"])
            usable = np.logical_not(blocked[j])
            if depth is not None and "max_depth_step_m" in spec:
                local_depth = depth[j].astype(np.float32)
                kernel = np.ones((3,3), np.uint8)
                span = cv2.dilate(local_depth,kernel) - cv2.erode(local_depth,kernel)
                usable = np.logical_and(usable, span <= spec["max_depth_step_m"])
            keep = usable.ravel() & np.isfinite(z) & (z > 0)
            xyz = extrinsic[:3,:3] @ (rays[:,keep]*z[keep]) + extrinsic[:3,3:4]
            p = np.asarray(spec["target_K"]) @ xyz
            front = p[2] > 0
            u,v = np.rint(p[:2,front]/p[2,front]).astype(int)
            ok = (u>=0)&(u<w)&(v>=0)&(v<h)
            indices = (v[ok]*w+u[ok])
            depths = p[2,front][ok]
            colors = images[j].reshape(-1,3)[keep][front][ok]
            order = np.argsort(depths)
            unique, first = np.unique(indices[order], return_index=True)
            nearest = order[first]
            if target_depth is not None:
                # Use a target scene depth estimate excluding the operator.
                limit = target_depth[i].ravel()[unique]
                visible = ~np.isfinite(limit) | (depths[nearest] <= limit + spec["depth_tolerance_m"])
                unique, nearest = unique[visible], nearest[visible]
            out.reshape(-1,3)[unique] = colors[nearest]
            valid.ravel()[unique] = True
        yield out, valid, j


def register_parts(clip, hand):
    """One independently tracked similarity transform per part and target frame."""
    layers = []
    for part in clip.spec.get("parts", [{}]):
        view = copy(clip)
        view.spec = clip.spec | part
        if "references" in view.spec:
            layers.append(register(view, hand))
    return layers


def reconstruct(clip, hand, method, layers=None):
    h,w = clip.rgb.shape[1:3]
    if method not in ("plate","reference","past","bidirectional","multiview"):
        raise ValueError(f"Unknown geometric method {method}")
    if method == "plate":
        layers = []
    elif layers is None:
        layers = register_parts(clip, hand)
    if not layers and method not in ("plate", "multiview"):
        raise ValueError(f"{method} requires a reference and corresponding landmarks")
    auxiliary = list(project_auxiliary(clip)) if method == "multiview" else None
    outputs, labels, donors = [], [], []
    for i, edit in enumerate(hand):
        image = clip.rgb[i].copy()
        source = np.zeros((h,w), np.uint8)
        donor = np.full((h,w), -1, np.int32)
        image[edit] = clip.plate[edit]
        source[edit] = PLATE if method == "plate" or (layers and all(poses[i] is not None for poses in layers)) else UNKNOWN
        for poses in layers:
            pose = poses[i]
            if pose is None and method in ("bidirectional", "multiview"):
                pose = interpolate_pose(poses, clip.times, i)
            if pose is None:
                continue
            _, k, target, ref = pose
            projected = cv2.warpAffine(ref, target, (w,h), flags=cv2.INTER_NEAREST)
            missing = np.logical_and(edit, projected[...,3] > 0)
            image[missing] = projected[missing,:3]
            if poses[i] is None:
                source[missing], donor[missing] = UNKNOWN, k
                continue
            source[missing], donor[missing] = REFERENCE, k
            if method in ("past", "bidirectional", "multiview"):
                candidates = range(i) if method == "past" else range(len(hand))
                for j in sorted(candidates, key=lambda j: abs(i-j)):
                    if not missing.any():
                        break
                    if i == j or poses[j] is None or poses[j][1] != k:
                        continue
                    warp = target @ np.vstack((cv2.invertAffineTransform(poses[j][2]), [0,0,1]))
                    support = cv2.warpAffine(ref[...,3], poses[j][2], (w,h), flags=cv2.INTER_NEAREST)>0
                    candidates_rgb = [(clip.rgb[j], np.logical_not(hand[j]), TEMPORAL, j)]
                    if auxiliary is not None:
                        aux_rgb, aux_valid, aux_id = auxiliary[j]
                        candidates_rgb.append((aux_rgb, aux_valid, AUXILIARY, aux_id))
                    for donor_rgb, donor_valid, origin, frame_id in candidates_rgb:
                        visible = np.logical_and(support, donor_valid).astype(np.uint8)
                        available = cv2.warpAffine(visible, warp, (w,h), flags=cv2.INTER_NEAREST)>0
                        use = np.logical_and(missing, available)
                        warped = cv2.warpAffine(donor_rgb, warp, (w,h), flags=cv2.INTER_NEAREST)
                        image[use], source[use], donor[use] = warped[use], origin, frame_id
                        missing = np.logical_and(missing, np.logical_not(use))
        if auxiliary is not None:
            rgb, valid, j = auxiliary[i]
            use = np.logical_and(edit, valid)
            image[use], source[use], donor[use] = rgb[use], AUXILIARY, j
        outputs.append(image)
        labels.append(source)
        donors.append(donor)
    return np.stack(outputs), np.stack(labels), np.stack(donors)
