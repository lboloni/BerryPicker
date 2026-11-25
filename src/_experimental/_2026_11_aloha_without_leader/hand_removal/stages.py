"""File-backed notebook stages. All output paths belong to the current exp/run."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .clip import Clip, pngs, video, write_json
from .fixture import generate
from .reconstruct import masks, reconstruct, register_parts, UNKNOWN, LEARNED, AUXILIARY, TEMPORAL


def prepare(exp):
    out = Path(exp["data_dir"])
    if exp["manifests"]:
        paths = [Path(p).resolve() for p in exp["manifests"]]
    else:
        paths = [generate(out/f"{split}_{obj}", obj, split, seed)
                 for seed,(split,obj) in enumerate([("train","fork"),("validation","fork"),
                                                     ("test","fork"),("test","spoon")],1)]
    names = [json.loads(p.read_text())["episode"] for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Episode IDs must be unique")
    write_json(out/"manifest.json", list(map(str, paths)))
    return paths


def clips(exp):
    path = Path(exp["results_root"])/"_experimental_hand_removal"/"prepare"/"manifest.json"
    return (Clip(p) for p in json.loads(path.read_text()))


def mask_stage(exp):
    rows = []
    for c in clips(exp):
        path = Path(exp["data_dir"])/c.spec["episode"]
        path.mkdir(exist_ok=True)
        edit = masks(c)
        np.save(path/"masks.npy", edit)
        overlay = c.rgb.copy()
        overlay[edit] = (overlay[edit]//2 + [0,0,120]).clip(0,255)
        video(path/"overlay.mp4", overlay, c.fps)
        if "hand_truth" in c.spec:
            truth = c.series("hand_truth")[...,0] > 0
            visible = np.logical_and(c.series("subject_truth")[...,0]>0, np.logical_not(truth))
            rows.append(dict(episode=c.spec["episode"], recall=float(np.logical_and(edit,truth).sum()/max(1,truth.sum())),
                             visible_subject_masked=float(np.logical_and(edit,visible).sum()/max(1,visible.sum()))))
    write_json(Path(exp["data_dir"])/"scores.json", rows)
    return rows


def edits(exp,c):
    return np.load(Path(exp["results_root"])/"_experimental_hand_removal"/"masks"/c.spec["episode"]/"masks.npy", mmap_mode="r")


def candidate(c, edit):
    if "references" in c.spec or "parts" in c.spec:
        return reconstruct(c, edit, "bidirectional")
    return reconstruct(c, edit, "plate")


def train(exp):
    from .learned import fit, inputs
    import torch
    sets = {"train":[], "validation":[]}
    for c in clips(exp):
        if c.spec["split"] in sets:
            edit = edits(exp,c)
            rgb, source, _ = candidate(c, edit)
            x = inputs(c.rgb, edit, rgb, source != UNKNOWN)
            y = torch.from_numpy(c.series("clean")).permute(0,3,1,2).float()/255
            sets[c.spec["split"]].append((x,y))
    if not all(sets.values()):
        raise ValueError("Learning requires paired training and validation clips")
    history = fit(sets["train"], sets["validation"], Path(exp["data_dir"]), exp["epochs"])
    write_json(Path(exp["data_dir"])/"history.json", history)
    return history


def score(c, edit, output, source):
    rows = []
    clean = c.series("clean") if "clean" in c.spec else None
    subject = c.series("subject_truth")[...,0]>0 if "subject_truth" in c.spec else None
    prior = None
    for i, mask in enumerate(edit):
        row = dict(frame=c.ids[i], time=float(c.times[i]), edited=int(mask.sum()),
                   unresolved=int((source[i]==UNKNOWN).sum()))
        if clean is not None and mask.any():
            diff = np.abs(output[i].astype(float)-clean[i])
            row["edited_mae"] = float(diff[mask].mean())
            if subject is not None:
                pred = np.abs(output[i].astype(float)-c.plate).max(2)>20
                active = np.logical_and(np.logical_or(pred, subject[i]), mask)
                if active.any():
                    row["hidden_subject_iou"] = float(np.logical_and(np.logical_and(pred,subject[i]),mask).sum()/active.sum())
                a, b = estimate_pose(pred), estimate_pose(subject[i])
                if a is not None and b is not None:
                    if prior is not None:
                        row["displacement_error_px"] = float(np.linalg.norm((a[:2]-b[:2])-prior))
                    prior = a[:2]-b[:2]
                    row["predicted_centroid"] = a[:2].tolist()
                    row["expected_centroid"] = b[:2].tolist()
                    row["centroid_error_px"] = float(np.linalg.norm(a[:2]-b[:2]))
                    row["orientation_error_deg"] = float(abs((a[2]-b[2]+90)%180-90))
                else:
                    prior = None
        if "hsv" in c.spec and mask.any():
            from .reconstruct import hsv_masks
            residue = hsv_masks(output[i:i+1], c.spec["hsv"])[0]
            row["residual_glove_fraction"] = float(np.logical_and(residue,mask).sum()/mask.sum())
        if "landmark_truth" in c.spec:
            expected = np.asarray(c.spec["landmark_truth"][i])
            # Local error to the nearest reconstructed foreground for tip/end landmarks.
            if subject is not None and clean is not None and mask.any() and np.any(pred):
                distance = cv2.distanceTransform(np.logical_not(pred).astype(np.uint8), cv2.DIST_L2, 5)
                x,y = np.rint(expected).astype(int).T
                row["landmark_to_foreground_px"] = float(distance[y,x].mean())
        rows.append(row)
    return rows


def estimate_pose(mask):
    y,x = np.nonzero(mask)
    if len(x)<3:
        return None
    coords = np.stack((x,y),axis=1)
    _, axes = np.linalg.eigh(np.cov(coords.T))
    angle = np.degrees(np.arctan2(*axes[::-1,-1]))
    return np.r_[coords.mean(0),angle]


def verify(exp, multi=False):
    for c in clips(exp):
        if c.spec["split"] != "test" or (multi and "auxiliary" not in c.spec):
            continue
        edit = edits(exp,c)
        methods = ["multiview"] if multi else exp["methods"]
        layers = register_parts(c, edit) if any(m != "plate" for m in methods) else []
        episode_dir = Path(exp["data_dir"])/c.spec["episode"]
        episode_dir.mkdir(parents=True, exist_ok=True)
        fits = [[None if pose is None else dict(reference=pose[1], residual_px=pose[0],
                                                transform=pose[2].tolist()) for pose in layer]
                for layer in layers]
        write_json(episode_dir/"registration.json", fits)
        for method in methods:
            path = Path(exp["data_dir"])/c.spec["episode"]/method
            path.mkdir(parents=True,exist_ok=True)
            if method=="learned":
                import torch
                from .learned import RemovalNet, inputs
                rgb,source,donor = candidate(c,edit)
                model = RemovalNet().eval()
                model.load_state_dict(torch.load(Path(exp["results_root"])/"_experimental_hand_removal"/"train"/"weights.pt",
                                                 weights_only=True))
                x = inputs(c.rgb, edit, rgb, source!=UNKNOWN)
                with torch.no_grad():
                    estimated = torch.cat([model(batch) for batch in x.split(4)]).permute(0,2,3,1)
                rgb[edit] = (estimated.clamp(0,1).numpy()*255).round().astype(np.uint8)[edit]
                learned_pixels = np.logical_and(edit, source != UNKNOWN)
                source[learned_pixels] = LEARNED
            else:
                rgb,source,donor = reconstruct(c,edit,method,layers=layers)
            outside = np.logical_not(edit)
            changed = np.any(np.not_equal(rgb, c.rgb), axis=3)
            if np.count_nonzero(np.logical_and(changed, outside)):
                raise AssertionError("Compositor changed a pixel outside the edit mask")
            pngs(path/"frames",rgb)
            np.savez_compressed(path/"provenance.npz", source=source, donor=donor, edit=edit,
                                unresolved=source==UNKNOWN)
            video(path/"cleaned.mp4", rgb, c.fps)
            panels = [c.rgb,rgb,cv2.applyColorMap(source.reshape(-1,source.shape[-1])*35,
                      cv2.COLORMAP_TURBO).reshape(c.rgb.shape)]
            if "clean" in c.spec:
                panels.append(c.series("clean"))
            video(path/"review.mp4",np.concatenate(panels,axis=2),c.fps)
            scores = score(c,edit,rgb,source)
            if method == "multiview":
                aux = c.spec["auxiliary"]
                times = np.asarray(aux["timestamps"]) if "timestamps" in aux else None
                for t, row in enumerate(scores):
                    ids = donor[t][source[t] == AUXILIARY]
                    row["auxiliary_pixels"] = len(ids)
                    if len(ids):
                        frame_ids = np.unique(ids)
                        stamps = frame_ids/aux["fps"] if times is None else times[frame_ids]
                        row["auxiliary_frames"] = frame_ids.tolist()
                        row["max_donor_time_difference_s"] = float(
                            np.max(np.abs(stamps + aux.get("offset_s",0) - c.times[t])))
            write_json(path/"frames.json",scores)
            write_json(path/"sources.json", {"episode": c.spec["episode"], "view": c.spec.get("view","target"),
                                            "frame_ids": c.ids, "timestamps": c.times.tolist(),
                                            "reference_images": [r["image"] for r in c.spec.get("references",[])],
                                            "auxiliary": c.spec.get("auxiliary",{}).get("rgb")})


def compare(exp):
    rows = []
    out = Path(exp["data_dir"])
    for stage in exp["compare_stages"]:
        root = Path(exp["results_root"])/"_experimental_hand_removal"/stage
        for c in clips(exp):
            if c.spec["split"] != "test":
                continue
            for p in sorted((root/c.spec["episode"]).glob("*/frames.json")):
                scores = json.loads(p.read_text())
                row = dict(episode=c.spec["episode"],method=p.parent.name,frames=len(scores),
                           coverage=1-sum(r["unresolved"] for r in scores)/max(1,sum(r["edited"] for r in scores)),
                           review=str(p.parent/"review.mp4"))
                for metric in ("edited_mae","hidden_subject_iou","centroid_error_px","orientation_error_deg","displacement_error_px",
                               "residual_glove_fraction","landmark_to_foreground_px"):
                    values = [r[metric] for r in scores if metric in r]
                    if values:
                        row[metric] = float(np.mean(values))
                rows.append(row)
    if not rows:
        raise ValueError("No reconstructed test episodes to compare")
    with (out/"scores.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Hand removal comparison","","| Episode | Method | Edit MAE (0–255) | Coverage | Review |",
             "|---|---|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['episode']} | {r['method']} | {r.get('edited_mae','—')} | "
                     f"{r['coverage']:.1%} | [video]({r['review']}) |")
    lines += ["", "## Contributions from future frames", ""]
    for c in clips(exp):
        if c.spec["split"] != "test" or "clean" not in c.spec:
            continue
        d = Path(exp["results_root"])/"_experimental_hand_removal"/"single"/c.spec["episode"]
        if not {"past","bidirectional"}.issubset(exp["methods"]):
            continue
        from .clip import frames
        with np.load(d/"bidirectional"/"provenance.npz") as bi, np.load(d/"past"/"provenance.npz") as past:
            future = np.logical_and(bi["source"] == TEMPORAL, bi["donor"] > np.arange(len(c.rgb))[:,None,None])
            revealed = np.logical_and(future, past["source"] != TEMPORAL)
        truth = c.series("clean").astype(float)
        for title, pixels in [("first seen later",revealed), ("all future donors",future)]:
            if pixels.any():
                errors = [np.abs(frames(d/m/"frames").astype(float)-truth)[pixels].mean()
                          for m in ("past","bidirectional")]
                lines.append(f"- {c.spec['episode']}, {title}: {pixels.sum()} pixels, "
                             f"past-only MAE {errors[0]:.2f}, bidirectional MAE {errors[1]:.2f}.")
    lines += ["", "## Frames to inspect", ""]
    for row in rows:
        directory = Path(row["review"]).parent
        scores = json.loads((directory/"frames.json").read_text())
        worst = max(scores, key=lambda r: (r["unresolved"],r.get("edited_mae",0)))
        lines.append(f"- {row['episode']} / {row['method']}: frame {worst['frame']}, "
                     f"{worst['unresolved']} unresolved pixels.")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    return rows
