"""Bridge BerryPicker demonstration I/O and ordinary clip manifests."""
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from demonstration.demonstration import Demonstration, get_simple_transform
from .clip import pngs, still, write_json


def joints_at_exposure(demo, camera, capture_delay_s, max_gap_s):
    """Interpolate six unwrapped angles and the finger reading on the host clock."""
    samples = [state for row in demo.annotations for state in row["teaching_arm"]["state_samples"]]
    stamps = np.asarray([state["read_monotonic"] for state in samples])
    values = np.asarray([state["joints"] for state in samples])
    values[:,:6] = np.unwrap(values[:,:6], axis=0)
    output = []
    for row in demo.annotations:
        exposure = row["cameras"]["read_monotonic"][camera] - capture_delay_s
        j = np.searchsorted(stamps, exposure)
        if j < len(stamps) and stamps[j] == exposure:
            output.append(values[j].tolist())
        elif j == 0 or j == len(stamps) or stamps[j]-stamps[j-1] > max_gap_s:
            output.append(None)
        else:
            fraction = (exposure-stamps[j-1])/(stamps[j]-stamps[j-1])
            output.append((values[j-1]+fraction*(values[j]-values[j-1])).tolist())
    return output


def import_view(demo, camera, destination, manifest):
    """Extract one deployment view. The caller supplies reference and mask settings."""
    destination = Path(destination)
    images = []
    for i in range(demo.metadata["maxsteps"]):
        tensor, _ = demo.get_image(i, camera=camera, transform=get_simple_transform())
        rgb = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
        images.append(cv2.cvtColor(np.rint(255*rgb).astype(np.uint8), cv2.COLOR_RGB2BGR))
    pngs(destination/"rgb", images)
    values = dict(manifest, rgb="rgb", episode=demo.demo, view=camera, frame_ids=list(range(len(images))))
    if "timestamps" not in values and "cameras" in demo.annotations[0]:
        values["timestamps"] = [a["cameras"]["read_monotonic"][camera] for a in demo.annotations]
    write_json(destination/"clip.json", values)
    return destination/"clip.json"


def export_view(demo, camera, method_dir, output_exp, prefix, rejected=(), aligned_joints=None):
    """Publish accepted contiguous intervals as single-camera demonstrations."""
    method_dir = Path(method_dir)
    rows = json.loads((method_dir/"frames.json").read_text())
    rejected = set(rejected)
    if aligned_joints is not None:
        rejected.update(i for i,q in enumerate(aligned_joints) if q is None)
    accepted = [i for i,r in enumerate(rows) if r["unresolved"]==0 and r["frame"] not in rejected]
    destination = Path(output_exp.data_dir())
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    for _, group in itertools.groupby(enumerate(accepted), key=lambda e: rows[e[1]]["frame"]-e[0]):
        ids = [i for _,i in group]
        name = f"{prefix}_{rows[ids[0]]['frame']:06}"
        final = destination/name
        if final.exists():
            raise FileExistsError(final)
        with TemporaryDirectory(dir=destination) as tmp:
            child = Demonstration(output_exp, Path(tmp).name)
            child.metadata = dict(demo.metadata, stored_as_video=False, stored_as_images=True,
                                  cameras=[camera], maxsteps=len(ids))
            for new,i in enumerate(ids):
                frame = rows[i]["frame"]
                action = dict(demo.actions[frame])
                if aligned_joints is not None:
                    action["taught-joints-at-image"] = aligned_joints[frame]
                child.actions.append(action)
                child.annotations.append(dict(demo.annotations[frame], source_episode=demo.demo,
                                              source_frame=frame, timestamp=rows[i]["time"]))
                image = still(method_dir/"frames"/f"{i:06}.png")
                if not cv2.imwrite(str(child.get_image_path(new,camera)), image):
                    raise OSError("Could not export demonstration image")
            child.save_metadata()
            loaded = Demonstration(output_exp, child.demo)
            loaded.get_image(len(ids)-1, camera=camera, transform=get_simple_transform())
            # Move the completed directory out of the temporary staging parent.
            child.demo_dir.rename(final)
        names.append(name)
    return names
