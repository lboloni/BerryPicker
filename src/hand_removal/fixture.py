"""Generate independent, labeled moving-object episodes for notebook verification."""
import cv2
import numpy as np

from .clip import pngs, write_json


def generate(root, subject, split, seed):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    h, w = 96, 144
    plate = rng.integers(38, 60, (h, w, 3), dtype=np.uint8)
    ref = np.zeros((h, w, 4), np.uint8)
    color = (180, 195, 205, 255)
    cv2.line(ref, (30, 48), (76, 48), color, 5)
    if subject == "fork":
        cv2.rectangle(ref, (73, 40), (82, 56), color, -1)
        for y in (40, 45, 51, 56):
            cv2.line(ref, (81, y), (105, y), color, 2)
    else:
        cv2.circle(ref, (86, 48), 13, color, -1)
    cv2.imwrite(str(root / "reference.png"), ref)
    cv2.imwrite(str(root / "plate.png"), plate)
    # The moving instance carries a marking absent from the reference photograph.
    marked = ref.copy()
    cv2.circle(marked, (42, 48), 2, (38, 65, 190, 255), -1)
    anchors = np.float32([[30, 48], [75, 48], [92, 45], [85, 53]])
    series = {k: [] for k in ("rgb", "clean", "hand", "subject", "aux")}
    tracks, landmarks = [], []
    for i in range(24):
        phase = i / 23
        m = cv2.getRotationMatrix2D((70, 48), 22*np.sin(phase*4 + seed), 1)
        m[:, 2] += [12*np.sin(phase*5), 9*np.cos(phase*4)]
        moved = cv2.warpAffine(marked, m, (w, h), flags=cv2.INTER_NEAREST)
        clean = np.where(moved[:, :, 3:] > 0, moved[:, :, :3], plate)
        points = cv2.transform(anchors[None], m)[0]
        hand = np.zeros((h, w), np.uint8)
        grip = tuple(np.rint(points[0] + (9 if i < 15 else 29, 0)).astype(int))
        cv2.ellipse(hand, grip, (20, 13), -18, 0, 360, 255, -1)
        cv2.line(hand, (grip[0]-4, grip[1]+8), (grip[0]-16, h), 255, 15)
        if i == 10:
            cv2.rectangle(hand, (5, 5), (w-5, h-5), 255, -1)
        if i in (7, 8):  # The fork passes in front of the moving fingers.
            hand[moved[...,3]>0] = 0
        rgb = clean.copy()
        rgb[hand > 0] = (20, 185, 32)
        # Rectified second view of a planar scene, with a known four-pixel disparity.
        aux = np.roll(clean, -4, axis=1)
        aux[:, -4:] = 0
        for key, image in zip(series, (rgb, clean, hand, moved[:, :, 3], aux)):
            series[key].append(image)
        visible = [None if hand[round(y), round(x)] else [float(x), float(y)] for x, y in points]
        tracks.append(visible)
        tips = [[30,48],[105,40],[105,45],[105,51],[105,56]] if subject=="fork" else [[30,48],[99,48]]
        landmarks.append(cv2.transform(np.float32([tips]), m)[0].tolist())
    for key, images in series.items():
        pngs(root/key, images)
    spec = dict(episode=root.name, split=split, rgb="rgb", plate="plate.png", fps=12,
                landmark_truth=landmarks, clean="clean", hand_truth="hand", subject_truth="subject",
                hsv=[[38, 110, 80], [85, 255, 255]], references=[
                    dict(image="reference.png", points=anchors.tolist())], tracks=tracks,
                auxiliary=dict(rgb="aux", fps=12, hsv=[[38,110,80],[85,255,255]],
                               K=[[100,0,72],[0,100,48],[0,0,1]], target_K=[[100,0,72],[0,100,48],[0,0,1]],
                               target_from_source=[[1,0,0,.04],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                               plane_depth_m=1.0, max_skew_s=.02))
    write_json(root/"clip.json", spec)
    return root/"clip.json"
