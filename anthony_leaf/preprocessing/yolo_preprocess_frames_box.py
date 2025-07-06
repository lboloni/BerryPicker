import os
import json
import shutil
import random
from collections import defaultdict
from PIL import Image

# ===== CONFIG =====
boxes_json_path = "/home/anthony/BerryPicker/anthony_leaf/boxes.json"
output_base_dir = "./datasets/leaf_box"
images_dir = os.path.join(output_base_dir, "images")
labels_dir = os.path.join(output_base_dir, "labels")

# ===== Create directories =====
for split in ["train", "val"]:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# ===== Load annotations =====
with open(boxes_json_path, "r") as f:
    annotations = json.load(f)

# ===== Group by labels for balanced split =====
label_to_ann = defaultdict(list)
for ann in annotations:
    label_to_ann[ann['label']].append(ann)

# ===== Perform 80-20 split per label =====
train_anns, val_anns = [], []
for label, anns in label_to_ann.items():
    random.shuffle(anns)
    split_idx = int(0.8 * len(anns))
    train_anns.extend(anns[:split_idx])
    val_anns.extend(anns[split_idx:])

print(f"Train: {len(train_anns)}, Val: {len(val_anns)}")

# ===== Process and save =====
def process_split(anns, split):
    for ann in anns:
        # ===== Paths =====
        ann_path = ann['path']
        img_path = ann_path.replace(".json", ".jpg")
        img_filename = os.path.basename(img_path)
        img_out_path = os.path.join(images_dir, split, img_filename)

        # ===== Copy image =====
        if os.path.exists(img_path):
            shutil.copyfile(img_path, img_out_path)
        else:
            print(f"[WARNING] Image not found: {img_path}")
            continue

        # ===== Get image size =====
        with Image.open(img_path) as img:
            width, height = img.size

        # ===== Convert to YOLO txt format =====
        x1, y1 = ann['points'][0]
        x2, y2 = ann['points'][1]
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        bbox_width = (xmax - xmin) / width
        bbox_height = (ymax - ymin) / height
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height

        label_int = int(ann['label'])

        txt_line = f"{label_int} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

        # ===== Save label .txt =====
        label_filename = os.path.basename(ann_path).replace(".json", ".txt")
        label_out_path = os.path.join(labels_dir, split, label_filename)
        with open(label_out_path, "w") as f:
            f.write(txt_line + "\n")

# ===== Process splits =====
process_split(train_anns, "train")
process_split(val_anns, "val")

print("✅ Done.")
