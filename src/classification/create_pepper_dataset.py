import os
import random
import json

# Set paths and labels
root_dir = '/data/breast-cancer/PlantVillage/'
target_labels = {
    "0": "Pepper__bell___Bacterial_spot",
    "1": "Pepper__bell___healthy"
}

output_json = 'src/classification/pepper_balanced_split.json'
split_ratio = 0.8  # 80% training, 20% testing

# Collect images per class
data_by_label = {0: [], 1: []}

for label_str, dirname in target_labels.items():
    label = int(label_str)
    dirpath = os.path.join(root_dir, dirname)
    for fname in os.listdir(dirpath):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            full_path = os.path.join(dirpath, fname)
            data_by_label[label].append(full_path)

# Shuffle and split each class
train_set = []
test_set = []

for label, items in data_by_label.items():
    random.shuffle(items)
    split_point = int(len(items) * split_ratio)
    train_set += [{"path": path, "label": label} for path in items[:split_point]]
    test_set += [{"path": path, "label": label} for path in items[split_point:]]

# Shuffle globally to avoid ordering bias
random.shuffle(train_set)
random.shuffle(test_set)

# Save as JSON
with open(output_json, 'w') as f:
    json.dump({"train": train_set, "test": test_set}, f, indent=2)

print(f"Saved {len(train_set)} training and {len(test_set)} testing samples to {output_json}")
