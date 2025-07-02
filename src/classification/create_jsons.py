import os
import csv
import json

# Root directory
root_dir = '/data/breast-cancer/PlantVillage/'

# Output files
csv_path = 'src/classification/image_index.csv'
json_path = 'src/classification/label_map.json'

# Initialize
image_entries = []
label_map = {}
label_id = 0

# Walk through label directories
for dirname in sorted(os.listdir(root_dir)):
    dirpath = os.path.join(root_dir, dirname)
    if not os.path.isdir(dirpath):
        continue

    label_map[label_id] = dirname  # Integer label → directory name

    for fname in sorted(os.listdir(dirpath)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            image_entries.append([os.path.join(dirpath, fname), label_id])

    label_id += 1

# Write CSV: path,label
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'label'])
    writer.writerows(image_entries)

# Write JSON: {int label: directory name}
with open(json_path, 'w') as f:
    json.dump(label_map, f, indent=2)

print(f"Saved {len(image_entries)} images to {csv_path}")
print(f"Saved label map to {json_path}")
