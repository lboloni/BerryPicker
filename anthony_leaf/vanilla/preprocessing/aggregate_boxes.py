import os
import json

# ===== Directory to search =====
base_dir = "/data/breast-cancer/leaf_vids/frames"

# ===== Initialize list to hold all annotation entries =====
annotations = []

# ===== Traverse all subdirectories =====
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(".json"):
            json_path = os.path.join(root, file)

            # Load JSON file
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Process each shape annotation
            for shape in data.get("shapes", []):
                entry = {
                    "path": json_path,
                    "label": shape.get("label"),
                    "points": shape.get("points")
                }
                annotations.append(entry)

# ===== Save the results as JSON =====
output_path = "anthony_leaf/boxes.json"

with open(output_path, 'w') as f:
    json.dump(annotations, f, indent=2)

print(f"Saved {len(annotations)} annotations to {output_path}")