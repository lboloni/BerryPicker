import os
import json
import pandas as pd

# Paths
frames_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\vid1"
csv_path = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\video\preprocessing\video1.csv"
output_path = os.path.join(frames_dir, "labels.json")

# Load CSV
df = pd.read_csv(csv_path)

# Convert start and end to comparable format (e.g., "4_800s" -> 4800) for numerical comparison
def frame_str_to_num(s):
    return int(s.replace("_", "").replace("s", ""))

# Parse all labeled ranges into list of tuples: (start_num, end_num, label)
label_ranges = []
for _, row in df.iterrows():
    start_num = frame_str_to_num(row['start'])
    end_num = frame_str_to_num(row['end'])
    label = int(row['label'])
    label_ranges.append((start_num, end_num, label))

# Process each frame in directory
labels_dict = {}
for fname in os.listdir(frames_dir):
    if fname.endswith(".jpg"):
        # Extract numeric part from filename: "frame_3_600s.jpg" -> 3600
        base = os.path.splitext(fname)[0]
        parts = base.split("_")
        frame_num = int(parts[1] + parts[2].replace("s", ""))
        
        # Check if this frame falls in any labeled range
        assigned_label = 2  # default
        for start, end, label in label_ranges:
            if start <= frame_num <= end:
                assigned_label = label
                break
        
        # Add full path as key
        full_path = os.path.join(frames_dir, fname)
        labels_dict[full_path] = assigned_label

# Save as json
with open(output_path, 'w') as f:
    json.dump(labels_dict, f, indent=4)

print(f"Labels saved to {output_path}")
