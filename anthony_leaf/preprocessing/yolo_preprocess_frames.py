import os
import csv
import random
import shutil

# ===== Paths =====
frames_base = "/data/breast-cancer/leaf_vids/leaf_vids/frames"
labels_base = "/home/anthony/BerryPicker/anthony_leaf/labels"
output_base = "/home/anthony/BerryPicker/leaf/datasets"

# ===== Create dataset directories =====
for split in ["train", "test"]:
    for label in ["0", "1"]:
        dir_path = os.path.join(output_base, split, label)
        os.makedirs(dir_path, exist_ok=True)

# ===== Get all video directories =====
video_dirs = sorted([d for d in os.listdir(frames_base) if os.path.isdir(os.path.join(frames_base, d))])

# ===== Split into train and test (8 train + 2 test) =====
random.shuffle(video_dirs)
train_videos = video_dirs[:8]
test_videos = video_dirs[8:10]

# ===== Initialize global image counter =====
global_counter = 0

# ===== Function to process split =====
def process_split(video_list, split_name):
    global global_counter

    for vid in video_list:
        frames_dir = os.path.join(frames_base, vid)
        label_csv = os.path.join(labels_base, f"{vid}.csv")

        # Load label ranges
        label_ranges = []
        with open(label_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = int(row["start"])
                end = int(row["end"])
                label = row["label"]
                label_ranges.append((start, end, label))

        # Process each label range
        for start, end, label in label_ranges:
            for frame_num in range(start, end + 1):
                frame_filename = f"frame_{frame_num:04d}.jpg"
                src_path = os.path.join(frames_dir, frame_filename)

                if not os.path.exists(src_path):
                    print(f"Frame not found: {src_path}")
                    continue

                # Destination path
                dest_dir = os.path.join(output_base, split_name, label)
                dest_path = os.path.join(dest_dir, f"{global_counter}.jpg")

                # Copy file
                shutil.copy(src_path, dest_path)
                global_counter += 1

# ===== Process train and test splits =====
process_split(train_videos, "train")
process_split(test_videos, "test")

print("✅ Dataset preparation complete.")
