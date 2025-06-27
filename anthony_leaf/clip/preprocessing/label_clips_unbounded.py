import os
import numpy as np
from PIL import Image

# Base directory containing clip folders
base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips"

# Iterate through each clip folder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if 'clip' not in folder_name:
        continue
    if os.path.isdir(folder_path):
        frame_arrays = []
        
        # Get all .jpg filenames sorted for consistency
        file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

        # Load each image and convert to numpy array
        for file_name in file_list:
            img_path = os.path.join(folder_path, file_name)
            img = Image.open(img_path).convert('RGB')
            frame_arrays.append(np.array(img))
        
        print(len(frame_arrays)) # max 58

        # Stack frames into a single numpy array: (num_frames, H, W, C)
        frames_np = np.stack(frame_arrays, axis=0)

        # Save as frames.npy in the clip folder
        npy_path = os.path.join(folder_path, "frames.npy")
        np.save(npy_path, frames_np)
        
        print(f"Saved {frames_np.shape} to {npy_path}")

import os
import json

# Base directory containing clip folders
base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips"
# Initialize dictionary
npy_labels_dict = {}

# Iterate through each clip folder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if 'clip' not in folder_name:
        continue
    if os.path.isdir(folder_path):
        # Extract label from folder name (e.g. clip_1_label_1 -> 1)
        label = int(folder_name.split('_')[-1])

        # Define path to frames.npy
        npy_path = os.path.join(folder_path, "frames.npy")

        # Save entry
        npy_labels_dict[folder_name] = {
            "npy_path": npy_path,
            "label": label
        }

# Save as npy_labels.json in base directory
output_path = os.path.join(base_dir, "labels.json")
with open(output_path, 'w') as f:
    json.dump(npy_labels_dict, f, indent=2)

print(f"Saved npy_labels.json with {len(npy_labels_dict)} entries to {output_path}")
