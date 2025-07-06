import os
import glob
import cv2
import numpy as np

# Input and output base directories
input_base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids\frames"
output_base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids\pre"
os.makedirs(output_base_dir, exist_ok=True)

# Get all subdirectories under input_base_dir
subdirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

# Process each subdirectory
for subdir in subdirs:
    input_dir = os.path.join(input_base_dir, subdir)

    # Get all jpg files sorted
    frame_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    processed_frames = []

    # Process every frame (removed %10 condition)
    for frame_path in frame_files:
        # Read frame
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Resize to 256x128 (width x height)
        resized = cv2.resize(img, (128, 256))

        # Append to list
        processed_frames.append(resized)

    if processed_frames:
        # Stack into numpy array with shape (num_frames, height, width, channels)
        frames_array = np.stack(processed_frames, axis=0)

        # Save as .npz file named after the directory
        output_path = os.path.join(output_base_dir, f"{subdir}.npz")
        np.savez_compressed(output_path, frames=frames_array)

        print(f"Saved {frames_array.shape[0]} frames to {output_path}")
    else:
        print(f"No frames processed for {subdir}")

print("All directories processed.")
