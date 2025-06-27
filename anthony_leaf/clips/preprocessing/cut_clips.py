import os

# Path to clips directory
clips_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips"

# List all clip subdirectories
clip_subdirs = [os.path.join(clips_dir, d) for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d))]

# Initialize smallest frame count
min_count = None
min_clip = None

# Iterate through each clip subdirectory
for clip_dir in clip_subdirs:
    # Count .jpg files
    frame_files = [f for f in os.listdir(clip_dir) if f.endswith('.jpg')]
    count = len(frame_files)
    
    print(f"{clip_dir} has {count} frames.")
    
    if (min_count is None) or (count < min_count):
        min_count = count
        min_clip = clip_dir

# Print smallest result
print(f"\nSmallest number of frames is {min_count}, in clip directory: {min_clip}")

import shutil

# Paths
clips_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips"
output_base_dir = os.path.join(clips_dir, "pre")

# Ensure output base directory exists
os.makedirs(output_base_dir, exist_ok=True)

# List all clip subdirectories (excluding 'pre' itself)
clip_subdirs = [d for d in os.listdir(clips_dir) if os.path.isdir(os.path.join(clips_dir, d)) and d != "pre"]

for clip_subdir in clip_subdirs:
    clip_path = os.path.join(clips_dir, clip_subdir)
    frame_files = sorted([f for f in os.listdir(clip_path) if f.endswith('.jpg')])
    
    num_frames = len(frame_files)
    
    # Determine middle indices for up to 7 frames
    if num_frames <= 7:
        selected_frames = frame_files
    else:
        mid = num_frames // 2
        half_window = 3
        start_idx = max(mid - half_window, 0)
        end_idx = start_idx + 7
        if end_idx > num_frames:
            end_idx = num_frames
            start_idx = end_idx - 7
        selected_frames = frame_files[start_idx:end_idx]
    
    # Create output subdirectory
    out_subdir = os.path.join(output_base_dir, clip_subdir)
    os.makedirs(out_subdir, exist_ok=True)
    
    # Copy selected frames
    for file in selected_frames:
        src = os.path.join(clip_path, file)
        dst = os.path.join(out_subdir, file)
        shutil.copy2(src, dst)
    
    print(f"Saved {len(selected_frames)} frames to {out_subdir}")

print("All clips processed with middle 7 frames saved.")