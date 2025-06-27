import os
import pandas as pd
import shutil

# Paths
frames_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\vid1"
csv_path = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\video1.csv"
output_base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids1\clips"

# Read CSV
df = pd.read_csv(csv_path)

# Ensure output directory exists
os.makedirs(output_base_dir, exist_ok=True)

# Get all frame files
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

# Process each clip interval
for idx, row in df.iterrows():
    start = row['start']
    end = row['end']
    label = row['label']
    
    # Parse times to floats for comparison
    start_time = float(start.replace('_', '.').replace('s',''))
    end_time = float(end.replace('_', '.').replace('s',''))
    
    # Create subdirectory for this clip
    clip_dir_name = f"clip_{idx+1}_label_{label}"
    clip_dir = os.path.join(output_base_dir, clip_dir_name)
    os.makedirs(clip_dir, exist_ok=True)
    
    # Copy frames within interval
    for file in frame_files:
        # Extract time from filename
        parts = file.replace('frame_','').replace('.jpg','').split('_')
        time_str = parts[0] + '.' + parts[1].replace('s','')
        frame_time = float(time_str)
        
        if start_time <= frame_time <= end_time:
            src_path = os.path.join(frames_dir, file)
            dst_path = os.path.join(clip_dir, file)
            shutil.copy2(src_path, dst_path)
    
    print(f"Saved clip {idx+1} with label {label} to {clip_dir}")

print("All clips processed.")
