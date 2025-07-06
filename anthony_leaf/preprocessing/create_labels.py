import os
import numpy as np
import pandas as pd

# Paths
npz_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids\pre"
csv_dir = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\videos2\labels"
output_dir = r"C:\Users\abmst\Documents\projects\BerryPicker\anthony_leaf\videos2\labels2"
os.makedirs(output_dir, exist_ok=True)

# Process each .npz file
for npz_name in os.listdir(npz_dir):
    if not npz_name.endswith(".npz"):
        continue

    base_name = os.path.splitext(npz_name)[0]
    npz_path = os.path.join(npz_dir, npz_name)
    csv_path = os.path.join(csv_dir, f"{base_name}.csv")

    # Skip if paired CSV does not exist
    if not os.path.exists(csv_path):
        print(f"CSV for {base_name} not found, skipping.")
        continue

    # Load video frames (assuming single array in .npz)
    npz_data = np.load(npz_path)
    frames = None
    for key in npz_data:
        frames = npz_data[key]
        break
    if frames is None:
        print(f"No data in {npz_name}, skipping.")
        continue

    num_frames = frames.shape[0]

    # Initialize label array with -1
    labels = -1 * np.ones(num_frames, dtype=int)

    # Load csv labels
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        start = int(row['start'])
        end = int(row['end'])
        label = int(row['label'])

        # Assign label to frames in range [start, end] inclusive
        labels[start:end+1] = label

    # Save label array as .npy
    output_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(output_path, labels)
    print(f"Saved labels for {base_name} to {output_path}")
