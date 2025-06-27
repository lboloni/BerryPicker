import cv2
import os

# Paths
video_path = "../data/leaf_imgs1/leaf_vid.mp4"
output_dir = "../projects/data/leaf_vids1"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)

# Check if loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_idx = 0
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Calculate timestamp in seconds (with milliseconds)
    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Format timestamp to 3 decimal places
    time_str = f"{time_sec:.3f}s"

    # Replace '.' with '_' to make filenames valid
    time_str = time_str.replace('.', '_')

    # Build output filename
    output_path = os.path.join(output_dir, f"frame_{time_str}.jpg")

    # Save frame as image
    cv2.imwrite(output_path, frame)
    frame_idx += 1

print(f"Saved {frame_idx} frames to {output_dir}")

# Release the video capture
cap.release()
