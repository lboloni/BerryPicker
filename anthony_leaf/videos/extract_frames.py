import cv2
import os
import glob

# Input and output directories
input_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids2\vids"
output_base_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids2\frames"

# Get all video files in the input directory
video_files = glob.glob(os.path.join(input_dir, "*.*"))  # matches all files

# Process each video
for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as JPEG with quality setting (adjust 85 as needed)
        output_path = os.path.join(output_dir, f"frame_{frame_num:04d}.jpg")
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame_num += 1

    cap.release()
    print(f"Finished extracting {frame_num} frames from {video_name}")

print("All videos processed.")
