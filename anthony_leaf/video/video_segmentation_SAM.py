import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ===== Paths =====
input_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids2\frames\IMG_8653"
output_dir = r"C:\Users\abmst\Documents\projects\data\leaf_vids2\frames\IMG_8653_seg"
os.makedirs(output_dir, exist_ok=True)

# ===== Load SAM model =====
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_checkpoint = "anthony_leaf/video/sam_vit_h_4b8939.pth"  # Adjust path if needed
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# ===== Process each frame =====
frame_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')])
for idx, frame_name in enumerate(frame_files):
    frame_path = os.path.join(input_dir, frame_name)
    output_path = os.path.join(output_dir, frame_name)

    # Load frame
    frame = cv2.imread(frame_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(rgb_frame)

    # Overlay masks onto frame
    overlay = frame.copy()
    for mask in masks:
        seg = mask['segmentation']
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        overlay[seg] = overlay[seg] * 0.3 + color * 0.7

    # Save processed frame
    cv2.imwrite(output_path, overlay.astype(np.uint8))
    print(f"Processed {idx+1}/{len(frame_files)}: {frame_name}", end="\r")

print("\nAll frames segmented and saved successfully.")
