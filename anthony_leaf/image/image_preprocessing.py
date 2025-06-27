import os
import glob
from PIL import Image
import json

image_dir = "../data/leaf_imgs1/" # We assume the file is run from the top-level directory of BerryPicker
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

images, paths = [], []
for path in image_paths:
    img = Image.open(path)
    images.append(img)
    paths.append(path)
print(f"Loaded {len(images)} images from {image_dir}.") # We expect 16 images

widths, heights = [], []
for img in images:
    widths.append(img.width)
    heights.append(img.height)
avg_width = sum(widths) / len(widths) if widths else 0
avg_height = sum(heights) / len(heights) if heights else 0
print(f"Loaded {len(images)} images from {image_dir}.")
print(f"Average width: {avg_width:.2f}px, Average height: {avg_height:.2f}px")
# Returns: Average width: 814.56px, Average height: 1275.69px

# Make new directory for preprocessed images
output_dir = os.path.join(image_dir, "pre/")
os.makedirs(output_dir, exist_ok=True)

# Resize images to a target size
target_size = (512, 1024)
for i in range(len(images)):
    img, path = images[i], paths[i]
    img_resized = img.resize(target_size)
    filename = os.path.basename(path)
    output_path = os.path.join(output_dir, filename)
    img_resized.save(output_path)

# Create json with labels
labels = [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1]
label_dict = {}
for i in range(len(images)):
    img, path = images[i], paths[i]
    filename = os.path.basename(path)
    label_dict[filename] = labels[i]
json_path = os.path.join(output_dir, "labels.json")
with open(json_path, "w") as f:
    json.dump(label_dict, f, indent=4)
