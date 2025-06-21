import os
import shutil
import re
import glob

def organize_files(root_dir):
    # Get the base name of the root directory
    root_dir_name = os.path.basename(root_dir)
    parent_dir = os.path.dirname(root_dir)

    # Create the new main directories
    singleview2_main_dir = os.path.join(parent_dir, f"{root_dir_name}_singleview2")
    singleview3_main_dir = os.path.join(parent_dir, f"{root_dir_name}_singleview3")

    os.makedirs(singleview2_main_dir, exist_ok=True)
    os.makedirs(singleview3_main_dir, exist_ok=True)

    # Find all subdirectories in the root directory
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in subdirs:
        source_dir_path = os.path.join(root_dir, subdir)

        # Create corresponding subdirectories in the new main directories
        singleview2_subdir = os.path.join(singleview2_main_dir, subdir)
        singleview3_subdir = os.path.join(singleview3_main_dir, subdir)

        os.makedirs(singleview2_subdir, exist_ok=True)
        os.makedirs(singleview3_subdir, exist_ok=True)

        # Find all JSON files
        json_files = glob.glob(os.path.join(source_dir_path, "*.json"))

        for json_file in json_files:
            json_basename = os.path.basename(json_file)

            # Extract the number part (e.g., "00001" from "00001.json")
            match = re.search(r'(\d+)\.json', json_basename)
            if match:
                file_number = match.group(1)

                # Find corresponding dev2 and dev3 images
                dev2_pattern = f"{file_number}_dev2.jpg"
                dev3_pattern = f"{file_number}_dev3.jpg"

                dev2_path = os.path.join(source_dir_path, dev2_pattern)
                dev3_path = os.path.join(source_dir_path, dev3_pattern)

                # Copy files to appropriate directories
                if os.path.exists(dev2_path):
                    shutil.copy2(json_file, singleview2_subdir)
                    shutil.copy2(dev2_path, singleview2_subdir)
                    print(f"Copied {json_basename} and {dev2_pattern} to {singleview2_subdir}")

                if os.path.exists(dev3_path):
                    shutil.copy2(json_file, singleview3_subdir)
                    shutil.copy2(dev3_path, singleview3_subdir)
                    print(f"Copied {json_basename} and {dev3_pattern} to {singleview3_subdir}")

if __name__ == "__main__":
    # Replace this with the path to your proprio_regressor_training directory
    root_directory = "/home/ssheikholeslami/SaharaBerryPickerData/demonstrations/demos/proprio_sp_validation"
    organize_files(root_directory)