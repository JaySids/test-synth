import json
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

# --- Configuration ---
# The COCO annotation file you want to process.
COCO_FILE = Path("coco_annotations.json")
# The directory where your original images are located.
SOURCE_IMAGE_DIR = Path("images")

# The base output directory will be created in your home directory.
# e.g., /home/user/goodOrBad/ or C:\Users\user\goodOrBad\
BASE_OUTPUT_DIR = Path.home() / "goodOrBad"
# The subdirectory where all cropped images will be saved.
OUTPUT_CROPS_DIR = BASE_OUTPUT_DIR / "datasets" / "black_crops"


def crop_all_bounding_boxes():
    """
    Crops all bounding boxes from a COCO file and saves them into a 
    single 'black_crops' directory in the user's home directory.
    """
    # 1. Setup: Clean and create the output directory
    if OUTPUT_CROPS_DIR.exists():
        shutil.rmtree(OUTPUT_CROPS_DIR)
        print(f"Removed existing directory: {OUTPUT_CROPS_DIR}")
    
    print(f"Creating directory: {OUTPUT_CROPS_DIR}")
    OUTPUT_CROPS_DIR.mkdir(parents=True, exist_ok=True)


    # 2. Load COCO data
    print(f"Loading annotations from '{COCO_FILE}'...")
    try:
        with open(COCO_FILE, "r") as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at '{COCO_FILE}'")
        return
    
    images_info = {img['id']: img for img in coco_data.get('images', [])}
    annotations = coco_data.get('annotations', [])

    if not annotations:
        print("No annotations found in the file.")
        return

    # 3. Process, crop, and save all valid bounding boxes
    saved_count = 0
    total_annotations = len(annotations)
    print(f"\nFound {total_annotations} annotations. Starting cropping process...")

    for ann in annotations:
        image_id = ann['image_id']
        x, y, w, h = ann['bbox']
        
        # Skip any boxes with no valid area
        if w <= 0 or h <= 0:
            continue

        # Get the source image path and open it
        try:
            image_info = images_info[image_id]
            image_path = SOURCE_IMAGE_DIR / Path(image_info['file_name']).name
            source_image = Image.open(image_path)
        except (KeyError, FileNotFoundError):
            print(f"Warning: Could not find source image for annotation {ann['id']}. Skipping.")
            continue

        # Define the crop box for Pillow: (left, upper, right, lower)
        crop_box = (x, y, x + w, y + h)
        cropped_image = source_image.crop(crop_box)

        # Create a unique filename for the cropped image
        original_stem = image_path.stem
        crop_filename = f"{original_stem}_crop_{ann['id']}.jpg"

        # Save the cropped image
        cropped_image.save(OUTPUT_CROPS_DIR / crop_filename)
        saved_count += 1

    print("\n--- Processing Complete ---")
    print(f"✅ Saved {saved_count} crops to '{OUTPUT_CROPS_DIR}'")

if __name__ == '__main__':
    crop_all_bounding_boxes()
