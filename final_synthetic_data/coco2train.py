import json
import numpy as np
import shutil
import random
import yaml
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# --- DATASET Configuration ---
COCO_FILE = Path("coco_annotations.json")
SOURCE_IMAGE_DIR = Path("images")
OUTPUT_DIR = Path("dataset")
TRAIN_RATIO = 0.8
STD_DEV_THRESHOLD = 2.0

# --- TRAINING Configuration ---
MODEL_TO_TRAIN = 'yolo11n.pt'  # Using YOLOv8n as the base model
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640

def process_and_train():
    """
    Processes a COCO dataset and then immediately starts the training process.
    """
    # 1. Clean and create output directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    train_img_path = OUTPUT_DIR / "images" / "train"
    val_img_path = OUTPUT_DIR / "images" / "val"
    train_lbl_path = OUTPUT_DIR / "labels" / "train"
    val_lbl_path = OUTPUT_DIR / "labels" / "val"
    for path in [train_img_path, val_img_path, train_lbl_path, val_lbl_path]:
        path.mkdir(parents=True, exist_ok=True)

    print("✅ Stage 1: Directory setup complete.")

    # 2. Load COCO data
    with open(COCO_FILE, "r") as f:
        coco_data = json.load(f)
    images_info = {img['id']: img for img in coco_data.get('images', [])}
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    print("✅ Stage 2: COCO annotation data loaded.")

    # 3. First Pass: Gather stats and filter outliers
    bbox_areas = []
    image_to_annotations = defaultdict(list)
    for ann in annotations:
        image_id = ann.get('image_id')
        bbox = ann.get('bbox')
        if not bbox or image_id not in images_info: continue
        _, _, bbox_w, bbox_h = bbox
        area = bbox_w * bbox_h
        if area > 0:
            bbox_areas.append(area)
            image_to_annotations[image_id].append(ann)

    mean_area = np.mean(bbox_areas)
    std_area = np.std(bbox_areas)
    lower_bound = mean_area - STD_DEV_THRESHOLD * std_area
    upper_bound = mean_area + STD_DEV_THRESHOLD * std_area
    
    valid_image_ids = []
    outlier_count = 0
    for image_id, anns in image_to_annotations.items():
        is_outlier = any(not (lower_bound <= (ann['bbox'][2] * ann['bbox'][3]) <= upper_bound) for ann in anns)
        if not is_outlier:
            valid_image_ids.append(image_id)
        else:
            outlier_count += 1
            
    print(f"✅ Stage 3: Outlier check complete. Removed {outlier_count} images.")

    # 4. Split data and write files
    random.shuffle(valid_image_ids)
    split_index = int(len(valid_image_ids) * TRAIN_RATIO)
    train_ids = set(valid_image_ids[:split_index])

    for image_id in valid_image_ids:
        image_info = images_info[image_id]
        is_train = image_id in train_ids
        dest_img_folder = train_img_path if is_train else val_img_path
        dest_lbl_folder = train_lbl_path if is_train else val_lbl_path

        source_img_file = SOURCE_IMAGE_DIR / Path(image_info['file_name']).name
        if source_img_file.exists():
            shutil.copy(source_img_file, dest_img_folder)
        else:
            continue

        label_filename = source_img_file.with_suffix('.txt').name
        output_path = dest_lbl_folder / label_filename
        
        yolo_lines = []
        for ann in image_to_annotations[image_id]:
            cat_id, bbox, img_w, img_h = ann['category_id'], ann['bbox'], image_info['width'], image_info['height']
            x_min, y_min, bbox_w, bbox_h = bbox
            cx = (x_min + bbox_w / 2) / img_w
            cy = (y_min + bbox_h / 2) / img_h
            w = bbox_w / img_w
            h = bbox_h / img_h
            yolo_lines.append(f"{cat_id - 1} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
        with open(output_path, "w") as f:
            f.writelines(yolo_lines)
    print("✅ Stage 4: Train/Val split and file creation complete.")

    # 5. Create data.yaml
    class_names = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]
    yaml_path = OUTPUT_DIR / 'data.yaml'
    yaml_data = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print("✅ Stage 5: data.yaml created successfully.")

    # 6. Start Training
    print("\n--- Starting Model Training ---")
    model = YOLO(MODEL_TO_TRAIN)
    
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=str(OUTPUT_DIR / 'runs'),
        name='train',
        exist_ok=True
    )
    print(f"🎉 All complete! Training run saved in '{OUTPUT_DIR / 'runs/train'}'")

if __name__ == '__main__':
    process_and_train()