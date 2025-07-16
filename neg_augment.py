import os
import cv2
import argparse
import albumentations as A
from pathlib import Path

def create_anomaly_augmentations():
    """
    Defines a comprehensive augmentation pipeline to intentionally create
    anomalies from good data. Each augmentation creates a different type of defect.
    The parameters have been increased to make the defects more definitive.
    """
    transform = A.Compose([
        # Apply ONE of the following defect types to each image.
        A.OneOf([
            # --- Defect 1: Deformation ---
            # Severely warps the geometry of the object. Parameters increased by ~50%.
            A.ElasticTransform(
                alpha=75, sigma=15, alpha_affine=15, p=1.0
            ),

            # --- Defect 2: Holes / Missing Parts ---
            # Cuts out multiple larger black holes from the image. Parameters increased by ~50%.
            A.CoarseDropout(
                max_holes=12, max_height=60, max_width=60,
                min_holes=6, min_height=30, min_width=30,
                fill_value=0, # Black holes
                p=1.0
            ),

            # --- Defect 3: Foreign Objects / Bright Spots ---
            # Simulates larger paint splotches or light spots. Parameters increased by ~50%.
            A.CoarseDropout(
                max_holes=12, max_height=60, max_width=60,
                min_holes=6, min_height=30, min_width=30,
                fill_value=255, # White spots
                p=1.0
            ),

            # --- Defect 4: Severe Color Inversion ---
            # Inverts the image colors, which is a major anomaly.
            A.InvertImg(p=1.0),

        ], p=1.0), # Always apply one of the anomaly types.
    ])
    return transform

def main(args):
    """
    Main function to process images in the input directory and save
    synthetically generated ANOMALOUS versions to the output directory.
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    num_versions = args.num_versions

    # --- Validate paths ---
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # --- Create output directory if it doesn't exist ---
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: '{output_dir}'")

    # --- Initialize augmentation pipeline ---
    augmentation_pipeline = create_anomaly_augmentations()

    # --- Find image files ---
    image_files = list(input_dir.glob('*.png')) + \
                  list(input_dir.glob('*.jpg')) + \
                  list(input_dir.glob('*.jpeg'))

    if not image_files:
        print(f"No images found in '{input_dir}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images to turn into anomalies.")

    # --- Process each image ---
    for img_path in image_files:
        try:
            # Load the image using OpenCV.
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            print(f"Generating anomalies for {img_path.name}...")

            # --- Generate and save augmented versions ---
            for i in range(num_versions):
                # Apply the augmentation pipeline to the image
                augmented = augmentation_pipeline(image=image)
                augmented_image = augmented['image']

                # Construct a new, unique filename for the augmented image
                base_name = img_path.stem
                extension = img_path.suffix
                new_filename = f"{base_name}_anomaly_{i+1}{extension}"
                output_path = output_dir / new_filename

                # Save the new image
                cv2.imwrite(str(output_path), augmented_image)

        except Exception as e:
            print(f"An error occurred while processing {img_path.name}: {e}")

    print("\nAnomaly generation complete!")
    print(f"Generated {len(image_files) * num_versions} new anomalous images in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply anomaly-creating augmentations to a directory of good images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing the original 'good' images."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where anomalous images will be saved."
    )
    parser.add_argument(
        "--num-versions",
        type=int,
        default=5,
        help="Number of anomalous versions to create for each original image."
    )

    args = parser.parse_args()
    main(args)
