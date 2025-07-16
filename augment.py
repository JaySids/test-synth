import os
import cv2
import argparse
import albumentations as A
from pathlib import Path

def create_augmentations():
    """
    Defines a comprehensive augmentation pipeline to simulate the visual
    artifacts of a real-world camera, focusing on extremely low quality and
    fixed 90-degree rotations without any scaling/zooming.
    """
    transform = A.Compose([
        # --- Step 1: Geometric Transformations ---
        # Fixed 90, 180, 270-degree rotations and flips. No scaling or random rotation.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0), # Always applies a 0, 90, 180, or 270 rotation

        # --- Step 2: Quality and Camera Artifacts ---
        # These simulate motion and focus issues.
        A.OneOf([
            A.MotionBlur(blur_limit=(9, 17), p=0.6),
            A.GaussianBlur(blur_limit=(9, 17), p=0.5),
        ], p=0.9),

        # EXTREMELY aggressive pixelation for a very low-resolution effect.
        A.Downscale(
            scale_min=0.05,  # Drastically reduced for maximum quality loss
            scale_max=0.15, # Drastically reduced for maximum quality loss
            interpolation=0, # cv2.INTER_NEAREST
            p=1.0 # Always apply this effect
        ),
        
        # Minor color channel shifts that can occur with camera sensors
        A.ChannelShuffle(p=0.3),
    ])
    return transform

def main(args):
    """
    Main function to process images in the input directory and save
    augmented versions to the output directory.
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
    augmentation_pipeline = create_augmentations()

    # --- Find image files ---
    image_files = list(input_dir.glob('*.png')) + \
                  list(input_dir.glob('*.jpg')) + \
                  list(input_dir.glob('*.jpeg'))

    if not image_files:
        print(f"No images found in '{input_dir}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images to augment.")

    # --- Process each image ---
    for img_path in image_files:
        try:
            # Load the image using OpenCV. Albumentations works with numpy arrays.
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            print(f"Augmenting {img_path.name}...")

            # --- Generate and save augmented versions ---
            for i in range(num_versions):
                # Apply the augmentation pipeline to the image
                augmented = augmentation_pipeline(image=image)
                augmented_image = augmented['image']

                # Construct a new, unique filename for the augmented image
                base_name = img_path.stem
                extension = img_path.suffix
                new_filename = f"{base_name}_aug_{i+1}{extension}"
                output_path = output_dir / new_filename

                # Save the new image
                cv2.imwrite(str(output_path), augmented_image)

        except Exception as e:
            print(f"An error occurred while processing {img_path.name}: {e}")

    print("\nAugmentation complete!")
    print(f"Generated {len(image_files) * num_versions} new images in '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply realistic augmentations to a directory of images."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing the original images."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where augmented images will be saved."
    )
    parser.add_argument(
        "--num-versions",
        type=int,
        default=5,
        help="Number of augmented versions to create for each original image."
    )

    args = parser.parse_args()
    main(args)
