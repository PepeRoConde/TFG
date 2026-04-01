from pathlib import Path
from PIL import Image
from tqdm import tqdm


def reescala_dataset(input_dir: str, output_dir: str, size: int = 512):
    """Resize RFMiD dataset images to 512x512."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reescalando de {input_dir} a {output_dir} ({size}x{size})")

    # Process each split
    for split in ["Training_Set", "Test_Set", "Validation_Set", "Evaluation_Set"]:
        split_input = input_path / split
        split_output = output_path / split

        # Find and resize images in the 'images' subdirectory
        images_dir = split_input / "images"
        if not images_dir.exists():
            continue

        output_images = split_output / "images"
        output_images.mkdir(parents=True, exist_ok=True)

        image_files = sorted(images_dir.glob("*.png")) + sorted(
            images_dir.glob("*.jpg")
        )

        print(f"  {split}: {len(image_files)} imágenes")
        for img_file in tqdm(image_files, leave=False):
            img = Image.open(img_file)
            # Crop to square from center, removing lateral black padding
            min_dim = min(img.width, img.height)
            left = (img.width - min_dim) // 2
            top = (img.height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img_cropped = img.crop((left, top, right, bottom))
            # Resize to target size
            img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)
            img_resized.save(output_images / img_file.name)


if __name__ == "__main__":
    import sys

    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    input_dir = sys.argv[2] if len(sys.argv) > 2 else "data/RFMiD"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else f"data/RFMiD_{size}x{size}"
    reescala_dataset(input_dir, output_dir, size)
