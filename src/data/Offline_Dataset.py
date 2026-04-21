import json
from pathlib import Path
import numpy as np
from PIL import Image
from .Base_Dataset import BaseDataset
import torch


class Offline_Dataset(BaseDataset):
    """
    Offline DRIVE dataset that loads pre-cropped patches.

    Expects patches to be pre-generated using crop_dataset.py script.
    This means:
    - All patches are pre-defined (deterministic)
    - Full coverage guaranteed per epoch
    - Faster per-sample (no full image loading)
    - Requires pre-processing step

    Image representation:
    - Returns numpy arrays [H, W, C] with uint8 values [0, 255]
    - Consistent with albumentations and other libraries

    Directory structure expected:
        patches_dir/
        ├── images/
        │   ├── patch_00000_img.png
        │   ├── patch_00001_img.png
        │   └── ...
        ├── masks/
        │   ├── patch_00000_mask.png
        │   ├── patch_00001_mask.png
        │   └── ...
        └── metadata.json

    Args:
        patches_dir: Path to directory containing pre-cropped patches
        data_augmentation: Whether to apply augmentation (default: True)
        label_mode: Label generation mode - 'vainilla', 'gaussian', or 'multiple' (default: 'vainilla')
        sigma: Sigma parameter for gaussian mode (default: 3)
        num_sigmas: Number of sigma scales for multiple mode (default: 4)
        total_epochs: Total number of training epochs (required if data_augmentation=True)
        warmup_epochs: Number of epochs for augmentation warmup (default: total_epochs // 2)
    """

    def __init__(
        self,
        patches_dir: str,
        data_augmentation: bool = True,
        label_mode: str = "vainilla",
        sigma: float = 3,
        num_sigmas: int = 4,
        total_epochs: int = None,
        warmup_epochs: int = None,
    ):
        self.patches_dir = Path(patches_dir)

        # Load metadata
        metadata_path = self.patches_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                "Did you run crop_dataset.py first?"
            )

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Extract patch size from metadata
        tamano_patch = self.metadata["patch_size"]

        # Initialize base class
        super().__init__(
            data_augmentation=data_augmentation,
            label_mode=label_mode,
            sigma=sigma,
            num_sigmas=num_sigmas,
            tamano_patch=tamano_patch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
        )

        # Subdirectories
        self.images_dir = self.patches_dir / "images"
        self.masks_dir = self.patches_dir / "masks"

        # Validate directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Get list of patches from metadata
        self.patches_info = self.metadata["patches"]

        # print(f"Loaded Offline Dataset:")
        # print(f"  Patches directory: {patches_dir}")
        # print(f"  Total patches: {len(self.patches_info)}")
        # print(f"  Patch size: {tamano_patch}x{tamano_patch}")
        # print(f"  Overlap rate: {self.metadata['overlap_rate']:.2%}")
        # print(f"  Stride: {self.metadata['stride']}")
        # print(f"  Label mode: {self.label_mode}")
        # if self.label_mode == 'multiple':
        #    print(f"  Number of sigmas: {self.num_sigmas}")
        # print(f"  Output size: {self.get_output_size()}")

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        # Get patch info from metadata
        patch_info = self.patches_info[idx]

        # Load image patch as numpy array [H, W, C] uint8 [0, 255]
        img_path = self.images_dir / patch_info["image_file"]
        image_pil = Image.open(img_path)
        imagen_patch = np.array(image_pil)  # [H, W, C] uint8

        # Load mask patch as numpy array [H, W, C] uint8 [0, 255]
        mask_path = self.masks_dir / patch_info["mask_file"]
        mask_pil = Image.open(mask_path)
        venas_patch = np.array(mask_pil)  # [H, W, C] uint8

        # Apply augmentation (if enabled)
        # Returns numpy arrays [H, W, C] uint8
        imagen_patch, venas_patch = self.apply_augmentation(imagen_patch, venas_patch)

        # Generate label from vessel patch
        label = self.get_label_from_patch(venas_patch)

        imagen_patch = (
            torch.from_numpy(imagen_patch).float() / 255.0
        )  # [H, W, C] -> [H, W, C] float [0, 1]
        imagen_patch = imagen_patch.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        # Convert label if needed
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)

        return imagen_patch, label

    def get_patch_info(self, idx: int) -> dict:
        """
        Get metadata for a specific patch.

        Args:
            idx: Patch index

        Returns:
            Dictionary with patch metadata (source_image, position, etc.)
        """
        return self.patches_info[idx]

    def get_dataset_info(self) -> dict:
        """
        Get overall dataset metadata.

        Returns:
            Dictionary with dataset metadata (patch_size, overlap_rate, etc.)
        """
        return {
            "patch_size": self.metadata["patch_size"],
            "overlap_rate": self.metadata["overlap_rate"],
            "stride": self.metadata["stride"],
            "total_patches": len(self.patches_info),
            "image_range": (
                self.metadata["image_start_idx"],
                self.metadata["image_end_idx"],
            ),
        }


# Example usage
if __name__ == "__main__":
    # Offline dataset (requires pre-cropping with crop_dataset.py)
    try:
        offline_dataset = Offline_Dataset(
            patches_dir="./DRIVE/training_patches",
            data_augmentation=True,
            label_mode="multiple",
            num_sigmas=5,
            total_epochs=100,
            warmup_epochs=50,
        )

        print("\nOffline Dataset Test:")

        # Test dataset access
        image_patch, label = offline_dataset[0]
        print("\nSample:")
        print(f"  Image patch shape: {image_patch.shape}, dtype: {image_patch.dtype}")
        print(f"  Image value range: [{image_patch.min()}, {image_patch.max()}]")
        print(f"  Label shape: {label.shape if hasattr(label, 'shape') else 'scalar'}")
        print(f"  Label: {label}")

        # Get patch info
        patch_info = offline_dataset.get_patch_info(0)
        print("\nPatch 0 info:")
        print(f"  Source image: {patch_info['source_image']}")
        print(f"  Position: (top={patch_info['top']}, left={patch_info['left']})")

        # Get dataset info
        dataset_info = offline_dataset.get_dataset_info()
        print("\nDataset info:")
        for key, value in dataset_info.items():
            print(f"  {key}: {value}")

        # Test epoch update
        print("\nAugmentation schedule:")
        for epoch in [0, 25, 50, 75, 99]:
            offline_dataset.set_epoch(epoch)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo create patches, run:")
        print("  python crop_dataset.py --input_dir ./DRIVE/training \\")
        print("                         --output_dir ./DRIVE/training_patches \\")
        print("                         --patch_size 32 \\")
        print("                         --overlap_rate 0.5")
