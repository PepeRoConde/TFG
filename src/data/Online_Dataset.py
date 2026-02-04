import os
import numpy as np
from PIL import Image
from .Base_Dataset import BaseDataset
import torch


class Online_Dataset(BaseDataset):
    """
    Online DRIVE dataset with random patch extraction on-the-fly.
    
    Each call to __getitem__ extracts a random patch from a random image.
    This means:
    - No guarantee all pixels are seen in an epoch
    - Different patches every epoch
    - More memory efficient (no pre-processing)
    - Slower per-sample (loads full image each time)
    
    Image representation:
    - Returns numpy arrays [H, W, C] with uint8 values [0, 255]
    - Consistent with albumentations and other libraries
    
    Args:
        drive_dir: Path to DRIVE dataset directory
        tamano_patch: Size of square patches to extract (default: 32)
        data_augmentation: Whether to apply augmentation (default: True)
        label_mode: Label generation mode - 'vainilla', 'gaussian', or 'multiple' (default: 'vainilla')
        sigma: Sigma parameter for gaussian mode (default: 3)
        num_sigmas: Number of sigma scales for multiple mode (default: 4)
        total_epochs: Total number of training epochs (required if data_augmentation=True)
        warmup_epochs: Number of epochs for augmentation warmup (default: total_epochs // 2)
    """
    
    def __init__(
        self, 
        drive_dir: str, 
        tamano_patch: int = 32, 
        data_augmentation: bool = True, 
        label_mode: str = 'vainilla', 
        sigma: float = 3,
        num_sigmas: int = 4,
        total_epochs: int = None,
        warmup_epochs: int = None
    ):
        # Initialize base class
        super().__init__(
            data_augmentation=data_augmentation,
            label_mode=label_mode,
            sigma=sigma,
            num_sigmas=num_sigmas,
            tamano_patch=tamano_patch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs
        )
        
        # Image dimensions for DRIVE dataset
        self.ancho, self.alto = 565, 584  # width, height
        self.drive_dir = drive_dir
        
        # Subdirectories
        self.images_subdir = 'images'
        self.venas_subdir = '1st_manual'
    
    def __len__(self):
        images_dir = os.path.join(self.drive_dir, self.images_subdir)
        return len(os.listdir(images_dir))
    
    def __getitem__(self, idx):
        # DRIVE training images are numbered 21-40
        idx += 21
        
        # Random crop coordinates (top-left corner)
        esquina_left = np.random.randint(self.ancho - self.tamano_patch)
        esquina_top = np.random.randint(self.alto - self.tamano_patch)
        
        # Load image as numpy array [H, W, C] uint8 [0, 255]
        img_path = os.path.join(
            self.drive_dir, 
            self.images_subdir, 
            f'{idx}_training.tif'
        )
        image_pil = Image.open(img_path)
        image_array = np.array(image_pil)  # [H, W, C] uint8
        
        # Crop image patch
        imagen_patch = image_array[
            esquina_top:esquina_top + self.tamano_patch,
            esquina_left:esquina_left + self.tamano_patch,
            :
        ]
        
        # Load vessel mask as numpy array [H, W, C] uint8 [0, 255]
        venas_path = os.path.join(
            self.drive_dir, 
            self.venas_subdir, 
            f'{idx}_manual1.gif'
        )
        # Convert GIF to RGB to get first frame
        venas_pil = Image.open(venas_path).convert('RGB')
        venas_array = np.array(venas_pil)  # [H, W, C] uint8
        
        # Crop vessel mask patch
        venas_patch = venas_array[
            esquina_top:esquina_top + self.tamano_patch,
            esquina_left:esquina_left + self.tamano_patch,
            :
        ]
        
        # Apply augmentation (if enabled)
        # Returns numpy arrays [H, W, C] uint8
        imagen_patch, venas_patch = self.apply_augmentation(imagen_patch, venas_patch)
        
        # Generate label from vessel patch
        label = self.get_label_from_patch(venas_patch)

        imagen_patch = torch.from_numpy(imagen_patch).float() / 255.0  # [H, W, C] -> [H, W, C] float [0, 1]
        imagen_patch = imagen_patch.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        # Convert label if needed
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
            
        return imagen_patch, label


# Example usage
if __name__ == "__main__":
    # Training dataset with augmentation
    train_dataset = Online_Dataset(
        drive_dir='./DRIVE/training',
        tamano_patch=32,
        data_augmentation=True,
        label_mode='multiple',
        num_sigmas=5,
        total_epochs=100,
        warmup_epochs=50
    )
    
    print(f"Online Dataset:")
    print(f"  Dataset length: {len(train_dataset)}")
    print(f"  Label mode: {train_dataset.label_mode}")
    print(f"  Output size: {train_dataset.get_output_size()}")
    print(f"  Label shape: {train_dataset.get_label_shape()}")
    
    # Test dataset access
    image_patch, label = train_dataset[0]
    print(f"\nSample:")
    print(f"  Image patch shape: {image_patch.shape}, dtype: {image_patch.dtype}")
    print(f"  Image value range: [{image_patch.min()}, {image_patch.max()}]")
    print(f"  Label shape: {label.shape if hasattr(label, 'shape') else 'scalar'}")
    print(f"  Label: {label}")
    
    # Test epoch update
    print(f"\nAugmentation schedule:")
    for epoch in [0, 25, 50, 75, 99]:
        train_dataset.set_epoch(epoch)
