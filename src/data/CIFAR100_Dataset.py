import os
import pickle
import numpy as np
import torch
from .Base_Dataset import BaseDataset


class CIFAR100Dataset(BaseDataset):
    """
    CIFAR100 dataset class that inherits from BaseDataset.

    CIFAR100 contains:
    - 60,000 32x32 RGB images
    - 100 classes (fine-grained labels)
    - Data organized in 5 training batches + 1 test batch

    Expected directory structure:
    data_dir/
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        test_batch (optional)
    """

    def __init__(
        self,
        data_dir,
        aumento_datos=False,
        label_mode="vainilla",
        sigma=3,
        num_sigmas=4,
        tamano_patch=32,
        total_epochs=2000,
        warmup_epochs=None,
        modo_aumento_datos="fixo",
        split="train",
    ):
        """
        Initialize CIFAR100 dataset.

        Args:
            data_dir: Directory containing CIFAR100 batch files (data_batch_1 to data_batch_5)
            aumento_datos: Whether to apply data augmentation
            label_mode: Label generation mode ('vainilla', 'gaussian', or 'multiple')
            sigma: Sigma parameter for gaussian label mode
            num_sigmas: Number of sigma scales for multiple label mode
            tamano_patch: Size of patches (CIFAR100 images are 32x32, so recommended=32)
            total_epochs: Total number of training epochs
            warmup_epochs: Number of epochs for augmentation warmup
            modo_aumento_datos: Augmentation mode ('fixo', etc.)
            split: 'train' or 'test' - which data split to load
        """
        super().__init__(
            aumento_datos=aumento_datos,
            label_mode=label_mode,
            sigma=sigma,
            num_sigmas=num_sigmas,
            tamano_patch=tamano_patch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            modo_aumento_datos=modo_aumento_datos,
        )

        self.data_dir = data_dir
        self.split = split
        self.images = []
        self.labels = []

        # Load appropriate batch files
        if split == "train":
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        elif split == "test":
            batch_files = ["test_batch"]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")

        # Load and concatenate all batches
        for batch_file in batch_files:
            batch_path = os.path.join(data_dir, batch_file)
            if not os.path.exists(batch_path):
                raise FileNotFoundError(f"Batch file not found: {batch_path}")

            with open(batch_path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            # CIFAR100 binary format: each row of data is 3072 bytes
            # First 1024 bytes are red, next 1024 green, last 1024 blue
            batch_images = batch[b"data"]  # Shape: (N, 3072)
            batch_labels = batch[b"fine_labels"]  # Fine labels for CIFAR100

            # Reshape images from (N, 3072) to (N, 3, 32, 32)
            # CIFAR100 stores as R, G, B channels sequentially
            batch_images = batch_images.reshape(-1, 3, 32, 32)
            # Convert from NCHW to NHWC format for consistency with image processing
            batch_images = batch_images.transpose(0, 2, 3, 1)  # (N, 32, 32, 3)

            self.images.append(batch_images)
            self.labels.extend(batch_labels)

        # Concatenate all batches
        self.images = np.concatenate(self.images, axis=0)  # (N, 32, 32, 3)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, label)
            - image: torch tensor of shape [C, H, W] with float32 values in [0, 1]
            - label: torch tensor with class label (0-99)
        """
        # Get image and label
        image = self.images[idx]  # Shape: (32, 32, 3), dtype: uint8, values: [0, 255]
        label = self.labels[idx]  # Single integer label

        # Apply augmentation if enabled
        if self.aumento_datos:
            image = self.apply_augmentation(image, None)

        # Convert to float and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32) / 255.0

        # Permute from HWC to CHW format
        image = image.permute(2, 0, 1)  # (3, 32, 32)

        label = torch.tensor(label, dtype=torch.int64)

        return image, label

    def set_epoch(self, epoch: int):
        """Update the augmentation pipeline for the current epoch."""
        if self.aumento_datos and self.aug_scheduler is not None:
            self.current_epoch = epoch
            self.augmentation = self.aug_scheduler.create_augmentation_pipeline(epoch)
