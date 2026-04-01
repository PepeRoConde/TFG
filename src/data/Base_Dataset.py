from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset
from .Aumento_Datos import Aumento_Datos


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for DRIVE retinal vessel segmentation datasets.

    Provides shared functionality for:
    - Label generation (vainilla, gaussian, and multiple modes)
    - Data augmentation with epoch-dependent scheduling
    - Consistent numpy array representation
    - Parameter validation

    Subclasses must implement:
    - __len__(): Return dataset size
    - __getitem__(idx): Return (image_patch, label) tuple

    Image representation convention:
    - All images are stored and processed as numpy arrays [H, W, C] with values in [0, 255] (uint8)
    - This ensures consistency across PIL, OpenCV, and augmentation libraries
    """

    def __init__(
        self,
        aumento_datos: bool = True,
        label_mode: str = "vainilla",
        sigma: float = 3,
        num_sigmas: int = 4,
        tamano_patch: int = 32,
        total_epochs: int = None,
        warmup_epochs: int = None,
        modo_aumento_datos: str = "fixo",
    ):
        """
        Initialize base dataset with shared parameters.

        Args:
            aumento_datos: Whether to apply augmentation
            label_mode: Label generation mode - 'vainilla', 'gaussian', or 'multiple'
            sigma: Sigma parameter for gaussian mode
            num_sigmas: Number of sigma scales for multiple mode (generates 2^(i/2) for i in range(num_sigmas))
            tamano_patch: Size of square patches
            total_epochs: Total number of training epochs (required if aumento_datos=True)
            warmup_epochs: Number of epochs for augmentation warmup (default: total_epochs // 2)
        """
        self.aumento_datos = aumento_datos
        self.label_mode = label_mode
        self.sigma = sigma
        self.num_sigmas = num_sigmas
        self.tamano_patch = tamano_patch
        self.current_epoch = 0

        # Validate label_mode
        if self.label_mode not in ["vainilla", "gaussian", "multiple"]:
            raise ValueError(
                f"Invalid label_mode: '{label_mode}'. Must be 'vainilla', 'gaussian', or 'multiple'"
            )

        # Setup augmentation scheduler
        if self.aumento_datos:
            if total_epochs is None:
                raise ValueError(
                    "total_epochs must be specified when aumento_datos=True"
                )
            self.aug_scheduler = Aumento_Datos(
                epocas_totais=total_epochs,
                epocas_quecemento=warmup_epochs,
                modo=modo_aumento_datos,
            )
            # Initialize augmentation pipeline for epoch 0
            self.augmentation = self.aug_scheduler.create_augmentation_pipeline(0)
        else:
            self.aug_scheduler = None
            self.augmentation = None

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_patch, label)
            - image_patch: numpy array [H, W, C] with uint8 values [0, 255]
            - label: int or numpy array depending on label_mode
        """
        pass

    def set_epoch(self, epoch: int):
        """
        Update the current epoch and rebuild augmentation pipeline.

        This should be called at the beginning of each training epoch,
        BEFORE creating the DataLoader iterator.

        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.current_epoch = epoch
        if self.aumento_datos and self.aug_scheduler is not None:
            self.augmentation = self.aug_scheduler.create_augmentation_pipeline(epoch)

    def apply_augmentation(self, imagen_patch: np.ndarray, venas_patch: np.ndarray):
        """
        Apply augmentation to image and mask patches.

        Args:
            imagen_patch: Image patch numpy array [H, W, C] with uint8 values [0, 255]
            venas_patch: Vessel mask patch numpy array [H, W, C] with uint8 values [0, 255]

        Returns:
            Tuple of (augmented_image_array, augmented_mask_array), both numpy [H, W, C] uint8
        """
        if not self.aumento_datos or self.augmentation is None:
            return imagen_patch, venas_patch

        # Apply same augmentation to both image and mask
        if venas_patch is not None:
            augmented = self.augmentation(image=imagen_patch, mask=venas_patch)

            return augmented["image"], augmented["mask"]

        else:
            augmented = self.augmentation(
                image=imagen_patch,
            )

            return augmented["image"]

    def get_etiqueta(self, venas_patch: np.ndarray):
        """
        Generate label from vessel mask patch.

        Args:
            venas_patch: Binary vessel mask patch [H, W, C], values are 0 or 255 (uint8)

        Returns:
            Label format depends on mode:
            - 'vainilla': int (0 or 1)
            - 'gaussian': np.ndarray([neg_score, pos_score]) shape [2]
            - 'multiple': np.ndarray([score_sigma1, score_sigma2, ...]) shape [num_sigmas]
        """
        if self.label_mode == "vainilla":
            # Binary classification: vessel (1) or background (0) at center pixel
            centro = self.tamano_patch // 2
            # venas_patch is [H, W, C], access as [y, x, channel]
            pixel_value = venas_patch[centro, centro, 0]
            return 0 if pixel_value == 0 else 1

        elif self.label_mode == "gaussian":
            # Normalize vessel mask to [0, 1]
            venas_gray = venas_patch.mean(axis=-1) / 255.0  # [H, W] in [0, 1]

            gaussian_kernel = self._gaussian_kernel_2d(self.sigma)  # Already sums to 1
            weighted_sum = np.sum(
                venas_gray * gaussian_kernel
            )  # No need to divide again

            # Apply tanh activation for soft labels
            pos_score = np.tanh(weighted_sum)  # Now input is [0, 1]
            neg_score = 1 - pos_score

            return np.float32([neg_score, pos_score])
        elif self.label_mode == "multiple":
            # Multi-scale weighted vessel density
            # Generate sigmas: 1, sqrt(2), 2, sqrt(4), 4, sqrt(8), ...
            sigmas = [2 ** (i / 2) for i in range(self.num_sigmas)]

            # Convert to grayscale once
            venas_gray = venas_patch.mean(axis=-1)  # [H, W]

            # Compute weighted sum for each sigma
            weights = []
            for sigma in sigmas:
                kernel = self._gaussian_kernel_2d(sigma)
                weighted_sum = np.sum(venas_gray * kernel) / 255.0
                weight = np.tanh(weighted_sum)
                weights.append(weight)

            return np.float32(weights)  # [num_sigmas]

    def _gaussian_kernel_2d(self, sigma: float) -> np.ndarray:
        x, y = np.meshgrid(
            np.linspace(
                -self.tamano_patch // 2, self.tamano_patch // 2, self.tamano_patch
            ),
            np.linspace(
                -self.tamano_patch // 2, self.tamano_patch // 2, self.tamano_patch
            ),
        )

        distancia_euclidea = x**2 + y**2
        kernel = np.exp(-distancia_euclidea / (2.0 * sigma**2))

        return kernel / kernel.sum()

    def get_label_shape(self):
        if self.label_mode == "vainilla":
            return ()
        elif self.label_mode == "gaussian":
            return (2,)
        elif self.label_mode == "multiple":
            return (self.num_sigmas,)

    def get_output_size(self):
        if self.label_mode == "vainilla":
            return 2
        elif self.label_mode == "gaussian":
            return 2
        elif self.label_mode == "multiple":
            return self.num_sigmas
