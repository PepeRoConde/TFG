import os
import numpy as np
import cv2
import torch
import math

from .Base_Dataset import BaseDataset
from .recorta_dataset import calcula_stride


class Online_Dataset(BaseDataset):
    def __init__(
        self,
        drive_dir: str = "data/DRIVE/train",
        tamano_patch: int = 32,
        aumento_datos: bool = True,
        label_mode: str = "vainilla",
        sigma: float = 3,
        num_sigmas: int = 4,
        sobrelapamento: float = 0.1,
        total_epochs: int = None,
        warmup_epochs: int = None,
        modo_aumento_datos: str = "fixo",
        contador_aumento: int = -1,
    ):
        # Initialize base class
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

        self.ancho, self.alto = 565, 584
        self.drive_dir = drive_dir
        self.contador_aumento = contador_aumento

        self.images_subdir = "images"
        self.venas_subdir = "1st_manual"
        print(f"drive dir {self.drive_dir} ")
        self.images_dir_ls = os.listdir(
            os.path.join(self.drive_dir, self.images_subdir)
        )

        self.stride = calcula_stride(tamano_patch, sobrelapamento)
        self.columnas = math.floor((self.ancho - self.tamano_patch) / self.stride) + 1
        self.filas = math.floor((self.alto - self.tamano_patch) / self.stride) + 1
        self.N = self.columnas * self.filas  # numero de parches por imagen

        # as imaxes van de: ou 21-36 para adestramento ou 36-39 para validacion
        # idx_0 valdra 21 ou 36 respectivamente
        self.idx_0 = min([int(imaxe.split("_")[0]) for imaxe in self.images_dir_ls])
        # Image caching state
        self._cached_img_idx = None
        self._cached_image = None
        self._cached_venas = None

        # Augmented image caching state
        self._aug_img_idx = None
        self._aug_counter = 0
        self._augmented_image = None
        self._augmented_venas = None

    def __len__(self):
        return len(self.images_dir_ls) * self.N  # cada imagen tiene N parches

    def __getitem__(self, idx):
        img_idx = (idx // self.N) + self.idx_0
        parche_idx = idx % self.N
        x = (parche_idx // self.columnas) * self.stride
        y = (parche_idx % self.columnas) * self.stride

        # Get augmented full images (cached when possible)
        image_array, venas_array = self._get_augmented_image_pair(img_idx)

        # Extract patches
        imagen_parche = image_array[
            x : x + self.tamano_patch, y : y + self.tamano_patch, :
        ]
        venas_parche = venas_array[
            x : x + self.tamano_patch, y : y + self.tamano_patch, :
        ]

        # Verify shape (augmentation should preserve dimensions)
        assert (
            imagen_parche.shape[:2]
            == (
                self.tamano_patch,
                self.tamano_patch,
            )
        ), f"Patch shape mismatch: got {imagen_parche.shape[:2]}, expected ({self.tamano_patch}, {self.tamano_patch})"

        # Generate label
        etiqueta = self.get_etiqueta(venas_parche)

        # Direct conversion from numpy to torch with permutation in one step (no redundant copies)
        imagen_parche = (
            torch.from_numpy(imagen_parche).permute(2, 0, 1).float() / 255.0
        )  # [H, W, C] -> [C, H, W] float [0, 1]

        if isinstance(etiqueta, np.ndarray):
            etiqueta = torch.from_numpy(etiqueta)

        return imagen_parche, etiqueta

    def _load_raw_image_pair(self, img_idx):
        """Load and cache raw image pair from disk."""
        img_path = os.path.join(
            self.drive_dir, self.images_subdir, f"{img_idx}_training.tif"
        )
        # cv2.imread returns BGR by default, convert to RGB if needed
        self._cached_image = cv2.imread(img_path)
        if self._cached_image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        # For grayscale or BGR images, ensure RGB format
        if len(self._cached_image.shape) == 2:  # Grayscale
            self._cached_image = cv2.cvtColor(self._cached_image, cv2.COLOR_GRAY2RGB)
        elif self._cached_image.shape[2] == 3:  # BGR to RGB
            self._cached_image = cv2.cvtColor(self._cached_image, cv2.COLOR_BGR2RGB)

        venas_path = os.path.join(
            self.drive_dir, self.venas_subdir, f"{img_idx}_manual1.gif"
        )
        self._cached_venas = cv2.imread(venas_path)
        if self._cached_venas is None:
            raise FileNotFoundError(f"Cannot read vessel mask: {venas_path}")
        # Ensure vessel mask is RGB
        if len(self._cached_venas.shape) == 2:  # Grayscale
            self._cached_venas = cv2.cvtColor(self._cached_venas, cv2.COLOR_GRAY2RGB)
        elif self._cached_venas.shape[2] == 3:  # BGR to RGB
            self._cached_venas = cv2.cvtColor(self._cached_venas, cv2.COLOR_BGR2RGB)

        self._cached_img_idx = img_idx

    def _augment_full_images(self, img_idx):
        """Augment full cached images with deterministic seed."""
        # Set seed for reproducibility: same augmentation per image per epoch
        np.random.seed(img_idx * 10000 + self.current_epoch)

        # Apply augmentation to full images
        self._augmented_image, self._augmented_venas = self.apply_augmentation(
            self._cached_image, self._cached_venas
        )

        self._aug_img_idx = img_idx
        self._aug_epoch = self.current_epoch

    def _get_augmented_image_pair(self, img_idx):
        """Get augmented image pair with intelligent caching."""
        # Check if we need to re-augment
        needs_reaugment = (
            self._aug_img_idx != img_idx  # Different image
            or not hasattr(self, "_aug_epoch")
            or self._aug_epoch != self.current_epoch  # Different epoch
            or (
                self.contador_aumento > 0 and self._aug_counter >= self.contador_aumento
            )  # Counter exceeded
        )

        if needs_reaugment:
            # Load raw image if not cached
            if self._cached_img_idx != img_idx:
                self._load_raw_image_pair(img_idx)

            # Augment the full images
            self._augment_full_images(img_idx)
            self._aug_counter = 0

        self._aug_counter += 1
        return self._augmented_image, self._augmented_venas
