import os
from typing import List

import cv2
import torch
from torch.utils.data import Dataset


class ImagenetDemoDataset(Dataset):
    """
    Minimal dataset for demo folders with plain images.

    It scans a directory for image files, loads them with OpenCV,
    resizes each image to tamano_patch while preserving aspect ratio,
    and returns (image_tensor, dummy_label).

    The dataset can expose a virtual length. Sample i is mapped to
    i % num_real_images, so memory usage stays proportional to the
    amount of real images on disk (8 in your demo folder).
    """

    def __init__(self, data_dir, tamano_patch, virtual_length=1000, cache_images=True):
        self.data_dir = data_dir
        self.tamano_patch = int(tamano_patch)

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        self.image_paths: List[str] = []

        for filename in sorted(os.listdir(self.data_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_exts:
                self.image_paths.append(os.path.join(self.data_dir, filename))

        if not self.image_paths:
            raise ValueError(f"No images found in: {self.data_dir}")

        self.num_real_images = len(self.image_paths)

        requested_virtual_length = int(virtual_length)
        if requested_virtual_length <= 0:
            raise ValueError(
                f"virtual_length must be > 0, got {requested_virtual_length}"
            )
        # Keep all real images visible when there are more than virtual_length.
        self.virtual_length = max(requested_virtual_length, self.num_real_images)

        self.cache_images = bool(cache_images)
        self._cached_images: List[torch.Tensor] = []
        if self.cache_images:
            self._cached_images = [
                self._load_and_preprocess_image(path) for path in self.image_paths
            ]

        self._dummy_label = torch.tensor(1, dtype=torch.long)

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError(f"Invalid image shape in: {image_path}")

        scale = self.tamano_patch / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Demo images are square, so this branch typically keeps exact tamano_patch x tamano_patch.
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        real_idx = int(idx) % self.num_real_images

        if self.cache_images:
            image = self._cached_images[real_idx]
        else:
            image = self._load_and_preprocess_image(self.image_paths[real_idx])

        return image, self._dummy_label
