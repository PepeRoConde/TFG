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
    """

    def __init__(self, data_dir, tamano_patch):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

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
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        dummy_label = torch.tensor(1, dtype=torch.long)
        return image, dummy_label
