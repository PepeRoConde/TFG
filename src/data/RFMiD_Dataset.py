import os
import csv
import cv2
import numpy as np
import torch
from .Base_Dataset import BaseDataset


class RFMiDDataset(BaseDataset):
    def __init__(
        self, data_dir, aumento_datos=False, tamano_patch=1005, total_epochs=2000
    ):
        super().__init__(
            aumento_datos=aumento_datos,
            tamano_patch=tamano_patch,
            total_epochs=total_epochs,
        )
        self.data_dir = data_dir
        csv_path = os.path.join(self.data_dir, "labels.csv")
        self.labels = []
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                self.labels.append(row)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels[idx]
        img_path = os.path.join(self.data_dir, "images", f"{row[0]}.png")

        # Load image with OpenCV (faster than PIL) - returns numpy array
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = np.array(
            row[1], dtype=np.int64
        )  # Assuming labels are from column 1 onwards

        h, w, _ = image.shape
        if h >= self.tamano_patch and w >= self.tamano_patch:
            top = (h - self.tamano_patch) // 2
            left = (w - self.tamano_patch) // 2
            image = image[
                top : top + self.tamano_patch, left : left + self.tamano_patch
            ]

        if self.aumento_datos:
            image = self.apply_augmentation(image, None)

        # Direct conversion from numpy to torch without redundant copies
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label
