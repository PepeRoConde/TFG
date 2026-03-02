import os
import pandas as pd
from PIL import Image
import numpy as np
from .Base_Dataset import BaseDataset

class RFMiDDataset(BaseDataset):
    def __init__(self, data_dir, labels_file, augmentation=True, tamano_patch=32):
        super().__init__(
            aumento_datos=augmentation,
            tamano_patch=tamano_patch
        )
        self.data_dir = data_dir
        self.labels = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.data_dir, f"{row['ID']}.png")
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        label = row[1:].values.astype(np.float32)  # Assuming labels are from column 1 onwards

        if self.augmentation:
            image, label = self.apply_augmentation(image, label)

        return image, label