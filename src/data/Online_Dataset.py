import os
import numpy as np
from PIL import Image
import torch
import math

from .Base_Dataset import BaseDataset
from .recorta_dataset import calcula_stride

class Online_Dataset(BaseDataset):
    
    def __init__(
        self, 
        drive_dir: str, 
        tamano_patch: int = 32, 
        aumento_datos: bool = True, 
        label_mode: str = 'vainilla', 
        sigma: float = 3,
        num_sigmas: int = 4,
        sobrelapamento: float = 0.1,
        total_epochs: int = None,
        warmup_epochs: int = None
    ):
        # Initialize base class
        super().__init__(
            aumento_datos=aumento_datos,
            label_mode=label_mode,
            sigma=sigma,
            num_sigmas=num_sigmas,
            tamano_patch=tamano_patch,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs
        )
        
        self.ancho, self.alto = 565, 584 
        self.drive_dir = drive_dir
        
        self.images_subdir = 'images'
        self.venas_subdir = '1st_manual'
        self.images_dir_ls = os.listdir(os.path.join(self.drive_dir, self.images_subdir))
    
        self.stride = calcula_stride(tamano_patch, sobrelapamento)
        self.columnas = math.floor(self.ancho / self.stride)
        self.filas = math.floor(self.alto / self.stride)
        self.N = self.columnas * self.filas # numero de parches por imagen

        # as imaxes van de: ou 21-36 para adestramento ou 36-39 para validacion
        # idx_0 valdra 21 ou 36 respectivamente
        self.idx_0 = min([int(imaxe.split('_')[0]) for imaxe in self.images_dir_ls])


    def __len__(self):
        return len(self.images_dir_ls) * self.N # cada imagen tiene N parches
    
    def __getitem__(self, idx):

        img_idx = (idx // self.N) + self.idx_0 
        parche_idx = idx % self.N
        x = (parche_idx // self.columnas) * self.stride
        y = (parche_idx % self.columnas) * self.stride
        
        # Carga imaxe como numpy array [H, W, C] uint8 [0, 255]
        img_path = os.path.join(self.drive_dir, self.images_subdir, f'{img_idx}_training.tif')
        image_pil = Image.open(img_path)
        image_array = np.array(image_pil)  # [H, W, C] uint8
        
        imagen_parche = image_array[x: x + self.tamano_patch, y: y + self.tamano_patch, :]
        
        # Carga mascara numpy array [H, W, C] uint8 [0, 255]
        venas_path = os.path.join(self.drive_dir, self.venas_subdir, f'{img_idx}_manual1.gif')
        venas_pil = Image.open(venas_path).convert('RGB')
        venas_array = np.array(venas_pil)  # [H, W, C] uint8
        
        
        venas_parche = venas_array[x : x + self.tamano_patch, y : y + self.tamano_patch, : ]
        
        # Aplica (se procede) aumento de datos, devolve numpy arrays [H, W, C] uint8
        imagen_parche, venas_patch = self.apply_augmentation(imagen_parche, venas_parche)
        
        etiqueta = self.get_etiqueta(venas_parche)

        imagen_parche = torch.from_numpy(imagen_parche).float() / 255.0  # [H, W, C] -> [H, W, C] float [0, 1]
        imagen_parche = imagen_parche.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        if isinstance(etiqueta, np.ndarray):
            etiqueta = torch.from_numpy(etiqueta)
            
        return imagen_parche, etiqueta
