import os
import cv2 as cv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms.functional import crop
from torchvision.transforms import ToTensor

class Online_Dataset(Dataset):
    def __init__(self, drive_dir, tamano_patch = 32, data_augmentation=True, label_mode= 'vainilla', sigma = 3):
        # imagenes de 565 x 584
        self.ancho, self.alto = 565, 584
        self.tamano_patch = tamano_patch
        self.drive_dir = drive_dir
        self.label_mode = label_mode
        self.sigma = sigma 

        self.images_subdir = 'images'
        self.venas_subdir = '1st_manual'

    def __len__(self):
        return len(os.listdir(os.path.join(self.drive_dir, 'images')))

    def __getitem__(self, idx):
        idx += 21 # las imágenes empiezan en 21 (van de 21_traning.tif a 40 training.tif)

        esquina_x = np.random.randint(self.ancho - self.tamano_patch) # las uso en el crop
        esquina_y = np.random.randint(self.alto - self.tamano_patch)


        img_path = os.path.join(self.drive_dir, self.images_subdir, f'{idx}_training.tif')
        image = ToTensor()(Image.open(img_path)) # toda esta parafernalia porque decode_image() no admite tif 
        venas_path = os.path.join(self.drive_dir, self.venas_subdir, f'{idx}_manual1.gif')
        venas = Image.open(venas_path).convert('RGB') # lo de convert rgb es porque es gif
        # https://stackoverflow.com/questions/63493726/using-the-first-frame-of-a-gif-from-a-url-for-image-processing-in-the-same-way-a

        # https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.crop.html
        imagen_patch = crop(image, esquina_y, esquina_x, self.tamano_patch, self.tamano_patch)
        
        label = self._get_label(venas,esquina_x, esquina_y, self.label_mode)

        return imagen_patch, label

    def _get_label(self, venas, esquina_x, esquina_y, mode='vainilla'):
        if mode == 'vainilla': 
            central_x = esquina_x + np.floor(self.tamano_patch // 2)
            central_y = esquina_y + np.floor(self.tamano_patch // 2)                         
            #>>> np.unique(torchvision.io.decode_image('21_manual1.gif').numpy())
            #array([  0, 255], dtype=uint8) es decir, son binarias
            if venas.getpixel((central_x, central_y))[0] == 0: return 0
            else: return 1
        elif mode == 'gaussian':
            #https://www.geeksforgeeks.org/python/how-to-generate-2-d-gaussian-array-using-numpy/
            def gaussian_filter(sigma=3):
                x, y = np.meshgrid(np.linspace(-1, 1, self.tamano_patch),
                                   np.linspace(-1, 1, self.tamano_patch))
                dst = np.sqrt(x**2 + y**2)
                normal = 1 / (2 * np.pi * sigma**2)
                return  np.exp(-((dst)**2 / (2.0 * sigma**2))) * normal

            def tanh(x):
                return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
            
            venas = np.asarray(venas)    
            # recorte, uso (y, x) (en vez de (x, y)) porque numpy es h-w-c mientras que pil es w-h
            venas_patch = venas[esquina_y:esquina_y + self.tamano_patch, esquina_x:esquina_x + self.tamano_patch, :]
            # paso a gris https://numpy.org/doc/2.2/reference/generated/numpy.mean.html#numpy.mean
            venas_patch = venas_patch.mean(axis=-1) 
            #https://peps.python.org/pep-0465/
            venas_patch = venas_patch * gaussian_filter(self.sigma)  

            suma = np.sum(venas_patch) / 255
            return np.float32([tanh(suma + 1e-8), 1 - tanh(suma + 1e-8)])

