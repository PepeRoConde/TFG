from torch.utils.data import Sampler
import numpy as np

class ImageGroupedSampler(Sampler):
    """Shuffles images but keeps all patches of each image together."""
    
    def __init__(self, dataset, shuffle=True):
        self.N = dataset.N  # parches por imagen
        self.num_images = len(dataset.images_dir_ls)
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            image_order = np.random.permutation(self.num_images)
        else:
            image_order = np.arange(self.num_images)
        
        for img_i in image_order:
            start = img_i * self.N
            yield from range(start, start + self.N)
    
    def __len__(self):
        return self.num_images * self.N
