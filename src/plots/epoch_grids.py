import random
import numpy as np
from matplotlib import pyplot as plt


def epoch_grids(dataset, epochs=[0, 20, 50, 100], grid_size=5):
    """
    Plot grids of random images from the dataset for each specified epoch.
    
    Parameters:
    -----------
    dataset : Offline_Dataset
        The dataset to sample from
    epochs : list
        List of epoch numbers to visualize
    grid_size : int
        Size of the grid (grid_size x grid_size images)
    """
    n_images = grid_size * grid_size
    
    for epoch in epochs:
        # Set the epoch in the dataset
        dataset.set_epoch(epoch)
        
        # Sample random indices
        total_samples = len(dataset)
        random_indices = random.sample(range(total_samples), n_images)
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f'Epoca {epoch}, p(aumento): {dataset.aug_scheduler.get_probability(epoch)}', fontsize=16, fontweight='bold')
        
        # Plot each image
        for idx, ax in enumerate(axes.flat):
            img, label = dataset[random_indices[idx]]
            
            # Convert tensor to numpy and transpose to (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            
            # Display image
            ax.imshow(img_np)
            ax.set_title(f'Label: {label}', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
