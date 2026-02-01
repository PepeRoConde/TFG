import albumentations as A


class AugmentationScheduler:
    """
    Scheduler for gradually increasing data augmentation probability during training.
    
    Implements a linear warmup schedule where augmentation probability increases
    from start_p to end_p over warmup_epochs, then stays constant.
    
    This curriculum learning approach helps models first learn basic features
    with less augmentation, then improve robustness with stronger augmentation.
    
    Args:
        total_epochs: Total number of training epochs
        warmup_epochs: Number of epochs for probability warmup (default: total_epochs // 2)
        start_p: Starting probability for augmentations (default: 0.1)
        end_p: Final probability for augmentations (default: 0.9)
    """
    
    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = None,
        start_p: float = 0.1,
        end_p: float = 0.9
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else total_epochs // 2
        self.start_p = start_p
        self.end_p = end_p
        
        # Validate parameters
        if not 0 <= start_p <= 1:
            raise ValueError(f"start_p must be in [0, 1], got {start_p}")
        if not 0 <= end_p <= 1:
            raise ValueError(f"end_p must be in [0, 1], got {end_p}")
        if start_p > end_p:
            raise ValueError(f"start_p ({start_p}) must be <= end_p ({end_p})")
        if self.warmup_epochs > total_epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be <= total_epochs ({total_epochs})"
            )
    
    def get_probability(self, epoch: int) -> float:
        """
        Get augmentation probability for a given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Augmentation probability in [start_p, end_p]
        """
        if epoch >= self.warmup_epochs:
            return self.end_p
        
        # Linear interpolation during warmup
        progress = epoch / self.warmup_epochs
        return self.start_p + (self.end_p - self.start_p) * progress
    
    def create_augmentation_pipeline(self, epoch: int) -> A.Compose:
        """
        Create augmentation pipeline with epoch-dependent probabilities.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Albumentations Compose object with adjusted probabilities
        """
        p = self.get_probability(epoch)
        
        return A.Compose([
            A.ElasticTransform(
                alpha=50, 
                sigma=5, 
                alpha_affine=None,
                p=p
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                p=p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=p
            ),
        ])
    
    def __repr__(self):
        return (
            f"AugmentationScheduler("
            f"total_epochs={self.total_epochs}, "
            f"warmup_epochs={self.warmup_epochs}, "
            f"start_p={self.start_p}, "
            f"end_p={self.end_p})"
        )
