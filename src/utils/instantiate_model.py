"""Utility function to instantiate models based on architecture."""

from src.models.architectures import *


def instantiate_model(arch, image_size, patch_size, num_classes=2):
    """
    Instantiate a model based on architecture name.
    
    Args:
        arch (str): Architecture name (e.g., 'CRATE_tiny', 'vit_small')
        image_size (int): Patch/image size (tamano_patch)
        patch_size (int): Token size (tamano_token)
        num_classes (int): Number of output classes (default: 2)
    
    Returns:
        model: Instantiated PyTorch model
    
    Raises:
        NotImplementedError: If architecture is not supported
    """
    
    if arch == 'vit_tiny':
        model = vit_tiny_patch16(global_pool=True)
    elif arch == 'vit_small':
        model = vit_small_patch16(global_pool=True)
    elif arch == 'CRATE_tiny':
        model = CRATE_tiny(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == 'CRATE_tiny2nd':
        model = CRATE_tiny2nd(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_small":
        model = CRATE_small(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_base":
        model = CRATE_base(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_base2nd":
        model = CRATE_base2nd(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_large":
        model = CRATE_large(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_verysmall":
        model = CRATE_verysmall(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    elif arch == "CRATE_verysmall2nd":
        model = CRATE_verysmall2nd(image_size=image_size, patch_size=patch_size, num_classes=num_classes)
    else:
        raise NotImplementedError(f"Architecture '{arch}' not implemented")
    
    return model
