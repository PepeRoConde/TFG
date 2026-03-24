from src.models.architectures import *


def instantiate_model(arch, image_size, patch_size, num_classes=2, **kwargs):
    if arch == "vit_tiny":
        model = vit_tiny_patch16(global_pool=True)
    elif arch == "vit_small":
        model = vit_small_patch16(global_pool=True)
    elif arch == "CRATE_tiny":
        model = CRATE_tiny(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    elif arch == "CRATE_small":
        model = CRATE_small(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    elif arch == "CRATE_base":
        model = CRATE_base(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    elif arch == "CRATE_large":
        model = CRATE_large(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    elif arch == "CRATE_verysmall":
        model = CRATE_verysmall(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    elif arch == "CRATE_enana":
        model = CRATE_enana(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError(f"Architecture '{arch}' not implemented")

    print(f"==> Construido o Modelo: {arch}")

    return model
