from src.models.architectures import (
    CRATE_small,
    CRATE_base,
    CRATE_large,
    CRATE_verysmall,
    CRATE_enana,
    CRATE_tiny,
    CRATE_base_demo,
)


def instantiate_model(arch, image_size, patch_size, num_classes=2, **kwargs):
    if arch == "CRATE_tiny":
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
    elif arch == "CRATE_base_demo":
        model = CRATE_base_demo()
    else:
        raise NotImplementedError(f"Architecture '{arch}' not implemented")

    print(f"==> Construido o Modelo: {arch}")

    return model
