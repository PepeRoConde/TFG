from src.models.crate import CRATE
from torch import nn

model_names = [
    "CRATE_tiny",
    "CRATE_small",
    "CRATE_base",
    "CRATE_large",
    "CRATE_enana",
    "CRATE_enana_6",
]


def CRATE_enana(image_size=64, patch_size=16, num_classes=2, **kwargs):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=4,
        heads=3,
        dropout=0.0,
        emb_dropout=0.1,
        dim_head=192 // 3,
        **kwargs,
    )


def CRATE_enana_6(image_size=64, patch_size=16, num_classes=2, **kwargs):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=4,
        heads=6,
        dropout=0.0,
        emb_dropout=0.1,
        dim_head=384 // 6,
        **kwargs,
    )


def CRATE_tiny(image_size=64, patch_size=16, num_classes=2, **kwargs):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6,
        **kwargs,
    )


def CRATE_small(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12,
    )


def CRATE_base(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12,
    )


def CRATE_large(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16,
    )


def CRATE_base_demo():
    model = CRATE(
        image_size=224,
        patch_size=8,
        num_classes=21842,
        dim=768,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 6,
    )

    model.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 768))
    model.head = nn.Linear(768, 21842)
    return model
