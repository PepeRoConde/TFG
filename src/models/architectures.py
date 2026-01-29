from src.models.crate import CRATE

def CRATE_tiny(image_size=64, patch_size=16):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6
    )

def CRATE_small(image_size=64, patch_size=16):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12
    )

def CRATE_base(image_size=64, patch_size=16):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12
    )

def CRATE_large(image_size=64, patch_size=16):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=2,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16
    )
