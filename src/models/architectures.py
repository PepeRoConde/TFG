from src.models.crate import CRATE

model_names = [
    "vit_tiny", "vit_small",
    "CRATE_tiny", "CRATE_tiny2nd",
    "CRATE_small", "CRATE_base",
    "CRATE_base2nd", "CRATE_large",
    "CRATE_verysmall", "CRATE_verysmall2nd",
    "CRATE_enana", "CRATE_enana2nd"
]

def CRATE_tiny(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6
    )

def CRATE_tiny2nd(image_size=64, patch_size=16, num_classes=2):
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
        order='second'
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
        dim_head=576 // 12
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
        dim_head=768 // 12
    )

def CRATE_base2nd(image_size=64, patch_size=16, num_classes=2):
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
        order='second'
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
        dim_head=1024 // 16
    )

def CRATE_verysmall(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=6,
        heads=3,
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=192 // 3
    )

def CRATE_verysmall2nd(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=6,
        heads=3,
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=192 // 3,
        order='second'
    )

def CRATE_enana(image_size=64, patch_size=16, num_classes=2):
    return CRATE(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=192,
        depth=4,
        heads=3,
        dropout=0.0,
        emb_dropout=0.1,
        dim_head=192 // 3
    )

def CRATE_enana2nd(image_size=64, patch_size=16, num_classes=2):
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
        order='second'
    )
