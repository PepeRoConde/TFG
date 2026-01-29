import CRATE

def CRATE_tiny(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=384 // 6
        )


def CRATE_small(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=576,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=576 // 12
        )


def CRATE_base(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=768 // 12
        )


def CRATE_large(num_classes=1000):
    return CRATE(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=1024,
        depth=24,
        heads=16,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=1024 // 16
        )
