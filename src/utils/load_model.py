from .instantiate_model import instantiate_model
import torch


def load_model(weights_path, arch, patch_size, token_size, num_classes, **kwargs):
    print(f"==> Cargando modelo {arch} dende {weights_path}...")

    model = instantiate_model(arch, patch_size, token_size, num_classes, **kwargs)

    if torch.cuda.is_available():
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location="cpu")

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["model"])

    return model
