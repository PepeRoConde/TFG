from .instantiate_model import instantiate_model
import torch


def load_model(weights_path, arch, patch_size, token_size, **kwargs):

    print(f"==> Cargando modelo {arch} dende {weights_path}...")

    model = instantiate_model(arch, patch_size, token_size, num_classes=2, **kwargs)

    checkpoint = torch.load(weights_path)

    model.load_state_dict(checkpoint["state_dict"])

    return model
