import argparse
import os
from pathlib import Path

from src.utils import cargar_config_yaml, load_model, get_device


def _model_param_count(model):
    return sum(p.numel() for p in model.parameters())


def _model_size_mb(model):
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_total / (1024**2)


def _resolve_checkpoints(checkpoint_arg, logs_dir):
    if checkpoint_arg.lower() == "all":
        logs_dir = Path(logs_dir)
        log_names = sorted(
            p.stem for p in logs_dir.glob("*.log") if not p.name.startswith(".")
        )
        checkpoint_paths = sorted(
            [
                str(Path("data/weights") / f"{stem}.pth.tar")
                for stem in log_names
                if (Path("data/weights") / f"{stem}.pth.tar").exists()
            ]
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No matching checkpoints found in data/weights/ for logs in {logs_dir}."
            )
        return checkpoint_paths

    return [checkpoint_arg]


def _load_config_for_checkpoint(checkpoint_path, logs_dir):
    return cargar_config_yaml(checkpoint_path, logs_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Muestra el tamaño y número de parámetros de un checkpoint de CRATE."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path a un checkpoint .pth.tar, o 'all' para evaluar todos los que haya en data/weights.",
    )
    parser.add_argument(
        "logs_dir",
        type=str,
        help="Directorio que contiene metadata/ y los .log asociados.",
    )
    args = parser.parse_args()

    for checkpoint in _resolve_checkpoints(args.checkpoint_path, args.logs_dir):
        config = _load_config_for_checkpoint(checkpoint, args.logs_dir)

        model = load_model(
            weights_path=checkpoint,
            arch=config["arch"],
            patch_size=config["tamano_patch"],
            token_size=config["tamano_token"],
            num_classes=config.get("num_classes", 2),
            order=config.get("order", "first"),
            shared_u=config.get("shared_u", False),
            shared_dict=config.get("shared_dict", False),
        )
        model.to(get_device()).eval()

        name = os.path.basename(checkpoint).replace(".pth.tar", "")
        params = _model_param_count(model)
        size_mb = _model_size_mb(model)

        print(f"El modelo {name} ocupa {size_mb:.2f} MB y tiene {params:,} parametros")


if __name__ == "__main__":
    main()
