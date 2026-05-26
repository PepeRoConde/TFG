import os
from src.data.Online_Dataset import Online_Dataset
from src.data.Offline_Dataset import Offline_Dataset
from src.data.recorta_dataset import recorta_dataset
from src.data.RFMiD_Dataset import RFMiDDataset
from src.data.ImagenetDataset import ImagenetDataset
from src.data.ImagenetDemoDataset import ImagenetDemoDataset


def _get_param(params, key, default=None):
    """
    Get parameter from either args (namespace) or config (dict).
    Handles both attribute access and dict access transparently.
    """
    if isinstance(params, dict):
        return params.get(key, default)
    else:
        return getattr(params, key, default)


def instantiate_dataset(args=None, config=None):
    """
    Instantiate datasets from either args (training) or config (evaluation).

    Args:
        args: Namespace object from argparse (used by main.py)
        config: Dict with config parameters (used by evaluation scripts)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    if args is None and config is None:
        raise ValueError("Either 'args' or 'config' must be provided")

    if args is not None and config is not None:
        raise ValueError("Cannot provide both 'args' and 'config'")

    # Use whichever parameter was provided
    params = args if args is not None else config

    # For evaluation with config, disable augmentation by default
    is_eval_mode = config is not None
    default_aumento = False if is_eval_mode else True

    if _get_param(params, "dataset", "online") == "online":
        train_dataset = Online_Dataset(
            _get_param(params, "directorio_train_base")
            or _get_param(params, "data_train_path", "data/DRIVE/train"),
            tamano_patch=_get_param(params, "tamano_patch"),
            label_mode=_get_param(params, "label_mode", "vainilla"),
            sigma=_get_param(params, "sigma", 3),
            num_sigmas=_get_param(params, "num_sigmas", 4),
            aumento_datos=_get_param(params, "aumento_datos", default_aumento),
            total_epochs=_get_param(params, "epochs"),
            sobrelapamento=_get_param(params, "overlap_rate", 0.1),
            contador_aumento=_get_param(params, "contador_aumento", -1),
        )

        val_dataset = Online_Dataset(
            _get_param(params, "directorio_val_base")
            or _get_param(params, "data_val_path", "data/DRIVE/val"),
            tamano_patch=_get_param(params, "tamano_patch"),
            label_mode=_get_param(params, "label_mode", "vainilla"),
            sigma=_get_param(params, "sigma", 3),
            num_sigmas=_get_param(params, "num_sigmas", 4),
            aumento_datos=False,  # No augmentation for validation
            total_epochs=_get_param(params, "epochs"),
            sobrelapamento=_get_param(params, "overlap_rate", 0.1),
            contador_aumento=_get_param(params, "contador_aumento", -1),
        )

    elif _get_param(params, "dataset") == "offline":
        # train -------

        directorio_train_base = _get_param(
            params, "directorio_train_base"
        ) or _get_param(params, "data_train_path")
        tamano_patch = _get_param(params, "tamano_patch")
        overlap_rate = _get_param(params, "overlap_rate", 0.1)

        directorio_train_cropeado = (
            f"{directorio_train_base}_{tamano_patch}_{overlap_rate}"
        )

        if not os.path.isdir(directorio_train_cropeado):
            print(
                "no hay el conjunto de datos de patches que quieres para entrenar, esperte y te lo hago"
            )
            recorta_dataset(
                input_dir=directorio_train_base,
                output_dir=directorio_train_cropeado,
                patch_size=tamano_patch,
                overlap_rate=overlap_rate,
                image_start_idx=21,
                image_end_idx=36,
            )

        train_dataset = Offline_Dataset(
            directorio_train_cropeado,
            label_mode=_get_param(params, "label_mode", "vainilla"),
            sigma=_get_param(params, "sigma", 3),
            num_sigmas=_get_param(params, "num_sigmas", 4),
            data_augmentation=_get_param(params, "aumento_datos", default_aumento),
            total_epochs=_get_param(params, "epochs"),
        )

        # val -----------

        directorio_val_base = _get_param(params, "directorio_val_base") or _get_param(
            params, "data_val_path"
        )

        directorio_val_cropeado = f"{directorio_val_base}_{tamano_patch}_{overlap_rate}"

        if not os.path.isdir(directorio_val_cropeado):
            print(
                "no hay el conjunto de datos de patches que quieres para validar, esperte y te lo hago"
            )
            recorta_dataset(
                input_dir=directorio_val_base,
                output_dir=directorio_val_cropeado,
                patch_size=tamano_patch,
                overlap_rate=overlap_rate,
                image_start_idx=37,
                image_end_idx=39,
            )

        val_dataset = Offline_Dataset(
            directorio_val_cropeado,
            label_mode=_get_param(params, "label_mode", "vainilla"),
            sigma=_get_param(params, "sigma", 3),
            num_sigmas=_get_param(params, "num_sigmas", 4),
            data_augmentation=False,
            total_epochs=_get_param(params, "epochs"),
        )

    elif _get_param(params, "dataset") == "rfmid":
        train_dataset = RFMiDDataset(
            data_dir=_get_param(params, "directorio_train_base")
            or _get_param(params, "data_train_path"),
            aumento_datos=_get_param(params, "aumento_datos", default_aumento),
            tamano_patch=_get_param(params, "tamano_patch"),
            total_epochs=_get_param(params, "epochs"),
        )

        val_dataset = RFMiDDataset(
            data_dir=_get_param(params, "directorio_val_base")
            or _get_param(params, "data_val_path"),
            aumento_datos=False,
            tamano_patch=_get_param(params, "tamano_patch"),
            total_epochs=_get_param(params, "epochs"),
        )

    elif _get_param(params, "dataset") == "imagenet":
        train_dataset = ImagenetDataset(
            aumento_datos=_get_param(params, "aumento_datos", default_aumento),
            split="train",
        )

        val_dataset = ImagenetDataset(
            aumento_datos=_get_param(params, "aumento_datos", default_aumento),
            split="validation",
        )

    elif _get_param(params, "dataset") == "demo":
        train_dataset = ImagenetDemoDataset(
            data_dir=_get_param(params, "directorio_train_base", "data/demo")
            or _get_param(params, "data_train_path"),
            tamano_patch=_get_param(params, "tamano_patch"),
        )

        val_dataset = ImagenetDemoDataset(
            data_dir=_get_param(params, "directorio_val_base", "data/demo")
            or _get_param(params, "data_val_path"),
            tamano_patch=_get_param(params, "tamano_patch"),
        )

    else:
        raise NotImplementedError(
            f'La opcion {_get_param(params, "dataset")} no existe, debe ser o "online" (cortar las imagenes bajo demanda) o "offline" (previamente se espera la ejecucion de src.data.crop_script) o "rfmid" o "cifar100"'
        )

    return train_dataset, val_dataset
