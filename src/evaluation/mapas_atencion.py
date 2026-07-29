import argparse
import numpy as np
from pathlib import Path

import torch

from src.data import (
    Online_Dataset,
    RFMiDDataset,
    ImagenetDemoDataset,
    ImagenetDataset,
    denormalize,
)
from src.utils import cargar_config_yaml, load_model, get_device
from src.plots.plot_mapas_atencion import plot_mapas_atencion


def cargar_imaxes(
    dataset_path,
    tamano_patch,
    num_images,
    dataset_type,
    overlap_rate,
    label_mode,
    sigma,
    num_sigmas,
):
    """Cargar imaxes do dataset con metade positivas e metade negativas."""
    # Instantiate dataset based on type
    if dataset_type == "online":
        dataset = Online_Dataset(
            drive_dir=dataset_path,
            tamano_patch=tamano_patch,
            label_mode=label_mode,
            sigma=sigma,
            num_sigmas=num_sigmas,
            aumento_datos=False,
            sobrelapamento=overlap_rate,
        )
    elif dataset_type == "rfmid":
        dataset = RFMiDDataset(
            data_dir=dataset_path,
            aumento_datos=False,
            tamano_patch=tamano_patch,
        )
    elif dataset_type == "imagenet":
        dataset = ImagenetDataset(aumento_datos=False, split="test")
        np.random.seed(42)
        indices = np.random.permutation(len(dataset))[:num_images]
        imaxes = []
        etiquetas = []
        for idx in indices:
            img, label = dataset[int(idx)]
            imaxes.append(denormalize(img))
            etiquetas.append(label)
        imaxes = torch.stack(imaxes)
        return imaxes, etiquetas

    elif dataset_type == "demo":
        dataset = ImagenetDemoDataset(
            data_dir="data/demo/insectos/",
            tamano_patch=224,
            virtual_length=num_images,
            cache_images=True,
        )
        imaxes = []
        etiquetas = []
        for i in range(num_images):
            img, label = dataset[i]
            imaxes.append(img)
            etiquetas.append(label)
        imaxes = torch.stack(imaxes)
        return imaxes, etiquetas

    else:
        raise ValueError(f"Descoñecido dataset type: {dataset_type}")

    imaxes = []
    etiquetas = []
    num_por_clase = num_images // 2
    counts = {1: 0, 0: 0}
    indices = np.random.permutation(len(dataset))
    i = 0

    while counts[1] < num_por_clase or counts[0] < num_por_clase:
        # print(f'len(datset): {len(dataset)}, i: {i}, indices[i]: {indices[i]}')
        img, label = dataset[int(indices[i])]
        label = int(label)  # Convert to int in case it's a numpy array
        if counts[label] < num_por_clase:
            imaxes.append(img)
            etiquetas.append(label)
            counts[label] += 1
        i += 1
        if i == len(dataset):  # hemos acabao con esos indices
            indices = np.random.permutation(len(dataset))
            i = 0

    sorted_indices = np.argsort(etiquetas)[::-1]  # descending
    imaxes = torch.stack([imaxes[i] for i in sorted_indices])
    etiquetas = [etiquetas[i] for i in sorted_indices]

    # print(f"Cargadas {len(imaxes)} imaxes ({counts[1]} positivas, {counts[0]} negativas)")
    return imaxes, etiquetas


def obter_mapas_atencion(
    modelo, imaxes, indices_capas, num_heads, tamano_patch, tamano_token, resolution=1
):
    """
    Extract direct attention matrices from CRATE model layers with high-resolution
    sliding-window trick (same as obter_mapas_atencion_cls).

    Args:
        modelo: The CRATE model with get_last_selfattention method.
        imaxes: Input images tensor of shape [B, C, H, W].
        indices_capas: List of layer indices to extract attention from.
        num_heads: Number of attention heads (for consistency).
        tamano_patch: Patch size used in the model (crop size fed to the model).
        tamano_token: Token size used in the model (stride = tamano_patch // num_patches).
        resolution: Resolution multiplier. 1 = standard, >1 = finer grid via sliding window.

    Returns:
        A dictionary where keys are image indices, and values are dicts with:
        - layer keys (e.g., 'layer.0'): attention tensor [H, G, G] for first-order,
          or tuple ([H, G, G], [H, G, G]) for second-order, where G = num_patches * resolution.
    """
    stride = tamano_token // resolution
    num_patches = tamano_patch // tamano_token
    fine_grained_size = num_patches * resolution

    attention_maps = {}

    with torch.no_grad():
        for img_idx, img in enumerate(imaxes):
            attention_maps[img_idx] = {}

            # Pad image so all (di, dj) shift positions yield a valid tamano_patch crop
            max_position = (resolution - 1) * stride
            required_size = max_position + tamano_patch
            current_size = img.shape[-1]  # assuming square [C, H, W]

            if required_size > current_size:
                pad_amount = required_size - current_size
                img_padded = torch.nn.functional.pad(
                    img, (0, pad_amount, 0, pad_amount)
                )
            else:
                img_padded = img

            # Accumulators: built lazily on first layer/crop encounter
            acc_first = {}  # layer_key -> [H, G, G]
            acc_second = {}  # layer_key -> [H, G, G]  (second-order only)

            for di in range(resolution):
                for dj in range(resolution):
                    start_i = di * stride
                    start_j = dj * stride
                    crop = img_padded[
                        :,
                        start_i : start_i + tamano_patch,
                        start_j : start_j + tamano_patch,
                    ].unsqueeze(0)  # [1, C, tamano_patch, tamano_patch]

                    for layer_idx in indices_capas:
                        layer_key = f"layer.{layer_idx}"
                        try:
                            attn = modelo.get_last_selfattention(
                                crop, layer=layer_idx, return_both_attentions=True
                            )

                            if modelo.transformer.order == "first":
                                # attn: [1, H, N, N]
                                nh = attn.shape[1]
                                # CLS row, patch columns -> [H, P, P]
                                patch_attn = attn[0, :, 0, 1:].reshape(
                                    nh, num_patches, num_patches
                                )

                                if layer_key not in acc_first:
                                    acc_first[layer_key] = torch.zeros(
                                        nh,
                                        fine_grained_size,
                                        fine_grained_size,
                                        device=attn.device,
                                        dtype=attn.dtype,
                                    )
                                # Interleave into fine-grained grid at offset (di, dj)
                                acc_first[layer_key][
                                    :, di::resolution, dj::resolution
                                ] = patch_attn

                            elif modelo.transformer.order == "second":
                                # attn = (attn1, attn2), each [1, H, N, N]
                                attn1, attn2 = attn

                                nh1 = attn1.shape[1]
                                patch_attn1 = attn1[0, :, 0, 1:].reshape(
                                    nh1, num_patches, num_patches
                                )

                                nh2 = attn2.shape[1]
                                patch_attn2 = attn2[0, :, 0, 1:].reshape(
                                    nh2, num_patches, num_patches
                                )

                                if layer_key not in acc_first:
                                    acc_first[layer_key] = torch.zeros(
                                        nh1,
                                        fine_grained_size,
                                        fine_grained_size,
                                        device=attn1.device,
                                        dtype=attn1.dtype,
                                    )
                                    acc_second[layer_key] = torch.zeros(
                                        nh2,
                                        fine_grained_size,
                                        fine_grained_size,
                                        device=attn2.device,
                                        dtype=attn2.dtype,
                                    )

                                acc_first[layer_key][
                                    :, di::resolution, dj::resolution
                                ] = patch_attn1
                                acc_second[layer_key][
                                    :, di::resolution, dj::resolution
                                ] = patch_attn2

                        except Exception as e:
                            print(
                                f"Error extracting attention for image {img_idx}, "
                                f"layer {layer_idx}, shift ({di},{dj}): {e}"
                            )

            # Store final accumulated maps
            for layer_key in acc_first:
                if modelo.transformer.order == "first":
                    attention_maps[img_idx][layer_key] = acc_first[layer_key]
                else:
                    attention_maps[img_idx][layer_key] = (
                        acc_first[layer_key],
                        acc_second[layer_key],
                    )

    return attention_maps


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar cabezas de atención de CRATE"
    )

    parser.add_argument("checkpoint", type=str, help="Ruta ao checkpoint")
    parser.add_argument(
        "logs_dir",
        type=str,
        default="data/runs/",
        help="Path to the metadata (e.g. data/runs/)",
    )
    parser.add_argument(
        "-cabezas", type=int, default=-1, help="Número de cabezas a visualizar por capa"
    )
    parser.add_argument(
        "-capas",
        "--num-last-layers",
        type=int,
        default=-1,
        help="Número de últimas capas a visualizar (use -1 para todas as capas)",
    )
    parser.add_argument(
        "-imaxes", type=int, default=12, help="Número de imaxes a visualizar"
    )
    parser.add_argument("-demo", action="store_true", help="Usar dataset imagenet demo")
    parser.add_argument(
        "--resolution",
        type=int,
        default=-1,
        help="Resolución para extracción de mapas de atención (mas grande es mas resolucion, maximo es -1)",
    )

    args = parser.parse_args()

    # Cargar configuración dende YAML
    config = cargar_config_yaml(args.checkpoint, args.logs_dir)
    tamano_patch = config["tamano_patch"]
    tamano_token = config["tamano_token"]
    dataset_type = config.get("dataset", "online")
    overlap_rate = config.get("overlap_rate", 0.1)
    label_mode = config.get("label_mode", "vainilla")
    sigma = config.get("sigma", 3)
    num_sigmas = config.get("num_sigmas", 4)
    directorio_val_base = config.get("directorio_val_base", "data/DRIVE/val")

    print(config)

    if args.resolution == -1:
        args.resolution = tamano_token

    device = get_device()
    modelo = load_model(
        weights_path=args.checkpoint,
        arch=config["arch"],
        patch_size=config["tamano_patch"],
        token_size=config["tamano_token"],
        num_classes=config.get("num_classes", 2),
        order=config.get("order", "first"),
        shared_u=config.get("shared_u", False),
        shared_dict=config.get("shared_dict", False),
        linformer=config["linformer"],
        project_dim=config["project_dim"],
    )

    modelo = modelo.to(device)
    modelo.eval()

    capas_modelo = modelo.transformer.depth
    cabezas_modelo = modelo.transformer.heads

    # Determinar capas a visualizar
    indices_capas = []

    if args.num_last_layers == -1:
        # Visualizar todas as capas
        indices_capas = list(range(capas_modelo))
    else:
        indices_capas.append(0)
        # Engadir últimas n capas
        for i in range(args.num_last_layers):
            layer_idx = capas_modelo - 1 - i
            if layer_idx not in indices_capas and layer_idx >= 0:
                indices_capas.append(layer_idx)
        indices_capas.sort()
    print(f"Vanse a visualizar as capas: {indices_capas}")

    args.cabezas = cabezas_modelo if args.cabezas == int(-1) else args.cabezas
    print(f"\nPre-seleccionando {args.cabezas} cabezas por capa...")
    indices_cabezas_por_capa = {}
    for layer_idx in indices_capas:
        selected = np.random.choice(cabezas_modelo, args.cabezas, replace=False)
        indices_cabezas_por_capa[layer_idx] = sorted(selected.tolist())

    imaxes, etiquetas = cargar_imaxes(
        dataset_path=directorio_val_base,
        tamano_patch=tamano_patch,
        num_images=args.imaxes,
        dataset_type=dataset_type if not args.demo else "demo",
        overlap_rate=overlap_rate,
        label_mode=label_mode,
        sigma=sigma,
        num_sigmas=num_sigmas,
    )
    imaxes = imaxes.to(device)

    mapas_atencion = obter_mapas_atencion(
        modelo,
        imaxes,
        indices_capas,
        args.cabezas,
        tamano_patch,
        tamano_token,
        args.resolution,
    )

    plots_dir = Path(args.logs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        f"{plots_dir / Path(args.checkpoint).stem}_atencion_r{args.resolution}.pdf"
    )

    # Crop images to match the analyzed region (only for cls mode)
    stride = tamano_token // args.resolution
    max_position = (args.resolution - 1) * stride
    analyzed_size = max_position + tamano_patch
    imaxes_cropped = imaxes[:, :, :analyzed_size, :analyzed_size]

    # Compute logits for each image
    logits = {}
    with torch.no_grad():
        for img_idx, img in enumerate(imaxes):
            img_batch = img.unsqueeze(0)  # Add batch dimension
            output = modelo(img_batch)
            # Extract logits (output could be (logits, other_outputs) or just logits)
            if isinstance(output, tuple):
                output = output[0]
            logits[img_idx] = output.squeeze(0).cpu()  # Remove batch dimension

    plot_mapas_atencion(
        imaxes=imaxes_cropped.cpu(),
        mapas_atencion=mapas_atencion,
        indices_capas=indices_capas,
        num_cabezas=args.cabezas,
        orden=config["order"],
        output_path=output_path,
        indices_cabezas_por_capa=indices_cabezas_por_capa,
        offset=(0, tamano_token),
        etiquetas=etiquetas,
        logits=logits,
    )


if __name__ == "__main__":
    main()
