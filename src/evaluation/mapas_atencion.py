import argparse
import numpy as np
from pathlib import Path

import torch

from src.data.Online_Dataset import Online_Dataset
from src.utils import cargar_config_yaml, load_model, get_device
from src.plots.plot_mapas_atencion import plot_mapas_atencion


def cargar_imaxes(dataset_path, tamano_patch, num_images):
    """Cargar imaxes do dataset con metade positivas e metade negativas."""
    dataset = Online_Dataset(
        drive_dir=dataset_path,
        tamano_patch=tamano_patch,
        aumento_datos=False,
        sobrelapamento=0.8,
    )

    imaxes = []
    etiquetas = []
    num_por_clase = num_images // 2
    counts = {1: 0, 0: 0}
    indices = np.random.permutation(len(dataset))
    i = 0

    while counts[1] < num_por_clase or counts[0] < num_por_clase:
        # print(f'len(datset): {len(dataset)}, i: {i}, indices[i]: {indices[i]}')
        img, label = dataset[int(indices[i])]
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


def obter_mapas_atencion_cls(
    modelo, imaxes, indices_capas, num_heads, tamano_patch, tamano_token, resolution=1
):
    """
    Extract fine-grained attention maps from the model's transformer layers.

    Args:
        modelo: The transformer model.
        imaxes: Input images tensor of shape [B, C, H, W].
        indices_capas: List of layer indices to extract attention from.
        num_heads: Number of attention heads in the model.
        tamano_patch: Patch size used in the model.
        tamano_token: Token size used in the model.
        resolution: Resolution factor for attention maps.

    Returns:
        A dictionary where keys are layer indices (e.g., 'layer.0') and values are
        tensors of shape [B, H, N], representing the attention maps for each image (B),
        head (H), and token (N).
    """
    stride = tamano_token // resolution
    num_patches = tamano_patch // tamano_token
    fine_grained_size = num_patches * resolution

    qkv_values = {}

    def make_qkv_hook(layer_idx):
        def hook(module, input, output):
            qkv_values[f"layer.{layer_idx}"] = output.detach()  # [B, N, inner_dim]

        return hook

    hooks = []
    for layer_idx in indices_capas:
        try:
            attn_module = modelo.transformer.layers[layer_idx][0].fn
            hook = attn_module.qkv.register_forward_hook(make_qkv_hook(layer_idx))
            hooks.append(hook)
        except (AttributeError, IndexError) as e:
            print(f"Aviso: {e}")

    fine_grained_attention = {}

    with torch.no_grad():
        for img_idx, img in enumerate(imaxes):
            fine_grained_attention[img_idx] = {}
            for di in range(resolution):
                for dj in range(resolution):
                    start_i, start_j = di * stride, dj * stride
                    subimage = img[
                        :,
                        start_i : start_i + tamano_patch,
                        start_j : start_j + tamano_patch,
                    ]
                    subimage = subimage.unsqueeze(0)  # Add batch dimension

                    _ = modelo(subimage)

                    for layer_idx in indices_capas:
                        key = f"layer.{layer_idx}"
                        if key in qkv_values:
                            qkv = qkv_values[key]  # [B, N, inner_dim]
                            B, N, inner_dim = qkv.shape
                            D_h = inner_dim // num_heads

                            # Reshape to heads: [B, N, H, D_h] -> [B, H, N, D_h]
                            q = qkv.reshape(B, N, num_heads, D_h).permute(0, 2, 1, 3)

                            # In CRATE, K = Q (symmetric), so dots are q @ q^T
                            cls = q[:, :, 0:1, :]  # [B, H, 1, D_h]
                            scores = (q @ cls.transpose(-2, -1)).squeeze(-1) * (
                                D_h**-0.5
                            )  # [B, H, N]

                            # Fill fine-grained grid
                            if key not in fine_grained_attention[img_idx]:
                                fine_grained_attention[img_idx][key] = torch.zeros(
                                    (num_heads, fine_grained_size, fine_grained_size)
                                )

                            fine_grained_attention[img_idx][key][
                                :, di::resolution, dj::resolution
                            ] = scores[:, :, 1:].reshape(
                                num_heads, num_patches, num_patches
                            )

    for hook in hooks:
        hook.remove()

    return fine_grained_attention


def obter_mapas_atencion(modelo, imaxes, indices_capas, num_heads):
    """Extraer mapas de atención das capas especificadas."""
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[f"layer.{layer_idx}"] = output.detach()

        return hook

    # Rexistrar hooks
    hooks = []
    for layer_idx in indices_capas:
        layer = modelo.transformer.layers[layer_idx]
        hook = layer[0].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        _ = modelo(imaxes)

    # Eliminar hooks
    for hook in hooks:
        hook.remove()

    # Procesar activacións para obter saídas das cabezas de atención
    results = {}
    for layer_idx in indices_capas:
        key = f"layer.{layer_idx}"
        if key in activations:
            act = activations[key]  # [B, N, D]
            B, N, D = act.shape

            # Reshape to separate heads: [B, N, D] -> [B, N, H, D_h] -> [B, H, N, D_h]
            act = act.reshape(B, N, num_heads, D // num_heads)
            act = act.permute(0, 2, 1, 3)  # [B, H, N, D_h]

            # Store the attention maps for this layer
            results[key] = act

    # Debug: Validate output structure
    for layer, tensor in results.items():
        print(f"Layer {layer}: Shape {tensor.shape}")

    return results


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
        default=1,
        help="Número de últimas capas a visualizar",
    )
    parser.add_argument(
        "-imaxes", type=int, default=2, help="Número de imaxes a visualizar"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cls",
        choices=["vainilla", "cls"],
        help="Modo de extracción: vainilla (saídas de camadas) o cls (atención desde token CLS)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=-1,
        help="Resolución para extracción de mapas de atención (1 = fina, 2 = media, etc.)",
    )

    args = parser.parse_args()

    # Cargar configuración dende YAML
    config = cargar_config_yaml(args.checkpoint, args.logs_dir)
    tamano_patch = config["tamano_patch"]
    tamano_token = config["tamano_token"]

    if args.resolution == -1:
        args.resolution = tamano_token

    device = get_device()
    modelo = load_model(
        weights_path=args.checkpoint,
        arch=config["arch"],
        patch_size=config["tamano_patch"],
        token_size=config["tamano_token"],
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
        dataset_path="data/DRIVE/val",
        tamano_patch=tamano_patch + tamano_patch,
        num_images=args.imaxes,
    )
    imaxes = imaxes.to(device)

    # Use 'cls' mode by default for proper attention matrix visualization
    if args.mode == "cls":
        mapas_atencion = obter_mapas_atencion_cls(
            modelo,
            imaxes,
            indices_capas,
            args.cabezas,
            tamano_patch,
            tamano_token,
            args.resolution,
        )
    elif args.mode == "vainilla":
        mapas_atencion = obter_mapas_atencion(
            modelo,
            imaxes,
            indices_capas,
            args.cabezas,
            tamano_patch,
            tamano_token,
            args.resolution,
        )
    else:
        raise NotImplementedError(
            f"O modo debe ser cls ou vainilla, dechesme {args.mode}"
        )
    plots_dir = Path(args.logs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = f"{plots_dir / Path(args.checkpoint).stem}_atencion.png"

    offset = (0, tamano_patch + tamano_token)

    plot_mapas_atencion(
        imaxes=imaxes.cpu(),
        mapas_atencion=mapas_atencion,
        indices_capas=indices_capas,
        num_cabezas=args.cabezas,
        output_path=output_path,
        indices_cabezas_por_capa=indices_cabezas_por_capa,
        offset=offset,
        etiquetas=etiquetas,
    )


if __name__ == "__main__":
    main()
