import argparse
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
from einops import rearrange

from src.utils import get_device, cargar_config_yaml, load_model, instantiate_dataset


def _clip_for_imshow(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255)
    return np.clip(arr, 0.0, 1.0)


def extract_random_patch(imaxe, patch_size):
    _, h, w = imaxe.shape
    max_h = max(0, h - patch_size)
    max_w = max(0, w - patch_size)
    start_h = random.randint(0, max_h) if max_h > 0 else 0
    start_w = random.randint(0, max_w) if max_w > 0 else 0
    patch = imaxe[:, start_h : start_h + patch_size, start_w : start_w + patch_size]
    return patch


def main():
    parser = argparse.ArgumentParser(
        description="Script para ver como de buenos son los embeddings (y su reconstrucción)."
    )
    parser.add_argument("pesos", type=str, help="Ruta ós pesos do modelo (.pth.tar)")
    parser.add_argument(
        "log_dir", type=str, help="Ruta ó arquivo .log (directorio pai do metadata)"
    )
    parser.add_argument(
        "-imaxes", type=int, default=5, help="Número de imaxes a visualizar"
    )
    parser.add_argument(
        "-k", type=int, default=5, help="Número de filtros a visualizar"
    )
    parser.add_argument(
        "-ganancia", type=float, default=1, help="Ganancia para escalar os filtros"
    )
    parser.add_argument(
        "-C",
        "--cumulativa",
        action="store_true",
        help="Mostrar filtros de maneira acumulativa",
    )

    args = parser.parse_args()

    device = get_device()
    config = cargar_config_yaml(args.pesos, args.log_dir)

    with torch.no_grad():
        modelo = load_model(
            weights_path=args.pesos,
            arch=config["arch"],
            patch_size=config["tamano_patch"],
            token_size=config["tamano_token"],
            order=config.get("order", "first"),
            shared_u=config.get("shared_u", False),
            shared_dict=config.get("shared_dict", False),
            linformer=config["linformer"],
            project_dim=config["project_dim"],
        )
        modelo.to(device)

        # Directly instantiate datasets from config (not using args)
        train_dataset, val_dataset = instantiate_dataset(config=config)

        # Create DataLoader for batched loading
        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )

        W = modelo.to_patch_embedding[
            2
        ].weight  # (dim, h*w*c) esta matriz pasa de parches rgb a embedings
        lista_pca = []
        lista_parche = []
        collected_count = 0

        for batch_imgs, batch_labels in data_loader:
            if collected_count >= args.imaxes:
                break

            batch_imgs = batch_imgs.to(device)

            # Filter for label == 1
            mask = batch_labels == 1
            filtered_imgs = batch_imgs[mask]

            if len(filtered_imgs) == 0:
                continue

            # Process each image in batch
            for imaxe in filtered_imgs:
                if collected_count >= args.imaxes:
                    break

                parche0 = extract_random_patch(imaxe, config["tamano_token"])
                parche0 = rearrange(
                    parche0, "c h w -> h w c", c=3, h=config["tamano_token"]
                )

                # Explicit normalization + detach for consistent scaling
                parche0_cpu = parche0.detach().cpu().numpy()
                parche0_cpu = (
                    parche0_cpu / (parche0_cpu.max() + 1e-8)
                    if parche0_cpu.max() > 1
                    else parche0_cpu
                )
                lista_parche.append(parche0_cpu)

                prod = parche0.ravel() @ W.t()
                values, indices = prod.topk(args.k)

                lista_pca.append(
                    [
                        rearrange(
                            W[idx.item()],
                            "(h w c) -> h w c",
                            c=3,
                            h=config["tamano_token"],
                        )
                        .mul(args.ganancia * val.item())
                        .to("cpu")
                        for idx, val in zip(indices, values)
                    ]
                )
                collected_count += 1

        plot_images_with_filters(
            lista_parche,
            lista_pca,
            args.pesos,
            args.log_dir,
            cumulativa=args.cumulativa,
        )


def plot_images_with_filters(imaxes, pca, pesos_path, logs_dir, cumulativa=False):
    num_images = len(imaxes)
    if num_images == 0:
        return

    num_filters = max(len(filters) for filters in pca) if pca else 0

    fig, axes = plt.subplots(
        num_images,
        num_filters + 1,
        figsize=(5 * (num_filters + 1), 5 * num_images),
    )
    axes = np.atleast_2d(axes)

    for i, imaxe in tqdm(enumerate(imaxes)):
        img = _clip_for_imshow(imaxe)
        axes[i, 0].imshow(img)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Parche {i + 1}")

        cumulative = None
        for j, filter_img in enumerate(pca[i]):
            if cumulativa:
                if cumulative is None:
                    cumulative = filter_img.clone()
                else:
                    cumulative = cumulative + filter_img
                to_plot = cumulative
            else:
                to_plot = filter_img

            to_plot = _clip_for_imshow(to_plot.numpy())

            axes[i, j + 1].imshow(to_plot)
            axes[i, j + 1].axis("off")
            axes[i, j + 1].set_title(f"Filtro {j + 1}")

    fig.tight_layout()
    stem = pesos_path.split("/")[-1].split(".")[0]
    out_path = Path(logs_dir) / "plots" / f"parches_grid_{stem}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
