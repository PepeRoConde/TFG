import argparse
import random
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import torch
from einops import rearrange

from src.data.Online_Dataset import Online_Dataset
from src.utils import get_device, cargar_config_yaml, load_model


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
        "-ganancia", type=int, default=0, help="Ganancia para escalar os filtros"
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
        )
        modelo.to(device)

        dataset = Online_Dataset(
            "data/DRIVE/train", config["tamano_patch"], aumento_datos=False
        )

        sampled_indices = random.sample(range(len(dataset)), args.imaxes)
        imaxes = [dataset[i][0] for i in sampled_indices]

        W = modelo.to_patch_embedding[
            2
        ].weight  # (dim, h*w*c) esta matriz pasa de parches rgb a embedings
        # b = modelo.to_patch_embedding[2].bias # (dim,)

        lista_pca = []
        lista_parche = []

        for imaxe in imaxes:
            imaxe = imaxe.to(device)

            parche0 = imaxe[:, : config["tamano_token"], : config["tamano_token"]]
            print(f"{imaxe.shape} {parche0.shape}")
            lista_parche.append(
                rearrange(parche0, "c h w -> h w c", c=3, h=config["tamano_token"]).to(
                    "cpu"
                )
            )
            embeddings = rearrange(
                imaxe,
                "c (h p1) (w p2) -> (p1 p2) (h w c)",
                c=3,
                h=config["tamano_token"],
                w=config["tamano_token"],
            )
            # tiene shape (#tokens, dim), donde cada token es un embedding

            embedding0 = embeddings[0]  # nos quedamos con un token (dim,)

            prod = embedding0 @ W.t()
            values, indices = prod.topk(args.k)
            print(f"{prod.shape} {embedding0.shape}  {W.shape}")

            # Scale filters by their corresponding top-k activation value
            lista_pca.append(
                [
                    rearrange(
                        W[idx.item()], "(h w c) -> h w c", c=3, h=config["tamano_token"]
                    )
                    .mul(args.ganancia * val.item())
                    .to("cpu")
                    for idx, val in zip(indices, values)
                ]
            )

            print(f"{len(lista_parche)} ")

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

    # Each entry in `pca` is the list of top-k filters for the corresponding image
    num_filters = max(len(filters) for filters in pca) if pca else 0

    fig, axes = plt.subplots(
        num_images,
        num_filters + 1,
        figsize=(5 * (num_filters + 1), 5 * num_images),
    )
    axes = np.atleast_2d(axes)

    for i, imaxe in enumerate(imaxes):
        img = imaxe

        axes[i, 0].imshow(img.numpy())
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Image {i+1} (Original)")

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

            axes[i, j + 1].imshow(to_plot.numpy())
            axes[i, j + 1].axis("off")
            axes[i, j + 1].set_title(f"Filter {j+1}")

    fig.tight_layout()
    stem = pesos_path.split("/")[-1].split(".")[0]
    out_path = Path(logs_dir) / "plots" / f"parches_grid_{stem}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
