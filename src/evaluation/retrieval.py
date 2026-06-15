import argparse
from pathlib import Path
import tqdm
import torch
from torchvision.transforms import ToPILImage
import numpy as np
import time
import matplotlib.pyplot as plt

from src.utils import load_model, cargar_config_yaml, get_device
from src.data import ImagenetDataset, RFMiDDataset, Online_Dataset, denormalize

parser = argparse.ArgumentParser(
    prog="Embedding retrieval",
    description="Ver parches característicos de cada dimensión del embedding.",
)
parser.add_argument("pesos_red")
parser.add_argument("directorio")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--capa",
    type=int,
    default=-1,
    help='Capa a visualizar, por defecto es "-1", que es la última. Si se pone a "-2" será la penultima, etc. Si se pone a cero o positivo da error (tenemos que ponernos de acuerdo).',
)
parser.add_argument(
    "--num_imagenes",
    type=int,
    default=150,
    help="Images que se usan para el retrieval (-1 para usar todas)",
)
parser.add_argument(
    "--K",
    type=int,
    default=2,
    help="Número de parches a recuperar por cabeza/dimensión",
)
parser.add_argument(
    "--dims_per_head",
    type=int,
    default=4,
    help="Número de dimensiones a recuperar por cabeza (si se quiere recuperar menos que el total)",
)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--prefetch_factor", type=int, default=2)

args = parser.parse_args()

assert (
    args.capa < 0
), "Vaites! a capa ten que ser negativa (-1 para a última, -2 para a penúltima, etc)"

config = cargar_config_yaml(args.pesos_red, args.directorio)
device = get_device()
modelo = load_model(
    weights_path=args.pesos_red,
    arch=config["arch"],
    patch_size=config["tamano_patch"],
    token_size=config["tamano_token"],
    num_classes=config.get("num_classes", 2),
    order=config.get("order", "first"),
    shared_u=config.get("shared_u", False),
    shared_dict=config.get("shared_dict", False),
).to(device)

num_capas = modelo.transformer.depth
num_cabezas = modelo.transformer.heads
num_tokens = modelo.pos_embedding.shape[1]
dim = int(modelo.transformer.dim / num_cabezas)
tt = config["tamano_token"]
print(f"==> Modelo con {num_capas} capas, {num_cabezas} cabezas y dimensión {dim}.")

if args.num_imagenes == -1:
    args.num_imagenes = None
if config["dataset"] == "imagenet":
    if args.num_imagenes is not None:
        ImagenetDataset.__len__ = lambda self: args.num_imagenes
    dataset = ImagenetDataset(aumento_datos=False, split="train")

elif config["dataset"] == "rfmid":
    if args.num_imagenes is not None:
        RFMiDDataset.__len__ = lambda self: args.num_imagenes
    dataset = RFMiDDataset(
        data_dir=config["directorio_train_base"],
        aumento_datos=False,
        tamano_patch=config["tamano_patch"],
    )

elif config["dataset"] == "online":
    if args.num_imagenes is not None:
        Online_Dataset.__len__ = lambda self: args.num_imagenes
    dataset = Online_Dataset(
        drive_dir=config["directorio_train_base"],
        tamano_patch=config["tamano_patch"],
        aumento_datos=False,
        sobrelapamento=0.9,
    )


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,
    pin_memory=True,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=True,
)

all_values = torch.full(
    (num_cabezas, dim, args.K), -float("inf"), dtype=torch.float
)  # init with -inf for max-k
all_image_idx = torch.zeros(num_cabezas, dim, args.K, dtype=torch.long)
all_token_idx = torch.zeros(num_cabezas, dim, args.K, dtype=torch.long)

# Part 1
with torch.no_grad():
    base_img = 0
    for images, _ in tqdm.tqdm(loader, total=len(loader), unit="batch"):
        b = images.size(0)
        images = images.to(device, non_blocking=True)

        tensor = modelo.get_last_ZU(images, layer=num_capas + args.capa).to(
            "cpu"
        )  # [h, d, b, n], args.capa por defecto es -1
        n_tokens = tensor.size(3)
        # norm = tensor.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # tensor = (tensor / norm).cpu()  # keep remaining ops on CPU to save VRAM

        scores = tensor.reshape(num_cabezas, dim, b * n_tokens)  # [h, d, b*n_tokens]
        flat_idx = (
            torch.arange(b * n_tokens)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(num_cabezas, dim, -1)
        )

        # Recover (image, token) from flat index
        img_idx = (
            torch.div(flat_idx, n_tokens, rounding_mode="floor").long() + base_img
        )  # global image index
        token_idx = flat_idx % n_tokens  # patch/token index

        # Merge with the running top-K: cat, re-sort, keep best K.
        merged_vals = torch.cat([all_values, scores], dim=-1)  # [h, d, K+k_batch]
        merged_img = torch.cat([all_image_idx, img_idx], dim=-1)
        merged_token = torch.cat([all_token_idx, token_idx], dim=-1)

        # Keep top K values
        all_values, order = merged_vals.topk(args.K, dim=-1, largest=True)  # [h, d, K]
        all_image_idx = merged_img.gather(-1, order)
        all_token_idx = merged_token.gather(-1, order)

        base_img += b

print("==> Índices obtenidos")

patches = np.zeros((num_cabezas, dim, args.K, tt, tt, 3), dtype=np.float32)
image_cache = {}


def token_idx_to_patch_coords(token_idx, num_tokens):
    if token_idx == num_tokens - 1:  # TODO: gestionar cls
        return 0, tt, 0, tt  # el parche de la izquierda
    num_patches_per_side = int(np.sqrt(num_tokens - 1))  # -1 for CLS token
    patch_row = token_idx // num_patches_per_side
    patch_col = token_idx % num_patches_per_side

    h_start = patch_row * tt
    h_end = h_start + tt
    w_start = patch_col * tt
    w_end = w_start + tt
    return h_start, h_end, w_start, w_end


print("==> Extrayendo parches...")
for h in range(num_cabezas):
    for d in range(args.dims_per_head):
        for k in range(args.K):
            img_idx = all_image_idx[h, d, k].item()
            token_idx = all_token_idx[h, d, k].item()

            if img_idx not in image_cache:
                if config["dataset"] == "imagenet":
                    img = denormalize(dataset[img_idx][0])
                else:
                    img = dataset[img_idx][0]
                img = ToPILImage()(img)
                image_cache[img_idx] = np.array(img, dtype=np.float32) / 255.0

            h_start, h_end, w_start, w_end = token_idx_to_patch_coords(
                token_idx, num_tokens
            )
            patch = image_cache[img_idx][h_start:h_end, w_start:w_end, :]

            patches[h, d, k] = patch

print("==> Parches almacenados")

n_rows = args.dims_per_head
n_cols = num_cabezas * args.K + (num_cabezas - 1)  # Add margins between heads

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 2), dpi=100)

inicio = time.time()
for d in range(args.dims_per_head):
    for h in range(num_cabezas):
        for k in range(args.K):
            idx_col = h * (args.K + 1) + k  # Account for gap columns
            ax = axes[d, idx_col] if n_rows > 1 else axes[idx_col]
            ax.imshow(patches[h, d, k])
            ax.set_title(f"H{h}D{d}K{k}", fontsize=8)
            ax.axis("off")

    # Hide the gap columns (margins between heads)
    for gap_idx in range(num_cabezas - 1):
        gap_col = (gap_idx + 1) * (args.K + 1) - 1
        ax = axes[d, gap_col] if n_rows > 1 else axes[gap_col]
        ax.axis("off")

plt.tight_layout()
print(f"==> Visualización creada en {time.time() - inicio:.2f} segundos")

plots_dir = Path(args.directorio) / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
outtut_path = f"{plots_dir / Path(args.pesos_red).stem}_retrieval.pdf"

plt.savefig(outtut_path, dpi=150, bbox_inches="tight")
print(f"==> Visualización guardada en '{outtut_path}'")
plt.close()
