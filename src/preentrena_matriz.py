# este escript preentrena a matriz do embedding
# que logo podrase usar no main.py con --pretrain_embedding

import argparse
import csv
from pathlib import Path
from uuid import uuid4
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from einops.layers.torch import Rearrange

from src.utils import instantiate_dataset, get_device, init_yaml
from src.data import ImageGroupedSampler


parser = argparse.ArgumentParser(
    description="script de preadestramento da matriz do embeding",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
# matriz
parser.add_argument(
    "-tp",
    "--tamano_patch",
    default=48,
    type=int,
    help="la subimagen que se recorta de la grande",
)
parser.add_argument(
    "-tt", "--tamano_token", default=16, type=int, help="el token del ViT"
)
parser.add_argument("-dim", default=192, type=int, help="A dimensión do embeddin")
# adestramento
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-e",
    "--epochs",
    default=5000,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "-b",
    "--batch_size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    default=0.00005,
    type=float,
    metavar="LR",
    help="initial learning rate (default 0.005)",
    dest="lr",
)
parser.add_argument(
    "--prefetch_factor",
    default=2,
    type=int,
    help="Number of samples prefetched by each worker (default: 2)",
)
# dataset
parser.add_argument(
    "-t_dir",
    "--directorio_train_base",
    default="data/DRIVE/train",
    type=str,
    help="directorio de las imagenes de train",
)
parser.add_argument(
    "-v_dir",
    "--directorio_val_base",
    default="data/DRIVE/val",
    type=str,
    help="directorio de las imagenes de val",
)
parser.add_argument(
    "-weights_dir",
    default="data/weights",
    type=str,
    help="a que directorio se van los pesos",
)
parser.add_argument(
    "--dataset",
    default="online",
    type=str,
    help='Dataset "online" (defecto), "offline" o "rfmid"',
)
# regularizacion
parser.add_argument(
    "--aumento_datos", action="store_true", help="Usar aumento de datos"
)
parser.add_argument(
    "--embedding_l1_weight",
    default=1.0,
    type=float,
    help="Peso da regularizacion L1 para a matriz de embedding",
)
parser.add_argument(
    "--embedding_l0_weight",
    default=1.0,
    type=float,
    help="Peso da regularizacion L0 para a matriz de embedding",
)
parser.add_argument(
    "--embedding_orthogonal_weight",
    default=1.0,
    type=float,
    help="Peso da regularizacion ortogonal para a matriz de embedding",
)
parser.add_argument(
    "--embedding_reconstruction_weight",
    default=1.0,
    type=float,
    help="Peso da regularizacion de reconstruccion para a matriz de embedding",
)

args = parser.parse_args()

weights_root = Path(args.weights_dir) / "Linear"
weights_root.mkdir(parents=True, exist_ok=True)

while True:
    run_id = uuid4().hex[:6]
    run_dir = weights_root / run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        break

args.runs_dir = run_dir
init_yaml(run_id, args)
weights_path = run_dir / f"{run_id}.pth"
losses_csv_path = run_dir / "loses.csv"

device = get_device()

# a matriz de marras é de forma: (in, out) , o sea (tt*tt*3, dim)

Linear = nn.Linear(3 * args.tamano_token**2, args.dim).to(device)
W, b = Linear.weight, Linear.bias

imaxes_a_parches = Rearrange(
    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=args.tamano_token, p2=args.tamano_token
)

# --

optimizer = Adam(Linear.parameters(), lr=args.lr)

train_dataset, val_dataset = instantiate_dataset(args)

train_sampler = None
if args.dataset == "online":
    train_sampler = ImageGroupedSampler(train_dataset, shuffle=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    num_workers=args.workers,
    pin_memory=True,
    prefetch_factor=args.prefetch_factor,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    persistent_workers=True,
)

# --
with losses_csv_path.open("w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "loss", "l1", "l0", "ortogonal", "reconstruccion"])

    for i in range(args.epochs):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0
        epoch_ortogonal = 0.0
        epoch_reconstruccion = 0.0
        num_batches = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)

            # l1
            l1 = args.embedding_l1_weight * (torch.norm(W, p=1) + torch.norm(b, p=1))
            # l0
            l0 = args.embedding_l0_weight * (torch.norm(W, p=0) + torch.norm(b, p=0))
            # ortogonal
            Ish = W.t() @ W
            ortogonal = args.embedding_orthogonal_weight * torch.norm(
                torch.eye(W.shape[1], device=W.device) - Ish, p="fro"
            )
            # reconstruccion
            parches = imaxes_a_parches(images)
            embeddings = Linear(parches)
            embeddings = embeddings - b.view(1, 1, -1)
            parches_ish = torch.matmul(embeddings, W)
            reconstruccion = args.embedding_reconstruction_weight * torch.norm(
                parches - parches_ish, p="fro"
            )

            loss = l1 + l0 + ortogonal + reconstruccion

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_l1 += l1.item()
            epoch_l0 += l0.item()
            epoch_ortogonal += ortogonal.item()
            epoch_reconstruccion += reconstruccion.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError(
                "train_loader devolvio 0 batches; no se puede calcular la media por epoca"
            )

        mean_loss = epoch_loss / num_batches
        mean_l1 = epoch_l1 / num_batches
        mean_l0 = epoch_l0 / num_batches
        mean_ortogonal = epoch_ortogonal / num_batches
        mean_reconstruccion = epoch_reconstruccion / num_batches

        writer.writerow(
            [i, mean_loss, mean_l1, mean_l0, mean_ortogonal, mean_reconstruccion]
        )
        csv_file.flush()

        print(
            f"epoca {i}, loss {mean_loss:.4f} l1 {mean_l1:.4f} l0 {mean_l0:.4f} "
            f"ortogonal {mean_ortogonal:.4f} reconstruccion {mean_reconstruccion:.4f}"
        )

torch.save(Linear.state_dict(), weights_path)
print(f"run_id: {run_id}")
print(f"pesos guardados en: {weights_path}")
print(f"losses guardadas en: {losses_csv_path}")


os._exit(0)
