from einops import rearrange
import argparse
import numpy as np
import os
from pathlib import Path

import torch

from src.models.coding_rate import CodingRate
from src.plots.metrics import plot_coding_rate, plot_sparsity, cal_sparsity
from src.data.Online_Dataset import Online_Dataset
from src.utils import cargar_config_yaml, load_model, get_device
from src.plots.utils import (
    get_varying_fields,
    config_to_label,
    get_colors,
)


coding_rate_list = []
sparsity_list = []


def forward_hook_codingrate(module, input, output):
    coding_rate_list.append(
        criterion(rearrange(output, "b n (h d) -> b h n d", h=model.transformer.heads))
    )


def forward_hook_sparsity(module, input, output):
    sparsity_list.append(cal_sparsity(output.cpu().numpy(), is_sparse=True))


def _build_dataloader(config, directorio, overlap_rate, batch_size, workers):
    """Instantiate an Online_Dataset DataLoader from a run config."""
    dataset = Online_Dataset(
        directorio,
        tamano_patch=config["tamano_patch"],
        label_mode=config["label_mode"],
        sigma=config["sigma"],
        num_sigmas=config["num_sigmas"],
        aumento_datos=False,
        total_epochs=1,
        sobrelapamento=overlap_rate,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


def _run_inference(model, train_loader):
    """
    Run a full epoch through *model* collecting per-layer coding-rate and
    sparsity statistics.  Returns:
        (means, std_devs, sparsities, std_sparsities)
    where each is a list with one entry per transformer layer.
    """
    global coding_rate_list, sparsity_list

    all_coding_rates = []
    all_sparsities = []

    print(f"  → procesando {len(train_loader)} minibatches …")
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            coding_rate_list = []
            sparsity_list = []

            if isinstance(batch, dict):
                imgs = (
                    batch["imgs"].cuda() if "imgs" in batch else batch["image"].cuda()
                )
            else:
                imgs = (
                    batch[0].cuda()
                    if isinstance(batch, (list, tuple))
                    else batch.cuda()
                )

            model(imgs)

            batch_cr = [(m.item(), s.item()) for m, s in coding_rate_list]
            batch_sp = list(sparsity_list)

            all_coding_rates.append(batch_cr)
            all_sparsities.append(batch_sp)

            print(f"  procesao batch {batch_idx + 1}/{len(train_loader)}", end="\r")

    print()

    # Average over batches for each layer
    n_layers_cr = len(all_coding_rates[0]) if all_coding_rates else 0
    n_layers_sp = len(all_sparsities[0]) if all_sparsities else 0

    means, std_devs = [], []
    for layer_i in range(n_layers_cr):
        layer_means = [
            all_coding_rates[b][layer_i][0] for b in range(len(all_coding_rates))
        ]
        layer_stds = [
            all_coding_rates[b][layer_i][1] for b in range(len(all_coding_rates))
        ]
        means.append(float(np.mean(layer_means)))
        std_devs.append(float(np.mean(layer_stds)))

    sparsities, std_sparsities = [], []
    for layer_i in range(n_layers_sp):
        layer_sp_means = [
            all_sparsities[b][layer_i][0] for b in range(len(all_sparsities))
        ]
        layer_sp_stds = [
            all_sparsities[b][layer_i][1] for b in range(len(all_sparsities))
        ]
        sparsities.append(float(np.mean(layer_sp_means)))
        std_sparsities.append(float(np.mean(layer_sp_stds)))

    return means, std_devs, sparsities, std_sparsities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a .pth.tar checkpoint, or 'all' to evaluate every "
        "checkpoint whose log file lives in logs_dir.",
    )
    parser.add_argument(
        "logs_dir",
        type=str,
        help="Directory that contains the .log / .yaml metadata files.",
        default="data/logs",
    )
    parser.add_argument("--directorio_train_base", type=str, default="data/DRIVE/val/")
    parser.add_argument("--overlap_rate", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    criterion = CodingRate()

    # ── Resolve the list of checkpoints to evaluate ───────────────────────
    if args.checkpoint_path.lower() == "all":
        # Discover every .pth.tar that has a matching log in logs_dir
        log_names = {
            Path(f).stem
            for f in os.listdir(args.logs_dir)
            if f.endswith(".log") and not f.startswith(".")
        }
        checkpoint_paths = sorted(
            [
                str(Path("data/weights") / f"{stem}.pth.tar")
                for stem in log_names
                if (Path("data/weights") / f"{stem}.pth.tar").exists()
            ]
        )
        if not checkpoint_paths:
            print(
                "No matching checkpoints found in data/weights/ for logs in "
                f"{args.logs_dir}. Exiting."
            )
            raise SystemExit(1)
        multi_mode = True
    else:
        checkpoint_paths = [args.checkpoint_path]
        multi_mode = False

    # ── Load configs and determine which fields vary (for legend labels) ──
    configs = {}
    for cp in checkpoint_paths:
        try:
            configs[cp] = cargar_config_yaml(cp, args.logs_dir)
        except (FileNotFoundError, SystemExit):
            print(f"  Warning: could not load config for {cp}, skipping.")
            configs[cp] = None

    valid_checkpoints = [cp for cp in checkpoint_paths if configs[cp] is not None]
    varying_fields = get_varying_fields([configs[cp] for cp in valid_checkpoints])
    palette = get_colors(max(len(valid_checkpoints), 1))

    # ── Run evaluation ────────────────────────────────────────────────────
    all_means, all_std_devs = [], []
    all_sparsities, all_std_sparsities = [], []
    labels, colors = [], []

    for idx, cp in enumerate(valid_checkpoints):
        config = configs[cp]
        print(f"\n[{idx + 1}/{len(valid_checkpoints)}] {cp}")
        print(
            f"  tamano_patch={config.get('tamano_patch')}  "
            f"tamano_token={config.get('tamano_token')}"
        )

        model = load_model(
            weights_path=cp,
            arch=config["arch"],
            patch_size=config["tamano_patch"],
            token_size=config["tamano_token"],
            order=config.get("order", "first"),
            shared_u=config.get("shared_u", False),
            shared_dict=config.get("shared_dict", False),
        )

        model.to(get_device()).eval()

        for layer in model.transformer.layers:
            layer[0].fn.qkv.register_forward_hook(forward_hook_codingrate)
            layer[1].register_forward_hook(forward_hook_sparsity)

        train_loader = _build_dataloader(
            config,
            args.directorio_train_base,
            args.overlap_rate,
            args.batch_size,
            args.workers,
        )

        means, std_devs, sparsities, std_sparsities = _run_inference(
            model, train_loader
        )

        all_means.append(means)
        all_std_devs.append(std_devs)
        all_sparsities.append(sparsities)
        all_std_sparsities.append(std_sparsities)
        labels.append(config_to_label(config, varying_fields))
        colors.append(palette[idx % len(palette)])

    plots_dir = Path(args.logs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if multi_mode:
        log_names = {
            Path(f).stem
            for f in os.listdir(args.logs_dir)
            if f.endswith(".log") and not f.startswith(".")
        }
        name = plots_dir / f"{'_'.join(log_names)}"

        plot_coding_rate(
            means=all_means,
            std_devs=all_std_devs,
            name=name,
            labels=labels,
            colors=colors,
            legend=True,
        )

        plot_sparsity(
            sparsities=all_sparsities,
            std_sparsities=all_std_sparsities,
            name=name,
            labels=labels,
            colors=colors,
            legend=True,
        )

    else:
        name = Path(plots_dir / args.checkpoint_path)
        cp = valid_checkpoints[0]
        name = cp.replace("data/weights", str(plots_dir)).replace(".pth.tar", "")
        plot_coding_rate([all_means[0]], [all_std_devs[0]], name)
        plot_sparsity([all_sparsities[0]], [all_std_sparsities[0]], name)

    print("\nlisto :)")
