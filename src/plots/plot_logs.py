import os
import matplotlib.pyplot as plt
import re
import yaml
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch

from src.utils import cargar_config_yaml, CSVLogger
from src.plots.utils import (
    get_varying_fields,
    config_to_label,
    get_marker_and_linewidth,
    get_colors,
)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_with_sigma(values, window=250):
    """
    Centered rolling mean ± std over *window* samples.
    Returns (mean, mean-std, mean+std) as equal-length numpy arrays.
    """
    values = np.array(values, dtype=float)
    n = len(values)
    half = window // 2
    mean = np.empty(n)
    lower = np.empty(n)
    upper = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        patch = values[lo:hi]
        m = patch.mean()
        s = patch.std()
        mean[i] = m
        lower[i] = m - s
        upper[i] = m + s
    return mean, lower, upper


def plot_sombra(
    ax,
    epochs,
    values,
    color,
    linestyle,
    alpha_line,
    alpha_fill,
    label=None,
    linewidth=1.5,
):
    """
    Plot a smoothed mean line with ±1σ flanking lines and a shaded band.

    alpha_line : opacity of the mean line
    alpha_fill : opacity of the shadow band (0.05 for train, 0.15 for val)
    """
    epochs = np.array(epochs)
    mean, lower, upper = smooth_with_sigma(np.array(values, dtype=float))

    ax.fill_between(epochs, lower, upper, color=color, alpha=alpha_fill, linewidth=0)
    ax.plot(
        epochs,
        lower,
        color=color,
        alpha=alpha_line * 0.5,
        linestyle=linestyle,
        linewidth=linewidth * 0.5,
    )
    ax.plot(
        epochs,
        upper,
        color=color,
        alpha=alpha_line * 0.5,
        linestyle=linestyle,
        linewidth=linewidth * 0.5,
    )

    kwargs = dict(
        color=color, alpha=alpha_line, linestyle=linestyle, linewidth=linewidth
    )
    if label is not None:
        kwargs["label"] = label
    ax.plot(epochs, mean, **kwargs)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------


def plot_logs(log_dir="data/runs", output_file="data/plots", modo="sombra"):
    """
    Read all log files and create loss / accuracy / AUC-ROC plots.

    modo='vainilla' — original scatter/line plot per run, no smoothing.
    modo='sombra'   — smoothed trend line with shaded ±1σ error band.
    """
    log_files = [
        f for f in os.listdir(log_dir) if not f.startswith(".") and f.endswith(".log")
    ]

    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    output_dir = Path(log_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_names = "_".join([f.replace(".log", "") for f in sorted(log_files)])
    output_filename = output_dir / f"{checkpoint_names}.png"

    # ── First pass: load configs to find varying fields ──────────────────
    log_file_configs = {}
    for log_file in log_files:
        base_name = log_file.replace(".log", "")
        checkpoint_path = f"data/weights/{base_name}.pth.tar"
        try:
            config = cargar_config_yaml(checkpoint_path, log_dir)
            log_file_configs[log_file] = config
        except (FileNotFoundError, SystemExit):
            print(f"Skipping {log_file} - couldn't load config")
            log_file_configs[log_file] = None

    varying_fields = get_varying_fields(log_file_configs.values())
    print(f"\nVarying fields across runs: {varying_fields}\n")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    valid_log_files = [f for f in log_files if log_file_configs.get(f) is not None]
    palette = get_colors(max(len(valid_log_files), 1))

    config_colors = {}
    config_labels_shown = {}
    color_idx = 0

    for log_file in log_files:
        config = log_file_configs.get(log_file)
        if config is None:
            continue

        filepath = os.path.join(log_dir, log_file)
        try:
            csv_logger = CSVLogger(filepath)
            rows = csv_logger.read()
            if not rows:
                print(f"Empty log file: {log_file}")
                continue
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue

        marker, linewidth = get_marker_and_linewidth(config, varying_fields)

        config_key = str(sorted(config.items()))
        if config_key not in config_colors:
            config_colors[config_key] = palette[color_idx % len(palette)]
            color_idx += 1
        color = config_colors[config_key]

        label = config_to_label(config, varying_fields)
        show_legend = config_key not in config_labels_shown
        config_labels_shown[config_key] = True

        try:
            epochs = [float(row["epoch"]) for row in rows]

            if modo == "sombra":
                if "loss" in rows[0]:
                    plot_sombra(
                        ax1,
                        epochs,
                        [float(r["loss"]) for r in rows],
                        color=color,
                        linestyle="-",
                        alpha_line=0.9,
                        alpha_fill=0.05,
                        linewidth=linewidth,
                    )
                if "val_loss" in rows[0]:
                    plot_sombra(
                        ax1,
                        epochs,
                        [float(r["val_loss"]) for r in rows],
                        color=color,
                        linestyle="--",
                        alpha_line=0.9,
                        alpha_fill=0.15,
                        linewidth=linewidth,
                    )

                if "train_accuracy" in rows[0]:
                    plot_sombra(
                        ax2,
                        epochs,
                        [float(r["train_accuracy"]) for r in rows],
                        color=color,
                        linestyle="-",
                        alpha_line=0.9,
                        alpha_fill=0.05,
                        linewidth=linewidth,
                    )
                if "val_accuracy" in rows[0]:
                    plot_sombra(
                        ax2,
                        epochs,
                        [float(r["val_accuracy"]) for r in rows],
                        color=color,
                        linestyle="--",
                        alpha_line=0.9,
                        alpha_fill=0.15,
                        linewidth=linewidth,
                    )

                if "train_auc" in rows[0]:
                    plot_sombra(
                        ax3,
                        epochs,
                        [float(r["train_auc"]) for r in rows],
                        color=color,
                        linestyle="-",
                        alpha_line=0.9,
                        alpha_fill=0.05,
                        label=label if show_legend else None,
                        linewidth=linewidth,
                    )
                if "val_auc" in rows[0]:
                    plot_sombra(
                        ax3,
                        epochs,
                        [float(r["val_auc"]) for r in rows],
                        color=color,
                        linestyle="--",
                        alpha_line=0.9,
                        alpha_fill=0.15,
                        linewidth=linewidth,
                    )

            else:  # vainilla
                if "loss" in rows[0]:
                    ax1.plot(
                        epochs,
                        [float(r["loss"]) for r in rows],
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.7,
                        linestyle="-",
                        markersize=6,
                    )
                if "val_loss" in rows[0]:
                    ax1.plot(
                        epochs,
                        [float(r["val_loss"]) for r in rows],
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.5,
                        linestyle="--",
                        markersize=6,
                    )

                if "train_accuracy" in rows[0]:
                    ax2.plot(
                        epochs,
                        [float(r["train_accuracy"]) for r in rows],
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.7,
                        linestyle="-",
                        markersize=6,
                    )
                if "val_accuracy" in rows[0]:
                    ax2.plot(
                        epochs,
                        [float(r["val_accuracy"]) for r in rows],
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.5,
                        linestyle="--",
                        markersize=6,
                    )

                if "train_auc" in rows[0]:
                    kwargs = dict(
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.7,
                        linestyle="-",
                        markersize=6,
                    )
                    if show_legend:
                        kwargs["label"] = label
                    ax3.plot(epochs, [float(r["train_auc"]) for r in rows], **kwargs)
                if "val_auc" in rows[0]:
                    ax3.plot(
                        epochs,
                        [float(r["val_auc"]) for r in rows],
                        marker=marker,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.5,
                        linestyle="--",
                        markersize=6,
                    )

        except (KeyError, ValueError) as e:
            print(f"Skipping {log_file} - error processing data: {e}")
            continue

    ax1.set_xlabel("Épocas", fontsize=12)
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Épocas", fontsize=12)
    ax2.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100.0)

    ax3.set_xlabel("Épocas", fontsize=12)
    ax3.set_title("AUC-ROC", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)

    handles, labels_leg = ax3.get_legend_handles_labels()
    square_handles = [
        Patch(facecolor=h.get_color(), label=l) for h, l in zip(handles, labels_leg)
    ]
    ax3.legend(
        square_handles,
        labels_leg,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout(pad=1.0)
    plt.savefig(str(output_filename), dpi=300, bbox_inches="tight")
    print(f"O Plot jardouse en: {output_filename}")
    plt.show()


if __name__ == "__main__":
    import sys

    log_dir = sys.argv[1] if len(sys.argv) > 1 else "data/runs"
    plot_dir = sys.argv[2] if len(sys.argv) > 2 else "data/plots/"
    modo = sys.argv[3] if len(sys.argv) > 3 else "sombra"
    plot_logs(log_dir, plot_dir, modo=modo)
