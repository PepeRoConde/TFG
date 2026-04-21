import os
import argparse
import matplotlib.pyplot as plt
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


def ensure_positive(values, min_positive=1e-12):
    """Clip values to a strictly positive floor for safe log-scale plotting."""
    arr = np.array(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, min_positive)
    return np.maximum(arr, min_positive)


def has_nonzero_signal(values, atol=1e-12):
    """Return True when the series contains any finite value with meaningful magnitude."""
    arr = np.array(values, dtype=float)
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size == 0:
        return False
    return np.any(np.abs(finite_vals) > atol)


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
    marker=None,
    markersize=6,
    markevery=None,
    use_log=False,
    min_positive=1e-12,
):
    epochs = np.array(epochs)
    mean, lower, upper = smooth_with_sigma(np.array(values, dtype=float))

    if use_log:
        mean = ensure_positive(mean, min_positive=min_positive)
        lower = ensure_positive(lower, min_positive=min_positive)
        upper = ensure_positive(upper, min_positive=min_positive)
        line_plot = ax.semilogy
    else:
        line_plot = ax.plot

    ax.fill_between(epochs, lower, upper, color=color, alpha=alpha_fill, linewidth=0)
    line_plot(
        epochs,
        lower,
        color=color,
        alpha=alpha_line * 0.5,
        linestyle=linestyle,
        linewidth=linewidth * 0.5,
    )
    line_plot(
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
    if marker is not None:
        kwargs["marker"] = marker
        kwargs["markersize"] = markersize
        kwargs["markevery"] = markevery
    if label is not None:
        kwargs["label"] = label
    line_plot(epochs, mean, **kwargs)


def plot_logs(
    log_dir="data/runs",
    output_file="data/plots",
    modo="sombra",
    normalize=False,
    markevery=None,
    min_positive=1e-12,
):
    """
    Read all log files and create loss / accuracy / AUC-ROC plots.

    modo='vainilla' — original scatter/line plot per run, no smoothing.
    modo='sombra'   — smoothed trend line with shaded ±1σ error band.
    normalize=True  — divide regularization losses by their reference weight (embedding_l1_weight) at plot time
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
    output_filename = output_dir / f"{checkpoint_names}.pdf"

    # ── First pass: load configs to find varying fields ──────────────────
    log_file_configs = {}
    for log_file in log_files:
        log_file_configs[log_file] = cargar_config_yaml(log_file, log_dir)

    varying_fields = get_varying_fields(log_file_configs.values())
    print(f"As execucións varian nos campos: {varying_fields}\n")

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
                print(f"Arquivo vacio: {log_file}")
                continue
        except Exception as e:
            print(f"Erro ó ler o arquivo {log_file}: {e}")
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

            # Extract lambda for normalization if requested
            lambda_val = 1.0
            if normalize and config:
                lambda_val = float(config.get("embedding_l1_weight", 1))
                if lambda_val <= 0:
                    lambda_val = 1.0  # Avoid division by zero

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
                        markevery=markevery,
                        use_log=True,
                        min_positive=min_positive,
                    )
                if "loss_l1" in rows[0]:
                    l1_values = [
                        (
                            float(r["loss_l1"]) / lambda_val
                            if normalize
                            else float(r["loss_l1"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(l1_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            l1_values,
                            color=color,
                            linestyle=":",
                            alpha_line=0.9,
                            alpha_fill=0.05,
                            linewidth=linewidth,
                            marker="^",
                            markersize=5,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
                        )
                if "loss_orthogonal" in rows[0]:
                    orth_values = [
                        (
                            float(r["loss_orthogonal"]) / lambda_val
                            if normalize
                            else float(r["loss_orthogonal"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(orth_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            orth_values,
                            color=color,
                            linestyle=":",
                            alpha_line=0.9,
                            alpha_fill=0.05,
                            linewidth=linewidth,
                            marker="d",
                            markersize=5,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
                        )
                if "loss_reconstruction" in rows[0]:
                    recon_values = [
                        (
                            float(r["loss_reconstruction"]) / lambda_val
                            if normalize
                            else float(r["loss_reconstruction"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(recon_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            recon_values,
                            color=color,
                            linestyle=":",
                            alpha_line=0.9,
                            alpha_fill=0.05,
                            linewidth=linewidth,
                            marker="*",
                            markersize=8,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
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
                        markevery=markevery,
                        use_log=True,
                        min_positive=min_positive,
                    )
                if "val_loss_l1" in rows[0]:
                    val_l1_values = [
                        (
                            float(r["val_loss_l1"]) / lambda_val
                            if normalize
                            else float(r["val_loss_l1"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_l1_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            val_l1_values,
                            color=color,
                            linestyle="-.",
                            alpha_line=0.9,
                            alpha_fill=0.15,
                            linewidth=linewidth,
                            marker="^",
                            markersize=5,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
                        )
                if "val_loss_orthogonal" in rows[0]:
                    val_orth_values = [
                        (
                            float(r["val_loss_orthogonal"]) / lambda_val
                            if normalize
                            else float(r["val_loss_orthogonal"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_orth_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            val_orth_values,
                            color=color,
                            linestyle="-.",
                            alpha_line=0.9,
                            alpha_fill=0.15,
                            linewidth=linewidth,
                            marker="d",
                            markersize=5,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
                        )
                if "val_loss_reconstruction" in rows[0]:
                    val_recon_values = [
                        (
                            float(r["val_loss_reconstruction"]) / lambda_val
                            if normalize
                            else float(r["val_loss_reconstruction"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_recon_values):
                        plot_sombra(
                            ax1,
                            epochs,
                            val_recon_values,
                            color=color,
                            linestyle="-.",
                            alpha_line=0.9,
                            alpha_fill=0.15,
                            linewidth=linewidth,
                            marker="*",
                            markersize=8,
                            markevery=markevery,
                            use_log=True,
                            min_positive=min_positive,
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
                        markevery=markevery,
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
                        markevery=markevery,
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
                        markevery=markevery,
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
                        markevery=markevery,
                    )

            else:  # vainilla
                if "loss" in rows[0]:
                    ax1.semilogy(
                        epochs,
                        ensure_positive(
                            [float(r["loss"]) for r in rows],
                            min_positive=min_positive,
                        ),
                        marker=marker,
                        markevery=markevery,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.7,
                        linestyle="-",
                        markersize=6,
                    )
                if "loss_l1" in rows[0]:
                    l1_values = [
                        (
                            float(r["loss_l1"]) / lambda_val
                            if normalize
                            else float(r["loss_l1"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(l1_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(l1_values, min_positive=min_positive),
                            marker="^",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.7,
                            linestyle=":",
                            markersize=6,
                        )
                if "loss_orthogonal" in rows[0]:
                    orth_values = [
                        (
                            float(r["loss_orthogonal"]) / lambda_val
                            if normalize
                            else float(r["loss_orthogonal"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(orth_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(orth_values, min_positive=min_positive),
                            marker="d",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.7,
                            linestyle=":",
                            markersize=6,
                        )
                if "loss_reconstruction" in rows[0]:
                    recon_values = [
                        (
                            float(r["loss_reconstruction"]) / lambda_val
                            if normalize
                            else float(r["loss_reconstruction"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(recon_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(recon_values, min_positive=min_positive),
                            marker="*",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.7,
                            linestyle=":",
                            markersize=8,
                        )
                if "val_loss" in rows[0]:
                    ax1.semilogy(
                        epochs,
                        ensure_positive(
                            [float(r["val_loss"]) for r in rows],
                            min_positive=min_positive,
                        ),
                        marker=marker,
                        markevery=markevery,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.5,
                        linestyle="--",
                        markersize=6,
                    )
                if "val_loss_l1" in rows[0]:
                    val_l1_values = [
                        (
                            float(r["val_loss_l1"]) / lambda_val
                            if normalize
                            else float(r["val_loss_l1"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_l1_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(val_l1_values, min_positive=min_positive),
                            marker="^",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.5,
                            linestyle="-.",
                            markersize=6,
                        )
                if "val_loss_orthogonal" in rows[0]:
                    val_orth_values = [
                        (
                            float(r["val_loss_orthogonal"]) / lambda_val
                            if normalize
                            else float(r["val_loss_orthogonal"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_orth_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(val_orth_values, min_positive=min_positive),
                            marker="d",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.5,
                            linestyle="-.",
                            markersize=6,
                        )
                if "val_loss_reconstruction" in rows[0]:
                    val_recon_values = [
                        (
                            float(r["val_loss_reconstruction"]) / lambda_val
                            if normalize
                            else float(r["val_loss_reconstruction"])
                        )
                        for r in rows
                    ]
                    if has_nonzero_signal(val_recon_values):
                        ax1.semilogy(
                            epochs,
                            ensure_positive(
                                val_recon_values, min_positive=min_positive
                            ),
                            marker="*",
                            markevery=markevery,
                            linewidth=linewidth,
                            color=color,
                            alpha=0.5,
                            linestyle="-.",
                            markersize=8,
                        )

                if "train_accuracy" in rows[0]:
                    ax2.plot(
                        epochs,
                        [float(r["train_accuracy"]) for r in rows],
                        marker=marker,
                        markevery=markevery,
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
                        markevery=markevery,
                        linewidth=linewidth,
                        color=color,
                        alpha=0.5,
                        linestyle="--",
                        markersize=6,
                    )

                if "train_auc" in rows[0]:
                    kwargs = dict(
                        marker=marker,
                        markevery=markevery,
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
                        markevery=markevery,
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
        Patch(facecolor=handle.get_color(), label=label)
        for handle, label in zip(handles, labels_leg)
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


def get_args_parser():
    """
    Parse command line arguments with backward compatibility.

    Supports both:
    - Positional: plot_logs.py data/runs sombra
    - Named: plot_logs.py -log_dir data/runs -modo sombra -norm
    - Mixed: plot_logs.py data/runs -modo sombra -norm
    """
    parser = argparse.ArgumentParser(
        description="Plot training logs from multiple runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "log_dir",
        nargs="?",
        default="data/runs",
        help="Directory containing log files",
    )

    parser.add_argument(
        "-m",
        "--modo",
        dest="modo",
        default="sombra",
        choices=["sombra", "vainilla"],
        help="Plot mode: sombra (smoothed with error bands) or vainilla (raw scatter)",
    )

    parser.add_argument(
        "-norm",
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Normalize regularization losses by dividing by embedding_l1_weight",
    )

    parser.add_argument(
        "-every",
        dest="markevery",
        type=int,
        default=None,
        help="Draw markers every N points (None means default matplotlib behavior)",
    )

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    plot_logs(
        log_dir=args.log_dir,
        output_file=args.log_dir,
        modo=args.modo,
        normalize=args.normalize,
        markevery=args.markevery,
    )
