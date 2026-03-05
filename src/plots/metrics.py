import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def cal_sparsity(matrix, is_sparse=False):
    absmatrix = np.abs(matrix)
    # matrix have shape [batch_size, num_patches, dim]
    if is_sparse == True:
        sparsity_list = [
            np.count_nonzero(absmatrix[i, :, :] == 0)
            / (matrix.shape[1] * matrix.shape[2])
            for i in range(matrix.shape[0])
        ]
        sparsity = np.mean(sparsity_list)
        stdev = np.std(sparsity_list)
    else:
        sparsity = None
        stdev = None

    return sparsity, stdev

def plot_sparsity(sparsities, std_sparsities, name, labels=None, colors=None, legend=False):
    fontsize = 20
    n = len(sparsities)

    if labels is None:
        labels = [f"val{i}" if i > 0 else "val" for i in range(n)]
    if colors is None:
        colors = [f"C{i + 1}" for i in range(n)]  # C1, C2, … to match original C1

    plt.rcParams["figure.figsize"] = (10, 6)
    fig, ax = plt.subplots()

    for mean_sparsity, std_sparsity, label, color in zip(
        sparsities, std_sparsities, labels, colors
    ):
        std_sparsity  = np.asarray(std_sparsity)
        x_labels      = np.arange(len(mean_sparsity)) + 1

        ax.plot(
            x_labels, mean_sparsity,
            marker="s", markersize=8, linewidth=2.5,
            markeredgecolor="black", markeredgewidth=1.0,
            color=color, alpha=0.9, label=label,
        )
        ax.fill_between(
            x_labels,
            mean_sparsity - std_sparsity,
            mean_sparsity + std_sparsity,
            color=color, alpha=0.15,
        )

    ax.set_title("Measure output sparsity across layers", fontdict={"fontsize": fontsize})
    ax.set_ylabel(r"Sparsity [ISTA block]",               fontdict={"fontsize": fontsize})
    ax.set_xlabel(r"Layer index - $\ell$",                fontdict={"fontsize": fontsize})
    ax.grid(linestyle="--", color="gray")

    handles, labels_leg = ax.get_legend_handles_labels()
    square_handles = [
        Patch(facecolor=h.get_color(), label=l) for h, l in zip(handles, labels_leg)
    ]
    if legend:
        ax.legend(
            square_handles,
            labels_leg,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f"{name}_sparsity.png", format="png", dpi=600)
    plt.close()


def plot_coding_rate(means, std_devs, name, labels=None, colors=None, legend=False):
    fontsize = 20
    n = len(means)

    if labels is None:
        labels = [f"val{i}" if i > 0 else "val" for i in range(n)]
    if colors is None:
        colors = [f"C{i}" for i in range(n)]  # C0, C1, … to match original C0

    plt.rcParams["figure.figsize"] = (10, 6)
    fig, ax = plt.subplots()

    for mean_mcr2, std_mcr2, label, color in zip(means, std_devs, labels, colors):
        mean_mcr2 = np.asarray(mean_mcr2)
        std_mcr2  = np.asarray(std_mcr2)
        x_labels  = np.arange(len(mean_mcr2)) + 1

        ax.plot(
            x_labels, mean_mcr2,
            marker="s", markersize=10, linewidth=2.5,
            markeredgecolor="black", markeredgewidth=1.0,
            color=color, alpha=0.9, label=label,
        )
        ax.fill_between(
            x_labels,
            mean_mcr2 - std_mcr2,
            mean_mcr2 + std_mcr2,
            color=color, alpha=0.15,
        )

    ax.set_title("Measure coding rate across layers",     fontdict={"fontsize": fontsize})
    ax.set_ylabel(r"$R^c(Z^{\ell})$ [SSA block]",        fontdict={"fontsize": fontsize})
    ax.set_xlabel(r"Layer index - $\ell$",                fontdict={"fontsize": fontsize})
    ax.grid(linestyle="--", color="gray")

    handles, labels_leg = ax.get_legend_handles_labels()
    square_handles = [
        Patch(facecolor=h.get_color(), label=l) for h, l in zip(handles, labels_leg)
    ]
    if legend:
        ax.legend(
            square_handles,
            labels_leg,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(f"{name}_mcr2.png", format="png", dpi=600)
    plt.close()
