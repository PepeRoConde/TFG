import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
from matplotlib.patches import Rectangle


def plot_mapas_atencion(
    imaxes,
    mapas_atencion,
    indices_capas,
    num_cabezas,
    output_path,
    indices_cabezas_por_capa,
    offset,
    etiquetas=None,
    logits=None,
):
    num_imaxes = imaxes.shape[0]
    num_capas = len(indices_capas)
    print(f"num_cabezas: {num_cabezas}, num_capas: {num_capas}")

    # Columnas: imaxe orixinal + num_cabezas * num_capas + 1 (logits barplot)
    num_cols = 1 + (num_cabezas * num_capas) + (1 if logits is not None else 0)

    fig, axes = plt.subplots(
        num_imaxes, num_cols, figsize=(num_cols * 3, num_imaxes * 3)
    )

    # iteracion sobre as imaxes
    for img_idx, layer_dict in mapas_atencion.items():
        # Plotear imaxe orixinal

        ax = axes[img_idx, 0]  # columna 0
        img = (
            imaxes[img_idx]
            .permute(1, 2, 0)
            .cpu()
            #            .numpy()[offset[0] : offset[1], offset[0] : offset[1]]
            .numpy()
        )
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.axis("off")

        # Crear título con info da etiqueta
        if etiquetas is not None and img_idx < len(etiquetas):
            label = etiquetas[img_idx]
            if isinstance(label, torch.Tensor):
                label_val = label[1].item() if label.numel() == 2 else label.item()
            else:
                label_val = label

            color = "green" if label_val >= 0.5 else "red"
            rect = Rectangle(
                (0, 0),
                img.shape[1] - 1,
                img.shape[0] - 1,
                linewidth=4,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

        # Iteracion sobre as capas
        for i, layer_idx in enumerate(indices_capas):
            key = f"layer.{layer_idx}"

            if key not in layer_dict:
                raise ValueError(f"Aviso: {key} non está en mapas_atencion[{img_idx}]")
                continue

            current_layer = layer_dict[key]

            # Usar os índices de cabezas pre-seleccionados para esta capa
            cabezas_seleccionadas = indices_cabezas_por_capa[layer_idx]

            # Iteracion sobre as cabezas
            for j, head_idx in enumerate(cabezas_seleccionadas):
                ax = axes[img_idx, i * len(cabezas_seleccionadas) + j + 1]

                try:
                    attn_matrix = current_layer[head_idx].cpu().numpy()

                    # igual esto devuelve algo? si va mal probar con eso
                    ax.imshow(attn_matrix, cmap="copper", interpolation="nearest")
                    ax.set_title(f"L{layer_idx} H{head_idx}", fontsize=10)
                    ax.axis("off")

                except Exception as e:
                    print(
                        f"    Erro para Imaxe {img_idx}, Capa {layer_idx}, Cabeza {head_idx}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"Erro\n{str(e)[:30]}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
                    ax.axis("off")

        # Plotear logits barplot at the end of each row
        if logits is not None and img_idx in logits:
            ax_logits = axes[img_idx, -1]
            logits_values = logits[img_idx].numpy()

            # Handle both binary (1D) and multi-class cases
            if logits_values.ndim == 0:
                logits_values = np.array([logits_values])
            elif logits_values.ndim > 1:
                logits_values = logits_values.flatten()

            # Create barplot
            colors = ["red" if v < 0.5 else "green" for v in logits_values]
            ax_logits.bar(
                range(len(logits_values)),
                logits_values,
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )
            ax_logits.set_ylim([0, 1])
            ax_logits.set_title("Logits", fontsize=10)
            # ax_logits.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            # for i, (bar, val) in enumerate(zip(bars, logits_values)):
            #    ax_logits.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            #                  ha='center', va='bottom', fontsize=5)

    print("Visulización confecionada")
    # plt.tight_layout()

    # Crear directorio se non existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
