import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import yaml
from pathlib import Path
import numpy as np

from src.utils import cargar_config_yaml, CSVLogger

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Field name shortcuts for cleaner labels
FIELD_SHORTCUTS = {
    'batch_size': 'b',
    'learning_rate': 'lr',
    'lr': 'lr',
    'weight_decay': 'wd',
    'label_smoothing': 'ls',
    'tamano_patch': 'tp',
    'tamano_token': 'tt',
    'num_sigmas': 'ns',
    'sigma': 's',
    'contador_aumento': 'ca',
    'overlap_rate': 'or',
    'label_mode': 'lm',
    'aumento_datos': 'aug',
    'use_amp': 'amp',
    'paciencia': 'pac',
    'optimizer': 'opt',
    'epochs': 'e',
    'arch': 'a',
}

def get_varying_fields(configs):
    """
    Identify which fields vary across all configs.
    Returns list of field names that have at least 2 different values.
    Treats missing fields as NaN.
    """
    if not configs:
        return []
    
    # Get all possible keys from all configs
    all_keys = set()
    for config in configs:
        if config:
            all_keys.update(config.keys())
    
    varying_fields = []
    for key in all_keys:
        values = set()
        for config in configs:
            if config:
                val = config.get(key, np.nan)
                # Convert unhashable types to string for set comparison
                if isinstance(val, (list, dict)):
                    val = str(val)
                values.add(val)
        
        # Keep field if it has more than 1 unique value
        if len(values) > 1:
            varying_fields.append(key)
    
    return sorted(varying_fields)

def config_to_label(config, varying_fields):
    """Create a label from config using only varying fields with shortcuts."""
    if not config or not varying_fields:
        return "unknown"
    
    parts = []
    for field in varying_fields:
        val = config.get(field, np.nan)
        # Use shortcut if available, otherwise use field name
        field_display = FIELD_SHORTCUTS.get(field, field)
        
        if isinstance(val, float) and np.isnan(val):
            parts.append(f"{field_display}:NaN")
        elif isinstance(val, bool):
            parts.append(f"{field_display}:{int(val)}")
        else:
            parts.append(f"{field_display}:{val}")
    
    return ", ".join(parts)

#TODO: refactorizar
def get_marker_and_linewidth(config, varying_fields):
    """Determine marker style and line width based on varying parameters."""
    label_mode = config.get('label_mode', 'vainilla')
    sigma = config.get('sigma', 1)
    
    if label_mode == 'gaussian':
        marker = 'o'
        linewidth = sigma * 0.5 if isinstance(sigma, (int, float)) else 1.5
    else:  # vanilla
        marker = 'o'
        linewidth = 1
    
    return marker, linewidth

def smooth_with_sigma(values, window=70):
    """
    Centered rolling mean and std over a window.
    Returns (mean, mean-std, mean+std) as numpy arrays of the same length.
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

def plot_sombra(ax, epochs, values, color, linestyle, alpha_line, alpha_fill, label=None, linewidth=1.5):
    """
    Plot a smoothed mean line with ±1σ flanking lines and a shaded band between them.

    alpha_line : opacity of the mean line
    alpha_fill : opacity of the shadow band (0.2 for train, 0.4 for val)
    """
    epochs = np.array(epochs)
    mean, lower, upper = smooth_with_sigma(np.array(values, dtype=float))

    # Shaded band between -1σ and +1σ
    ax.fill_between(epochs, lower, upper, color=color, alpha=alpha_fill, linewidth=0)

    # ±1σ flanking lines (thinner, same color, more transparent)
    ax.plot(epochs, lower, color=color, alpha=alpha_line * 0.5, linestyle=linestyle, linewidth=linewidth * 0.5)
    ax.plot(epochs, upper, color=color, alpha=alpha_line * 0.5, linestyle=linestyle, linewidth=linewidth * 0.5)

    # Mean trend line
    kwargs = dict(color=color, alpha=alpha_line, linestyle=linestyle, linewidth=linewidth)
    if label is not None:
        kwargs['label'] = label
    ax.plot(epochs, mean, **kwargs)


def plot_logs(log_dir='data/runs', output_file='data/plots', modo='sombra'):
    """
    Read all log files and create plots.

    modo='vainilla' — original scatter/line plot per run, no smoothing.
    modo='sombra'   — smoothed trend line per run with a shaded error band
                      between the raw noisy signal and the smooth.
                      Shadow alpha: 0.2 for train curves, 0.4 for val curves.
    """
    # Get all log files
    log_files = [f for f in os.listdir(log_dir) if not f.startswith('.') and f.endswith('.log')]
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    # Create output directory structure
    output_dir = Path(log_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename by concatenating all checkpoint names
    checkpoint_names = '_'.join([f.replace('.log', '') for f in sorted(log_files)])
    output_filename = output_dir / f'{checkpoint_names}.png'
    
    # First pass: load all configs to determine varying fields
    configs = []
    log_file_configs = {}
    
    for log_file in log_files:
        base_name = log_file.replace('.log', '')
        checkpoint_path = f"data/weights/{base_name}.pth.tar"
        
        try:
            config = cargar_config_yaml(checkpoint_path, log_dir)
            configs.append(config)
            log_file_configs[log_file] = config
        except (FileNotFoundError, SystemExit):
            print(f"Skipping {log_file} - couldn't load config")
            log_file_configs[log_file] = None
    
    # Determine which fields vary
    varying_fields = get_varying_fields(configs)
    print(f"\nVarying fields across runs: {varying_fields}\n")
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Track unique configs for color assignment and legend
    config_colors = {}
    config_labels_shown = {}
    
    discrete_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
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
            config_colors[config_key] = discrete_colors[color_idx % len(discrete_colors)]
            color_idx += 1
        color = config_colors[config_key]
        
        label = config_to_label(config, varying_fields)
        show_legend = config_key not in config_labels_shown
        config_labels_shown[config_key] = True
        
        try:
            epochs = [float(row['epoch']) for row in rows]

            if modo == 'sombra':
                # ── SOMBRA MODE ────────────────────────────────────────────
                if 'loss' in rows[0]:
                    plot_sombra(ax1, epochs,
                                [float(r['loss']) for r in rows],
                                color=color, linestyle='-',
                                alpha_line=0.9, alpha_fill=0.05,
                                linewidth=linewidth)
                if 'val_loss' in rows[0]:
                    plot_sombra(ax1, epochs,
                                [float(r['val_loss']) for r in rows],
                                color=color, linestyle='--',
                                alpha_line=0.9, alpha_fill=0.15,
                                linewidth=linewidth)

                if 'train_accuracy' in rows[0]:
                    plot_sombra(ax2, epochs,
                                [float(r['train_accuracy']) for r in rows],
                                color=color, linestyle='-',
                                alpha_line=0.9, alpha_fill=0.05,
                                linewidth=linewidth)
                if 'val_accuracy' in rows[0]:
                    plot_sombra(ax2, epochs,
                                [float(r['val_accuracy']) for r in rows],
                                color=color, linestyle='--',
                                alpha_line=0.9, alpha_fill=0.15,
                                linewidth=linewidth)

                if 'train_auc' in rows[0]:
                    plot_sombra(ax3, epochs,
                                [float(r['train_auc']) for r in rows],
                                color=color, linestyle='-',
                                alpha_line=0.9, alpha_fill=0.05,
                                label=label if show_legend else None,
                                linewidth=linewidth)
                if 'val_auc' in rows[0]:
                    plot_sombra(ax3, epochs,
                                [float(r['val_auc']) for r in rows],
                                color=color, linestyle='--',
                                alpha_line=0.9, alpha_fill=0.15,
                                linewidth=linewidth)

            else:
                # ── VAINILLA MODE (original behaviour) ─────────────────────
                if 'loss' in rows[0]:
                    loss = [float(row['loss']) for row in rows]
                    ax1.plot(epochs, loss,
                             marker=marker, linewidth=linewidth,
                             color=color, alpha=0.7, linestyle='-', markersize=6)
                if 'val_loss' in rows[0]:
                    val_loss = [float(row['val_loss']) for row in rows]
                    ax1.plot(epochs, val_loss,
                             marker=marker, linewidth=linewidth,
                             color=color, alpha=0.5, linestyle='--', markersize=6)

                if 'train_accuracy' in rows[0]:
                    train_accuracy = [float(row['train_accuracy']) for row in rows]
                    ax2.plot(epochs, train_accuracy,
                             marker=marker, linewidth=linewidth,
                             color=color, alpha=0.7, linestyle='-', markersize=6)
                if 'val_accuracy' in rows[0]:
                    val_accuracy = [float(row['val_accuracy']) for row in rows]
                    ax2.plot(epochs, val_accuracy,
                             marker=marker, linewidth=linewidth,
                             color=color, alpha=0.5, linestyle='--', markersize=6)

                if 'train_auc' in rows[0]:
                    train_auc = [float(row['train_auc']) for row in rows]
                    kwargs = dict(marker=marker, linewidth=linewidth,
                                  color=color, alpha=0.7, linestyle='-', markersize=6)
                    if show_legend:
                        kwargs['label'] = label
                    ax3.plot(epochs, train_auc, **kwargs)
                if 'val_auc' in rows[0]:
                    val_auc = [float(row['val_auc']) for row in rows]
                    ax3.plot(epochs, val_auc,
                             marker=marker, linewidth=linewidth,
                             color=color, alpha=0.5, linestyle='--', markersize=6)

        except (KeyError, ValueError) as e:
            print(f"Skipping {log_file} - error processing data: {e}")
            continue
    
    ax1.set_xlabel('Épocas', fontsize=12)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Épocas', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100.0)
    
    ax3.set_xlabel('Épocas', fontsize=12)
    ax3.set_title('AUC-ROC', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    if ax3.get_lines():
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout(pad=1.0)
    plt.savefig(str(output_filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/runs'
    plot_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/plots/'
    modo    = sys.argv[3] if len(sys.argv) > 3 else 'sombra'
    plot_logs(log_dir, plot_dir, modo=modo)
