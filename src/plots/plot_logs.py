import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import yaml
from pathlib import Path
import numpy as np

from src.utils import cargar_config_yaml, CSVLogger

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

def get_marker_and_linewidth(config, varying_fields):
    """Determine marker style and line width based on varying parameters."""
    label_mode = config.get('label_mode', 'vainilla')
    sigma = config.get('sigma', 1)
    
    if label_mode == 'gaussian':
        marker = 'o'
        linewidth = sigma * 0.5 if isinstance(sigma, (int, float)) else 1.5
    else:  # vanilla
        marker = 'x'
        linewidth = 1
    
    return marker, linewidth

def plot_logs(log_dir='data/runs', output_file='data/plots'):
    """Read all log files and create plots."""
    # Get all log files
    log_files = [f for f in os.listdir(log_dir) if not f.startswith('.') and f.endswith('.log')]
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    # Create output directory structure: log_dir/plots/ (same as patch_inference)
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
    config_labels_shown = {}  # Track which labels have been added to legend
    
    # Use tab20 for distinct colors, cycle through if needed
    discrete_colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_idx = 0

    for log_file in log_files:
        config = log_file_configs.get(log_file)
        if config is None:
            continue
        
        # Read CSV using CSVLogger
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
        
        # Get plotting parameters
        marker, linewidth = get_marker_and_linewidth(config, varying_fields)
        
        # Get or assign color based on config
        config_key = str(sorted(config.items()))
        if config_key not in config_colors:
            # Assign sequential color for better diversity
            config_colors[config_key] = discrete_colors[color_idx % len(discrete_colors)]
            color_idx += 1
        color = config_colors[config_key]
        
        # Create label from varying fields only
        label = config_to_label(config, varying_fields)
        
        # Only add to legend if this is the first time we see this label
        show_legend = config_key not in config_labels_shown
        config_labels_shown[config_key] = True
        
        try:
            epochs = [float(row['epoch']) for row in rows]
            
            # Plot train loss if available
            if 'loss' in rows[0]:
                loss = [float(row['loss']) for row in rows]
                ax1.plot(epochs, loss, 
                        marker=marker, linewidth=linewidth, 
                        color=color, alpha=0.7, linestyle='-',
                        markersize=6)
            
            # Plot val loss if available
            if 'val_loss' in rows[0]:
                val_loss = [float(row['val_loss']) for row in rows]
                ax1.plot(epochs, val_loss, 
                        marker=marker, linewidth=linewidth, 
                        color=color, alpha=0.5, linestyle='--',
                        markersize=6)
            
            # Plot train accuracy if available
            if 'train_accuracy' in rows[0]:
                train_accuracy = [float(row['train_accuracy']) for row in rows]
                ax2.plot(epochs, train_accuracy,
                        marker=marker, linewidth=linewidth, 
                        color=color, alpha=0.7, linestyle='-',
                        markersize=6)
            
            # Plot val accuracy if available
            if 'val_accuracy' in rows[0]:
                val_accuracy = [float(row['val_accuracy']) for row in rows]
                ax2.plot(epochs, val_accuracy, 
                        marker=marker, linewidth=linewidth, 
                        color=color, alpha=0.5, linestyle='--',
                        markersize=6)

            # Plot train AUC if available
            if 'train_auc' in rows[0]:
                train_auc = [float(row['train_auc']) for row in rows]
                if show_legend:
                    ax3.plot(epochs, train_auc,
                            marker=marker, linewidth=linewidth, 
                            color=color, alpha=0.7, linestyle='-',
                            label=label,
                            markersize=6)
                else:
                    ax3.plot(epochs, train_auc,
                            marker=marker, linewidth=linewidth, 
                            color=color, alpha=0.7, linestyle='-',
                            markersize=6)
            
            # Plot val AUC if available
            if 'val_auc' in rows[0]:
                val_auc = [float(row['val_auc']) for row in rows]
                ax3.plot(epochs, val_auc, 
                        marker=marker, linewidth=linewidth, 
                        color=color, alpha=0.5, linestyle='--',
                        markersize=6)
        except (KeyError, ValueError) as e:
            print(f"Skipping {log_file} - error processing data: {e}")
            continue
    
    # Configure loss plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Configure accuracy plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100.0)
    
    # Configure AUC plot
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC-ROC', fontsize=12)
    ax3.set_title('AUC-ROC', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    # Only show legend if there are labeled artists
    if ax3.get_lines():
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout(pad=1.0)
    
    # Save figure
    plt.savefig(str(output_filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    # You can specify a different directory if needed
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/runs'
    plot_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/plots/'
    plot_logs(log_dir, plot_dir)
