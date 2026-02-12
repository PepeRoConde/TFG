import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import yaml
from pathlib import Path
import numpy as np

from src.utils import cargar_config_yaml

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
        linewidth = 1.5
    
    return marker, linewidth

def plot_logs(log_dir='data/runs', output_file='data/plots'):
    """Read all log files and create plots."""
    # Get all log files
    log_files = [f for f in os.listdir(log_dir) if not f.startswith('.') and f.endswith('.log')]
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Track unique configs for color assignment and legend
    config_colors = {}
    config_labels_shown = {}  # Track which labels have been added to legend
    

    for log_file in log_files:
        config = log_file_configs.get(log_file)
        if config is None:
            continue
        
        # Read CSV
        filepath = os.path.join(log_dir, log_file)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
            continue
        
        # Get plotting parameters
        marker, linewidth = get_marker_and_linewidth(config, varying_fields)
        
        # Get or assign color based on config
        config_key = str(sorted(config.items()))
        if config_key not in config_colors:
            # Generate color from hash using continuous colormap
            import hashlib
            import matplotlib.cm as cm
            hash_val = int(hashlib.md5(config_key.encode()).hexdigest(), 16)
            # Normalize to [0, 1] using modulo to ensure unique mapping
            normalized = (hash_val % 10000) / 10000.0
            # Use hsv colormap for better color distribution
            config_colors[config_key] = cm.hsv(normalized)
        color = config_colors[config_key]
        
        # Create label from varying fields only
        label = config_to_label(config, varying_fields)
        
        # Only add to legend if this is the first time we see this label
        show_legend = config_key not in config_labels_shown
        config_labels_shown[config_key] = True
        
        legend_label = label if show_legend else None
        
        # Plot loss
        ax1.plot(df['epoch'], df['loss'], 
                marker=marker, linewidth=linewidth, 
                color=color, alpha=0.7, label=legend_label,
                markersize=6)
        
        # Plot accuracies
        ax2.plot(df['epoch'], df['train_accuracy'],
                marker=marker, linewidth=linewidth, 
                color=color, alpha=0.7, linestyle='-',
                label=f"{label} (train)" if show_legend else None, markersize=6)
        ax2.plot(df['epoch'], df['val_accuracy'], 
                marker=marker, linewidth=linewidth, 
                color=color, alpha=0.5, linestyle='--',
                markersize=6)

        output_file += log_file.replace('.log', '_')
    
    # Configure loss plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Configure accuracy plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Train (solid) vs Val (dashed) Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # You can specify a different directory if needed
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/runs'
    plot_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/plots/'
    plot_logs(log_dir, plot_dir)
