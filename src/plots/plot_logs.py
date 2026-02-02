import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import yaml
from pathlib import Path

def parse_filename(filename):
    """Extract parameters from YAML metadata file."""
    # Remove extension if present and add .yaml
    base_name = filename.replace('.log', '')
    print(base_name)
    yaml_path = f"data/runs/metadata/{base_name}.yaml"
    
    try:
        with open(yaml_path, 'r') as f:
            args_dict = yaml.safe_load(f)
        
        return {
            'tp': args_dict.get('tamano_patch'),
            'tt': args_dict.get('tamano_token'),
            's': args_dict.get('sigma'),
            'lm': args_dict.get('label_mode')
        }
    except FileNotFoundError:
        print(f"YAML file not found: {yaml_path}")
        return None
    except Exception as e:
        print(f"Error reading YAML: {e}")
        return None

def get_marker_and_linewidth(params):
    """Determine marker style and line width based on parameters."""
    if params['lm'] == 'gaussian':
        marker = 'o'
        linewidth = params['s'] * 0.5  # Scale thickness based on s parameter
    else:  # vanilla
        marker = 'x'
        linewidth = 1.5
    
    return marker, linewidth

def get_color(tp, tt):
    """Generate a unique color for each tp/tt combination."""
    # Create a unique identifier from tp and tt
    combo = f"tp:{tp}_tt:{tt}"
    # Use a hash to get consistent colors
    import matplotlib.cm as cm
    import hashlib
    
    # Create a hash and normalize to [0, 1]
    hash_val = int(hashlib.md5(combo.encode()).hexdigest(), 16)
    normalized = (hash_val % 1000) / 1000.0
    
    # Use a colormap
    return cm.tab10(normalized % 1.0)

def plot_logs(log_dir='data/runs'):
    """Read all log files and create plots."""
    # Get all log files
    log_files = [f for f in os.listdir(log_dir) if not f.startswith('.')]
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Track unique tp/tt combinations for legend
    tp_tt_colors = {}
    
    output_file = 'data/plots/'

    for log_file in log_files:
        # Parse filename
        params = parse_filename(log_file)
        if params is None:
            print(f"Skipping {log_file} - couldn't parse filename")
            continue
        
        # Read CSV
        filepath = os.path.join(log_dir, log_file)
        df = pd.read_csv(filepath)
        
        # Get plotting parameters
        marker, linewidth = get_marker_and_linewidth(params)
        
        # Get or assign color based on tp/tt
        tp_tt_key = f"tp:{params['tp']}_tt:{params['tt']}"
        if tp_tt_key not in tp_tt_colors:
            tp_tt_colors[tp_tt_key] = get_color(params['tp'], params['tt'])
        color = tp_tt_colors[tp_tt_key]
        
        # Create label
        label = f"{tp_tt_key}, s:{params['s']}, {params['lm']}"
        
        # Plot loss
        ax1.plot(df['epoch'], df['loss'], 
                marker=marker, linewidth=linewidth, 
                color=color, alpha=0.7, label=label,
                markersize=6)
        
        # Plot accuracies
        ax2.plot(df['epoch'], df['train_accuracy'],
                marker=marker, linewidth=linewidth, 
                color=color, alpha=0.7, linestyle='-',
                label=f"{label} (train)", markersize=6)
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
    plot_logs(log_dir)
