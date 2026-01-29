



# NECESITA REFACTORIZACION






#!/usr/bin/env python3
"""
Visualize CRATE attention head activations.
Simple and direct approach for PyTorch 2.4.1
Modified to align heads consistently across all images.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import model
sys.path.insert(0, str(Path.cwd()))
from model.crate import CRATE

# Import dataset
try:
    from data.DRIVE_SSL_Dataset import DriveSSLDataset
    HAS_DATASET = True
except ImportError:
    HAS_DATASET = False
    from PIL import Image
    from torchvision import transforms


def load_checkpoint(checkpoint_path):
    """Load checkpoint - handles both file and directory formats."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully")
    return checkpoint


def create_model(checkpoint, args):
    """Create and load CRATE model."""
    print("Creating CRATE model...")
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Auto-detect model configuration from checkpoint
    print("Auto-detecting model configuration from checkpoint...")
    
    # Get position embedding shape to infer num_patches
    if 'pos_embedding' in state_dict:
        pos_emb_shape = state_dict['pos_embedding'].shape
        num_patches_plus_cls = pos_emb_shape[1]  # includes CLS token
        num_patches = num_patches_plus_cls - 1
        print(f"  Position embedding shape: {pos_emb_shape}")
        print(f"  Number of patches (excluding CLS): {num_patches}")
        
        # Calculate image size from num_patches
        # num_patches = (image_size // patch_size) ** 2
        patches_per_side = int(np.sqrt(num_patches))
        detected_image_size = patches_per_side * args.patch_size
        print(f"  Detected patches per side: {patches_per_side}")
        print(f"  Detected image size: {detected_image_size}")
        
        if detected_image_size != args.image_size:
            print(f"  WARNING: Provided image_size ({args.image_size}) != detected ({detected_image_size})")
            print(f"  Using detected image_size: {detected_image_size}")
            args.image_size = detected_image_size
    
    # Get model dimension
    dim = 384
    if 'cls_token' in state_dict:
        dim = state_dict['cls_token'].shape[-1]
        print(f"  Model dimension: {dim}")
    
    # Get number of heads
    num_heads = 6
    depth = 12
    
    # Try to infer from transformer layers
    transformer_keys = [k for k in state_dict.keys() if 'transformer.layers' in k]
    if transformer_keys:
        # Count layers
        layer_indices = set()
        for k in transformer_keys:
            parts = k.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
        if layer_indices:
            depth = max(layer_indices) + 1
            print(f"  Detected depth: {depth}")
        
        # Try to infer heads from qkv weight shape
        qkv_keys = [k for k in state_dict.keys() if 'qkv.weight' in k]
        if qkv_keys:
            qkv_shape = state_dict[qkv_keys[0]].shape
            # qkv_weight shape is [inner_dim, dim] where inner_dim = heads * dim_head
            # We know dim, so: heads = inner_dim / dim_head
            inner_dim = qkv_shape[0]
            dim_head = dim // num_heads  # default assumption
            num_heads = inner_dim // dim_head
            print(f"  Detected heads: {num_heads}")
    
    model = CRATE(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        dim=dim,
        depth=depth,
        heads=num_heads,
        dropout=0.0,
        emb_dropout=0.0,
        dim_head=dim // num_heads
    )
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        if len(missing) < 10:
            for k in missing:
                print(f"  - {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) < 10:
            for k in unexpected:
                print(f"  - {k}")
    
    model.eval()
    print("Model loaded successfully")
    return model, depth, num_heads


def load_images(args):
    """Load images from dataset."""
    images = []
    
    if HAS_DATASET and args.dataset_path:
        print(f"Loading from DRIVE dataset: {args.dataset_path}")
        try:
            dataset = DriveSSLDataset(
                args.dataset_path,
                tamano_patch=args.image_size,
                label_mode='binary',
                sigma=1.0
            )
            
            # Sample random images
            indices = np.random.choice(len(dataset), min(args.num_images, len(dataset)), replace=False)
            for idx in indices:
                img, _ = dataset[int(idx)]
                images.append(img)
            
            images = torch.stack(images)
            print(f"Loaded {len(images)} images")
            return images
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    # Generate random images as fallback
    print(f"Generating {args.num_images} random images")
    images = torch.randn(args.num_images, 3, args.image_size, args.image_size)
    return images


def get_attention_maps(model, images, layer_indices, num_heads):
    """Extract attention maps from specified layers."""
    activations = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # Store the output after attention
            activations[f'layer_{layer_idx}'] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for layer_idx in layer_indices:
        layer = model.transformer.layers[layer_idx]
        # Hook the PreNorm->Attention module (first in the layer)
        hook = layer[0].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process activations to get attention head outputs
    results = {}
    for layer_idx in layer_indices:
        key = f'layer_{layer_idx}'
        if key in activations:
            act = activations[key]  # [B, N, D]
            B, N, D = act.shape
            
            # Reshape to separate heads: [B, N, D] -> [B, N, H, D_h] -> [B, H, N, D_h]
            act = act.reshape(B, N, num_heads, D // num_heads)
            act = act.permute(0, 2, 1, 3)
            
            results[key] = act
    
    return results


def visualize(images, attention_maps, layer_indices, num_heads_to_show, patch_size, output_path, head_indices_per_layer):
    """Create visualization grid with consistent head alignment across all images."""
    
    num_images = images.shape[0]
    num_layers = len(layer_indices)
    
    # Columns: original image + num_heads_to_show * num_layers
    num_cols = 1 + (num_heads_to_show * num_layers)
    
    fig, axes = plt.subplots(num_images, num_cols, figsize=(num_cols * 3, num_images * 3))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    # Calculate spatial dimensions
    img_size = images.shape[-1]
    num_patches = img_size // patch_size
    
    print(f"Image size: {img_size}, Patch size: {patch_size}, Patches per side: {num_patches}")
    print(f"\nUsing consistent head indices per layer:")
    for layer_idx in layer_indices:
        print(f"  Layer {layer_idx}: heads {head_indices_per_layer[layer_idx]}")
    
    for img_idx in range(num_images):
        col_idx = 0
        
        # Plot original image
        ax = axes[img_idx, col_idx]
        img = images[img_idx].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'Image {img_idx + 1}')
        ax.axis('off')
        col_idx += 1
        
        # Plot attention heads for each layer (using consistent head indices)
        for layer_idx in layer_indices:
            key = f'layer_{layer_idx}'
            
            if key not in attention_maps:
                print(f"Warning: {key} not in attention_maps")
                # Fill with empty plots
                for _ in range(num_heads_to_show):
                    ax = axes[img_idx, col_idx]
                    ax.axis('off')
                    col_idx += 1
                continue
            
            heads = attention_maps[key]  # [B, H, N, D_h]
            num_tokens = heads.shape[2]
            
            # Use the pre-selected head indices for this layer
            selected_heads = head_indices_per_layer[layer_idx]
            
            for head_idx in selected_heads:
                ax = axes[img_idx, col_idx]
                
                # Get activation for this head and image
                head_act = heads[img_idx, head_idx]  # [N, D_h]
                
                # Remove CLS token (first token)
                if num_tokens > 1:
                    head_act = head_act[1:]  # [N-1, D_h]
                    num_spatial_tokens = head_act.shape[0]
                else:
                    num_spatial_tokens = num_tokens
                
                # Average over feature dimension
                spatial_act = head_act.mean(dim=-1).cpu().numpy()  # [N-1]
                
                # Reshape to spatial grid
                try:
                    # Calculate actual patches per side from number of tokens
                    actual_patches_per_side = int(np.sqrt(num_spatial_tokens))
                    
                    if actual_patches_per_side * actual_patches_per_side != num_spatial_tokens:
                        raise ValueError(f"Cannot reshape {num_spatial_tokens} tokens into square grid")
                    
                    act_map = spatial_act.reshape(actual_patches_per_side, actual_patches_per_side)
                    
                    # Upsample to image size
                    from scipy.ndimage import zoom
                    zoom_factor = img_size / actual_patches_per_side
                    act_map_up = zoom(act_map, zoom_factor, order=1)
                    
                    # Plot
                    im = ax.imshow(act_map_up, cmap='jet', interpolation='bilinear')
                    ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=10)
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"    Error for Image {img_idx}, Layer {layer_idx}, Head {head_idx}: {e}")
                    ax.text(0.5, 0.5, f'Error\n{e}', ha='center', va='center', fontsize=8)
                    ax.axis('off')
                
                col_idx += 1
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CRATE attention heads')
    
    # Required
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    
    # Model config
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--patch-size', type=int, default=16, help='Patch size')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    
    # Visualization
    parser.add_argument('-h_vis', '--num-heads', type=int, default=4, 
                        help='Number of heads to visualize per layer')
    parser.add_argument('-n', '--num-last-layers', type=int, default=1,
                        help='Number of last layers to visualize')
    parser.add_argument('--num-images', '--img', type=int, default=2, 
                        help='Number of images to visualize')
    parser.add_argument('--first-layer', action='store_true',
                        help='Include first layer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for head selection')
    
    # Data
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Path to DRIVE dataset')
    
    # Output
    parser.add_argument('-o', '--output', type=str, default='attention_visualization.png',
                        help='Output path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load checkpoint and model
    checkpoint = load_checkpoint(args.checkpoint)
    model, depth, num_heads = create_model(checkpoint, args)
    model = model.to(args.device)
    
    # Determine layers to visualize
    layer_indices = []
    
    if args.first_layer:
        layer_indices.append(0)
    
    # Add last n layers
    for i in range(args.num_last_layers):
        layer_idx = depth - 1 - i
        if layer_idx not in layer_indices and layer_idx >= 0:
            layer_indices.append(layer_idx)
    
    layer_indices.sort()
    print(f"Visualizing layers: {layer_indices}")
    
    # PRE-SELECT head indices for each layer (consistent across all images)
    print(f"\nPre-selecting {args.num_heads} heads per layer...")
    head_indices_per_layer = {}
    for layer_idx in layer_indices:
        # Select random heads, but do it once for all images
        selected = np.random.choice(num_heads, min(args.num_heads, num_heads), replace=False)
        head_indices_per_layer[layer_idx] = sorted(selected.tolist())
    
    # Load images
    images = load_images(args)
    images = images.to(args.device)
    
    # Extract attention maps
    print("\nExtracting attention maps...")
    attention_maps = get_attention_maps(model, images, layer_indices, num_heads=num_heads)
    
    # Visualize with consistent head alignment
    print("\nCreating visualization...")
    visualize(
        images=images.cpu(),
        attention_maps=attention_maps,
        layer_indices=layer_indices,
        num_heads_to_show=args.num_heads,
        patch_size=args.patch_size,
        output_path=args.output,
        head_indices_per_layer=head_indices_per_layer
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
