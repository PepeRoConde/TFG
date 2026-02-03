#!/usr/bin/env python3
"""
Script to pre-crop DRIVE dataset into patches using sliding window approach.

This script extracts patches from DRIVE images and masks using a sliding window
with configurable overlap rate. The patches are saved as individual PNG files
along with metadata in JSON format.

Usage:
    python crop_dataset.py --input_dir ./DRIVE/training \\
                           --output_dir ./DRIVE/training_patches \\
                           --patch_size 32 \\
                           --overlap_rate 0.5
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


def calculate_stride(patch_size: int, overlap_rate: float) -> int:
    """
    Calculate stride from patch size and overlap rate.
    
    Args:
        patch_size: Size of the square patch
        overlap_rate: Overlap rate in [0, 1]
            - 0.0: no overlap (stride = patch_size)
            - 0.5: 50% overlap (stride = patch_size / 2)
            - 0.75: 75% overlap (stride = patch_size / 4)
    
    Returns:
        Stride value (int)
    """
    if not 0 <= overlap_rate < 1:
        raise ValueError(f"overlap_rate must be in [0, 1), got {overlap_rate}")
    
    stride = int(patch_size * (1 - overlap_rate))
    if stride < 1:
        stride = 1
    
    return stride


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    stride: int
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Extract patches from image and mask using sliding window.
    
    Args:
        image: Image array [H, W, C]
        mask: Mask array [H, W, C]
        patch_size: Size of square patches
        stride: Stride for sliding window
    
    Returns:
        List of tuples: (image_patch, mask_patch, top, left)
    """
    height, width = image.shape[:2]
    patches = []
    
    # Sliding window
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            # Extract patches
            img_patch = image[top:top + patch_size, left:left + patch_size]
            mask_patch = mask[top:top + patch_size, left:left + patch_size]
            
            patches.append((img_patch, mask_patch, top, left))
    
    return patches


def crop_drive_dataset(
    input_dir: str,
    output_dir: str,
    patch_size: int = 32,
    overlap_rate: float = 0.5,
    image_start_idx: int = 21,
    image_end_idx: int = 36
):
    """
    Crop DRIVE dataset into patches.
    
    Args:
        input_dir: Path to DRIVE dataset directory (e.g., ./DRIVE/training)
        output_dir: Path to output directory for patches
        patch_size: Size of square patches
        overlap_rate: Overlap rate in [0, 1)
        image_start_idx: Starting image index (default: 21 for training)
        image_end_idx: Ending image index (default: 40 for training)
    """
    # Create output directories
    output_path = Path(output_dir)
    images_output = output_path / 'images'
    masks_output = output_path / 'masks'
    images_output.mkdir(parents=True, exist_ok=True)
    masks_output.mkdir(parents=True, exist_ok=True)
    
    # Calculate stride
    stride = calculate_stride(patch_size, overlap_rate)
    
    print(f"Cropping DRIVE dataset:")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Overlap rate: {overlap_rate:.2%}")
    print(f"  Stride: {stride}")
    print(f"  Images: {image_start_idx} to {image_end_idx}")
    print()
    
    # Subdirectories
    images_subdir = 'images'
    masks_subdir = '1st_manual'
    
    # Metadata
    metadata = {
        'patch_size': patch_size,
        'overlap_rate': overlap_rate,
        'stride': stride,
        'image_start_idx': image_start_idx,
        'image_end_idx': image_end_idx,
        'patches': []
    }
    
    patch_counter = 0
    total_images = image_end_idx - image_start_idx + 1
    
    # Process each image
    for img_idx in tqdm(range(image_start_idx, image_end_idx + 1), 
                        desc="Processing images", 
                        total=total_images):
        
        # Load image
        img_path = os.path.join(input_dir, images_subdir, f'{img_idx}_training.tif')
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = np.array(Image.open(img_path))
        
        # Load mask
        mask_path = os.path.join(input_dir, masks_subdir, f'{img_idx}_manual1.gif')
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found: {mask_path}")
            continue
        
        mask = np.array(Image.open(mask_path).convert('RGB'))
        
        # Extract patches
        patches = extract_patches(image, mask, patch_size, stride)
        
        # Save patches
        for img_patch, mask_patch, top, left in patches:
            # Generate patch filename
            patch_filename = f'patch_{patch_counter:05d}'
            
            # Save image patch
            img_patch_path = images_output / f'{patch_filename}_img.png'
            Image.fromarray(img_patch).save(img_patch_path)
            
            # Save mask patch
            mask_patch_path = masks_output / f'{patch_filename}_mask.png'
            Image.fromarray(mask_patch).save(mask_patch_path)
            
            # Add to metadata
            metadata['patches'].append({
                'patch_id': patch_counter,
                'source_image': img_idx,
                'top': int(top),
                'left': int(left),
                'image_file': f'{patch_filename}_img.png',
                'mask_file': f'{patch_filename}_mask.png'
            })
            
            patch_counter += 1
    
    # Save metadata
    metadata['total_patches'] = patch_counter
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nCropping complete!")
    print(f"  Total patches: {patch_counter}")
    print(f"  Patches per image: {patch_counter / total_images:.1f}")
    print(f"  Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Crop DRIVE dataset into patches with sliding window'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to DRIVE dataset directory (e.g., ./DRIVE/training)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to output directory for patches'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=32,
        help='Size of square patches (default: 32)'
    )
    parser.add_argument(
        '--overlap_rate',
        type=float,
        default=0.5,
        help='Overlap rate in [0, 1) (default: 0.5 for 50%% overlap)'
    )
    parser.add_argument(
        '--image_start_idx',
        type=int,
        default=21,
        help='Starting image index (default: 21 for training set)'
    )
    parser.add_argument(
        '--image_end_idx',
        type=int,
        default=40,
        help='Ending image index (default: 40 for training set)'
    )
    
    args = parser.parse_args()
    
    crop_drive_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap_rate=args.overlap_rate,
        image_start_idx=args.image_start_idx,
        image_end_idx=args.image_end_idx
    )


if __name__ == '__main__':
    main()
