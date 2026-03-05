#!/usr/bin/env python3
"""
Sliding Window Inference with Vision Transformer (ViT)
Performs patch-wise inference on an image with stride=1 and outputs a same-size result.
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F

from src.models.architectures import *
from src.utils.checkpoint import *
from src.plots.prediction_mask_plot import prediction_mask_plot
from src.utils import load_model, cargar_config_yaml, get_device


def find_gmm_threshold(output):
    # Build histogram
    counts, bin_edges = np.histogram(output.flatten(), bins=256, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Repeat each bin center by its count to create a weighted sample
    samples = np.repeat(bin_centers, counts).reshape(-1, 1)

    # Fit 2-component GMM
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
    gmm.fit(samples)

    means = sorted(gmm.means_.flatten())
    threshold = (means[0] + means[1]) / 2.0

    print(f"GMM means: {means[0]:.4f}, {means[1]:.4f}")
    print(f"GMM threshold: {threshold:.4f}")

    return threshold


def preprocess_image(image_path, patch_size):
    """
    Load and preprocess the image.
    Adds padding of patch_size//2 on all sides.
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # Calculate padding
    pad = patch_size // 2

    # Pad the image (top, bottom, left, right)
    padded = np.pad(img_array, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    return padded, img_array.shape[:2], img_array


def load_mask(mask_path, expected_shape):
    """Load and validate the ground truth mask."""
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask_array = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # Threshold to binary
    mask_array = (mask_array > 0.5).astype(np.float32)

    if mask_array.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask_array.shape} does not match image shape {expected_shape}"
        )

    return mask_array


def sliding_window_inference_batched(
    model, padded_image, patch_size, device, batch_size=256
):
    """
    Perform sliding window inference with stride=1 using batched processing.

    Args:
        model: The ViT model
        padded_image: Padded image array (H+2*pad, W+2*pad, C)
        patch_size: Size of the patch
        device: torch device
        batch_size: Number of patches to process in parallel

    Returns:
        output_image: Same size as original image (H, W)
    """
    pad = patch_size // 2
    H, W, C = padded_image.shape

    # Original image dimensions (without padding)
    orig_H = H - 2 * pad
    orig_W = W - 2 * pad

    # Initialize output
    output = np.zeros((orig_H, orig_W), dtype=np.float32)

    print(
        f"Processing {orig_H}x{orig_W} image with {patch_size}x{patch_size} patches..."
    )
    print(f"Total patches to process: {orig_H * orig_W}")
    print(f"Batch size: {batch_size}")

    # Process row by row with batching
    for i in tqdm(range(orig_H), desc="Processing rows"):
        # Collect all patches for this row
        patches = []
        for j in range(orig_W):
            # Extract patch centered at (i, j) in the original image
            patch = padded_image[i : i + patch_size, j : j + patch_size, :]
            patches.append(patch)

        # Process patches in batches
        row_output = []
        for batch_start in range(0, orig_W, batch_size):
            batch_end = min(batch_start + batch_size, orig_W)
            batch_patches = patches[batch_start:batch_end]

            # Stack patches into a batch: (B, C, H, W)
            batch_tensor = torch.stack(
                [torch.from_numpy(p).permute(2, 0, 1) for p in batch_patches]
            ).to(device)

            # Normalize to [0, 1] if needed (assuming input was [0, 255])
            if batch_tensor.max() > 1.0:
                batch_tensor = batch_tensor / 255.0

            # Inference
            with torch.no_grad():
                logits = model(batch_tensor)  # Shape: (B, num_classes)

                # Apply softmax to convert logits to probabilities
                probs = F.softmax(logits, dim=1)  # Shape: (B, num_classes)

                # Extract vessel probability (positive class)
                # For 'vainilla' and 'gaussian': class 1 is vessel
                # For 'multiple': we need to aggregate or use a specific scale

                if probs.shape[1] == 2:
                    # Binary classification (vainilla/gaussian mode)
                    # Class 0 = background, Class 1 = vessel
                    vessel_probs = probs[:, 1]  # Shape: (B,)
                else:
                    # Multi-scale classification (multiple mode)
                    # Option 1: Take mean across all scales
                    vessel_probs = probs.mean(dim=1)  # Shape: (B,)

                    # Option 2: Take max across scales (most confident scale)
                    # vessel_probs = probs.max(dim=1)[0]  # Shape: (B,)

                    # Option 3: Use first/finest scale only
                    # vessel_probs = probs[:, 0]  # Shape: (B,)

                # Move to CPU and convert to numpy
                values = vessel_probs.cpu().numpy()
                row_output.extend(values)

        # Assign to output
        output[i, :] = row_output

    print("\nProcessing complete!")
    return output


def compute_dice_score(pred, target, threshold=0.5):
    """
    Compute Dice coefficient between prediction and target.

    Args:
        pred: Prediction array (H, W), values in [0, 1]
        target: Ground truth binary mask (H, W), values in {0, 1}
        threshold: Threshold to binarize predictions

    Returns:
        dice: Dice coefficient
    """
    # Binarize prediction
    pred_binary = (pred > threshold).astype(np.float32)

    # Compute Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    intersection = np.sum(pred_binary * target)
    union = np.sum(pred_binary) + np.sum(target)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = 2.0 * intersection / union
    return dice


def save_output(output, output_path):
    """Save the output as an image or numpy array."""
    output_path = Path(output_path)

    if output_path.suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
        # Normalize to [0, 255] for image saving
        output_normalized = (
            (output - output.min()) / (output.max() - output.min() + 1e-8) * 255
        ).astype(np.uint8)
        Image.fromarray(output_normalized).save(output_path)
        print(f"Saved output image to: {output_path}")
    elif output_path.suffix == ".npy":
        np.save(output_path, output)
        print(f"Saved output array to: {output_path}")
    else:
        # Default to .npy
        output_path = output_path.with_suffix(".npy")
        np.save(output_path, output)
        print(f"Saved output array to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sliding window inference with ViT on image patches"
    )
    parser.add_argument(
        "weights_path", type=str, help="Path to the model weights (.pth.tar file)"
    )
    parser.add_argument(
        "log_dir", type=str, help="Path to the metadata (e.g. data/runs/)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/DRIVE/test/images/40_training.tif",
        help="Path to the input image",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="data/DRIVE/test/1st_manual/40_manual1.gif",
        help="Path to the ground truth mask",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1,
        help="Threshold for binarizing predictions. If -1 is passed, then compute one based on 2 gmm (default).",
    )

    args = parser.parse_args()

    # Load configuration from yaml
    config = cargar_config_yaml(args.weights_path, args.log_dir)
    patch_size = config.get("tamano_patch")
    token_size = config.get("tamano_token")
    arch = config.get("arch", "CRATE_tiny2nd")  # Default to CRATE_tiny2nd if not found
    print(f"Architecture from config: {arch}")

    # Validate inputs

    if not Path(args.weights_path).exists():
        print(f"Error: Weights file not found: {args.weights_path}")
        sys.exit(1)

    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    if not Path(args.mask_path).exists():
        print(f"Error: Mask file not found: {args.mask_path}")
        sys.exit(1)

    # Get best available device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.weights_path, patch_size, token_size, arch)
    model.to(device)

    # Load and preprocess image
    print("Loading and preprocessing image...")
    padded_image, original_shape, original_img = preprocess_image(
        args.image_path, patch_size
    )
    print(f"Original image shape: {original_shape}")
    print(f"Padded image shape: {padded_image.shape}")

    # Load mask
    print("Loading mask...")
    mask = load_mask(args.mask_path, original_shape)
    print(f"Mask shape: {mask.shape}")

    # Perform inference
    output = sliding_window_inference_batched(
        model, padded_image, patch_size, device, args.batch_size
    )

    if args.threshold == -1.0:
        args.threshold = find_gmm_threshold(output)

    # Normalize output to [0, 1]
    output_normalized = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # Compute Dice score
    dice = compute_dice_score(output_normalized, mask, args.threshold)

    # Create plots directory if it doesn't exist
    plots_dir = Path(args.log_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get checkpoint name (without extension)
    checkpoint_name = Path(args.weights_path).stem

    # Save output
    output_path = plots_dir / f"{checkpoint_name}_complete_inference.png"
    save_output(output_normalized, str(output_path))

    # Create comparison plot
    comparison_path = plots_dir / f"{checkpoint_name}_comparison_plot.png"
    prediction_mask_plot(
        original_img,
        mask,
        output_normalized,
        str(comparison_path),
        dice,
        args.threshold,
    )

    # Print statistics
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Output shape:        {output.shape}")
    print(f"Output min:          {output.min():.4f}")
    print(f"Output max:          {output.max():.4f}")
    print(f"Output mean:         {output.mean():.4f}")
    print(f"Output std:          {output.std():.4f}")
    print(f"{'='*50}")
    print(f"DICE COEFFICIENT:    {dice:.4f}")
    print(f"umbral:    {args.threshold:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
