#!/usr/bin/env python3
"""
Diagnostic script to check:
1. What is the model actually predicting?
2. Is the dataset balanced?
3. Are the predictions correct?
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add your src to path if needed
# sys.path.insert(0, 'path/to/your/project')

from src.models.architectures import CRATE_tiny
from src.data.Offline_Dataset import Offline_Dataset


def analyze_model_predictions(weights_path, dataset_path, patch_size=48, num_samples=1000):
    """
    Analyze what the model is actually predicting.
    """
    print("="*60)
    print("MODEL PREDICTION ANALYSIS")
    print("="*60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CRATE_tiny(patch_size, 16, 2)
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset = Offline_Dataset(
        patches_dir=dataset_path,
        data_augmentation=False,
        label_mode='gaussian',
        total_epochs=1
    )
    
    print(f"\nDataset size: {len(dataset)} patches")
    print(f"Analyzing {min(num_samples, len(dataset))} samples...\n")
    
    # Collect predictions and labels
    all_logits = []
    all_probs = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            image, label = dataset[i]
            
            # Add batch dimension and move to device
            image = image.unsqueeze(0).to(device)
            
            # Get prediction
            logits = model(image)  # Shape: (1, 2)
            probs = F.softmax(logits, dim=1)  # Shape: (1, 2)
            pred_class = logits.argmax(dim=1).item()
            
            # Get true class from label
            if isinstance(label, torch.Tensor):
                true_class = label.argmax().item() if label.dim() > 0 else int(label.item())
            else:
                true_class = 1 if label > 0.5 else 0
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(true_class)
            all_predictions.append(pred_class)
    
    all_logits = np.concatenate(all_logits, axis=0)  # Shape: (N, 2)
    all_probs = np.concatenate(all_probs, axis=0)    # Shape: (N, 2)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Analysis
    print("="*60)
    print("DATASET BALANCE")
    print("="*60)
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(all_labels)
        cls_name = "Background" if cls == 0 else "Vessel"
        print(f"Class {cls} ({cls_name}): {count:5d} samples ({pct:5.2f}%)")
    
    print("\n" + "="*60)
    print("MODEL PREDICTIONS")
    print("="*60)
    unique, counts = np.unique(all_predictions, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / len(all_predictions)
        cls_name = "Background" if cls == 0 else "Vessel"
        print(f"Predicts {cls} ({cls_name}): {count:5d} times ({pct:5.2f}%)")
    
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Accuracy: {100 * (all_predictions == all_labels).mean():.2f}%")
    
    # Class-wise accuracy
    for cls in [0, 1]:
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = 100 * (all_predictions[mask] == cls).mean()
            cls_name = "Background" if cls == 0 else "Vessel"
            print(f"{cls_name} accuracy: {cls_acc:.2f}%")
    
    print("\n" + "="*60)
    print("PROBABILITY DISTRIBUTIONS")
    print("="*60)
    vessel_probs = all_probs[:, 1]  # Probability of vessel class
    print(f"Vessel probability stats:")
    print(f"  Min:    {vessel_probs.min():.4f}")
    print(f"  Max:    {vessel_probs.max():.4f}")
    print(f"  Mean:   {vessel_probs.mean():.4f}")
    print(f"  Median: {np.median(vessel_probs):.4f}")
    print(f"  Std:    {vessel_probs.std():.4f}")
    
    # Probability distribution for each true class
    for cls in [0, 1]:
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_vessel_probs = vessel_probs[mask]
            cls_name = "Background" if cls == 0 else "Vessel"
            print(f"\nVessel probability for true {cls_name} patches:")
            print(f"  Mean:   {cls_vessel_probs.mean():.4f}")
            print(f"  Median: {np.median(cls_vessel_probs):.4f}")
    
    print("\n" + "="*60)
    print("LOGITS STATISTICS")
    print("="*60)
    print(f"Logit[0] (background) - Mean: {all_logits[:, 0].mean():.4f}, Std: {all_logits[:, 0].std():.4f}")
    print(f"Logit[1] (vessel)     - Mean: {all_logits[:, 1].mean():.4f}, Std: {all_logits[:, 1].std():.4f}")
    
    # Check if model always predicts same class
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if len(np.unique(all_predictions)) == 1:
        print("⚠️  WARNING: Model ALWAYS predicts the same class!")
        print("    This is a collapsed model - it hasn't learned anything useful.")
    
    if all_predictions.mean() > 0.95 or all_predictions.mean() < 0.05:
        print("⚠️  WARNING: Model is heavily biased to one class!")
        print(f"    Predicts class {all_predictions[0]} {100*all_predictions.mean():.1f}% of the time")
    
    # Check if high accuracy is just from predicting majority class
    majority_class = np.bincount(all_labels).argmax()
    majority_baseline = 100 * (all_labels == majority_class).mean()
    actual_acc = 100 * (all_predictions == all_labels).mean()
    
    print(f"\nMajority class baseline: {majority_baseline:.2f}%")
    print(f"Model accuracy:          {actual_acc:.2f}%")
    
    if abs(actual_acc - majority_baseline) < 2:
        print("⚠️  WARNING: Model accuracy ≈ majority class baseline!")
        print("    The model is just predicting the most common class.")
    
    return {
        'labels': all_labels,
        'predictions': all_predictions,
        'probs': all_probs,
        'logits': all_logits
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_path', type=str, help='Path to model weights (.pth.tar)')
    parser.add_argument('dataset_path', type=str, help='Path to dataset (patches directory)')
    parser.add_argument('--patch_size', type=int, default=48, help='Patch size')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    results = analyze_model_predictions(
        args.weights_path,
        args.dataset_path,
        args.patch_size,
        args.num_samples
    )
