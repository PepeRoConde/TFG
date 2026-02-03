import matplotlib.pyplot as plt
import numpy as np

def prediction_mask_plot(original_img, mask, prediction, output_path, dice_score, threshold=0.5):
    """
    Create a 4-panel comparison plot.
    
    Args:
        original_img: Original RGB image (H, W, 3)
        mask: Ground truth binary mask (H, W)
        prediction: Model prediction (H, W), values in [0, 1]
        output_path: Path to save the plot
        dice_score: Dice coefficient to display
        threshold: Threshold for binarizing predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. Original Image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. Ground Truth Mask
    axes[0, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Mask', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Prediction
    axes[1, 0].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Prediction (Dice: {dice_score:.4f})', fontsize=14)
    axes[1, 0].axis('off')
    
    # 4. Prediction with Mask Overlay (Red=Miss, Green=Hit)
    pred_binary = (prediction > threshold).astype(np.float32)
    
    # Create RGB overlay
    overlay = np.zeros((*pred_binary.shape, 3), dtype=np.float32)
    
    # True Positives: Green
    tp = (pred_binary == 1) & (mask == 1)
    overlay[tp] = [0, 1, 0]
    
    # True Negatives: Black (background)
    tn = (pred_binary == 0) & (mask == 0)
    overlay[tn] = [0, 0, 0]
    
    # False Positives: Red
    fp = (pred_binary == 1) & (mask == 0)
    overlay[fp] = [1, 0, 0]
    
    # False Negatives: Blue
    fn = (pred_binary == 0) & (mask == 1)
    overlay[fn] = [0, 0, 1]
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        'Overlay (Green=TP, Red=FP, Blue=FN)', 
        fontsize=14
    )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    plt.close()
