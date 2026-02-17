import torch
import numpy as np

def compute_auc(output, target):
    """
    Computes AUC-ROC score for binary classification.
    
    Args:
        output: model logits/scores of shape (batch_size, num_classes) or (batch_size,)
        target: ground truth labels of shape (batch_size,)
    
    Returns:
        auc_score: float, AUC-ROC value in range [0, 1]
    """
    with torch.no_grad():
        # Handle different output shapes
        if output.dim() == 2:
            # Multi-class output, take softmax probabilities for positive class
            scores = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        else:
            # Already a 1D tensor of scores
            scores = output.cpu().numpy()
        
        targets = target.cpu().numpy()
        
        # Handle gaussian labels (soft labels) by rounding
        if targets.dtype == np.float32 or targets.dtype == np.float64:
            targets = np.round(targets).astype(np.int32)
        
        # Simple AUC calculation
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(targets, scores)
            return torch.tensor(auc)
        except ImportError:
            # Fallback: manual AUC calculation
            return _compute_auc_manual(scores, targets)

def _compute_auc_manual(scores, targets):
    """Manual AUC-ROC computation without sklearn."""
    # Sort by scores
    sorted_indices = np.argsort(-scores)
    sorted_targets = targets[sorted_indices]
    
    # Count positives and negatives
    n_pos = np.sum(targets == 1)
    n_neg = np.sum(targets == 0)
    
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(0.5)  # Undefined for single-class
    
    # Calculate AUC
    tp = np.cumsum(sorted_targets == 1)
    fp = np.cumsum(sorted_targets == 0)
    
    # TPR and FPR at each threshold
    tpr = tp / n_pos
    fpr = fp / n_neg
    
    # Calculate AUC as area under ROC curve
    auc = np.trapz(tpr, fpr)
    return torch.tensor(auc)
