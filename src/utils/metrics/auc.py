import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(target, output):
    with torch.no_grad():
        target = target.cpu().numpy()
        output = torch.softmax(output, dim=1).cpu().numpy()

    present_classes = np.unique(target)

    if len(present_classes) < 2:
        return float("nan")

    output_filtered = output[:, present_classes]

    return roc_auc_score(
        target, output_filtered, multi_class="ovr", labels=list(present_classes)
    )
