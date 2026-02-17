import csv
from pathlib import Path

def init_csv(path):
    file_exists = Path(path).exists()
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "val_loss", "train_accuracy", "val_accuracy", "train_auc", "val_auc"])
    if not file_exists:
        writer.writeheader()
    return f, writer
