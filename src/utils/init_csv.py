import csv
from pathlib import Path


class CSVLogger:
    """Utility class for reading and writing CSV log files."""

    def __init__(self, path):
        self.path = Path(path)
        self.fieldnames = [
            "epoch",
            "loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
            "train_auc",
            "val_auc",
        ]

    def read(self):
        """Read CSV file and return list of dictionaries."""
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        rows = []
        with open(self.path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        return rows

    def init_write(self):
        """Initialize CSV file for writing, returns file object and writer."""
        file_exists = self.path.exists()
        f = open(self.path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
        if not file_exists:
            writer.writeheader()
        return f, writer


def init_csv(file_name, args):
    """Legacy function for backward compatibility. Returns file object and writer."""
    path = args.runs_dir + "/" + file_name + ".log"
    logger = CSVLogger(path)
    return logger.init_write()
