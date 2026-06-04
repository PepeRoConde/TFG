import yaml


def imagenet_labels(path="./data/imagenet_labels.yaml"):
    with open(path, "r") as f:
        labels = yaml.safe_load(f)
    return labels
