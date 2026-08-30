import torch
import argparse
from tqdm import tqdm
from src.data import ImagenetDataset, RFMiDDataset, Online_Dataset
from src.utils import load_model, cargar_config_yaml, get_device, compute_auc

parser = argparse.ArgumentParser(
    prog="Imagenet Evaluation",
    description="Calcular Acc@1 y Acc@5 para una red con la posibilidad de invertir el orden",
)
parser.add_argument("pesos_red")
parser.add_argument("directorio")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--toggle_order", action="store_true")

args = parser.parse_args()

config = cargar_config_yaml(args.pesos_red, args.directorio)
device = get_device()
if args.toggle_order:
    order = "second" if config["order"] == "first" else "first"
else:
    order = config["order"]

modelo = load_model(
    weights_path=args.pesos_red,
    arch=config["arch"],
    patch_size=config["tamano_patch"],
    token_size=config["tamano_token"],
    num_classes=config["num_classes"],
    order=order,
    shared_u=config.get("shared_u", False),
    shared_dict=config.get("shared_dict", False),
).to(device)

dataset_name = config["dataset"]

if dataset_name == "imagenet":
    # ImagenetDataset.__len__ = lambda self: 100
    test_dataset = ImagenetDataset(aumento_datos=False, split="validation")
elif dataset_name == "rfmid":
    test_dataset = RFMiDDataset(
        data_dir="data/RFMiD/Test_Set",
        aumento_datos=False,
        tamano_patch=config["tamano_patch"],
        total_epochs=config["epochs"],
    )
elif dataset_name == "online":
    test_dataset = Online_Dataset(aumento_datos=False, split="validation")


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False
)

modelo.eval()

if dataset_name == "imagenet":
    correct_top1 = torch.tensor(0, device=device)
    correct_top5 = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluando", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = modelo(imgs)

            labels_idx = labels.argmax(dim=1) if labels.dim() > 1 else labels
            top1 = outputs.argmax(dim=1)
            correct_top1 += (top1 == labels_idx).sum()
            top5 = outputs.topk(5, dim=1).indices
            correct_top5 += (top5 == labels_idx.unsqueeze(1)).any(dim=1).sum()

            total += labels_idx.size(0)

    acc1 = (correct_top1.float() / total.float()).item()
    acc5 = (correct_top5.float() / total.float()).item()

    print(f"Acc@1: {acc1:.4f}")
    print(f"Acc@5: {acc5:.4f}")

elif dataset_name in {"rfmid", "online"}:
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluando", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = modelo(imgs)

            labels_idx = labels.argmax(dim=1) if labels.dim() > 1 else labels
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_idx).sum()
            total += labels_idx.size(0)

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels_idx.detach().cpu())

    acc = (correct.float() / total.float()).item()

    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        auc = compute_auc(all_targets, all_outputs)
    else:
        auc = float("nan")

    print(f"Acc: {acc:.4f}")
    print(f"AUC: {auc:.4f}")

else:
    raise ValueError(f"Dataset no soportado para evaluacion: {dataset_name}")
