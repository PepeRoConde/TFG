import torch
import argparse
from tqdm import tqdm
from src.data.ImagenetDataset import ImagenetDataset
from src.utils import load_model, cargar_config_yaml, get_device

parser = argparse.ArgumentParser(
    prog="Embedding retrieval",
    description="Ver parches característicos de cada dimensión del embedding.",
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

# ImagenetDataset.__len__ = lambda self: 100
test_dataset = ImagenetDataset(aumento_datos=False, split="validation")

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False
)

correct_top1 = torch.tensor(0, device=device)
correct_top5 = torch.tensor(0, device=device)
total = torch.tensor(0, device=device)

modelo.eval()

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluando", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = modelo(imgs)

        top1 = outputs.argmax(dim=1)
        correct_top1 += (top1 == labels).sum()
        top5 = outputs.topk(5, dim=1).indices
        correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum()

        total += labels.size(0)

acc1 = (correct_top1.float() / total.float()).item()
acc5 = (correct_top5.float() / total.float()).item()

print(f"Acc@1: {acc1:.4f}")
print(f"Acc@5: {acc5:.4f}")
