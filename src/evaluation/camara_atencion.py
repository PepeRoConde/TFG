# inspirado en https://www.reddit.com/r/Python/comments/cpymni/7_lines_of_python_code_to_show_your_webcam_in_a/?tl=es-es
import argparse
from src.utils import load_model, cargar_config_yaml, get_device
import torch
import cv2
import PySimpleGUI as sg
import numpy as np
from torchvision.transforms import Resize

parser = argparse.ArgumentParser(
    prog="Demo camara atencion", description="Ver la segmentacion emergente en directo"
)

parser.add_argument("pesos_red")
parser.add_argument("directorio")
parser.add_argument("--resize", default=10, type=int)
parser.add_argument(
    "--capa",
    default=-1,
    type=int,
    help="Capa de la que mostrar la atención (-1 para la última)",
)

args = parser.parse_args()
config = cargar_config_yaml(args.pesos_red, args.directorio)

modelo = load_model(
    weights_path=args.pesos_red,
    arch=config["arch"],
    patch_size=config["tamano_patch"],
    token_size=config["tamano_token"],
    num_classes=config.get("num_classes", 2),
    order=config.get("order", "first"),
    shared_u=config.get("shared_u", False),
    shared_dict=config.get("shared_dict", False),
).to(get_device())

num_layers = modelo.transformer.depth
num_heads = modelo.transformer.heads
print(
    f"        vamos a ver la capa {args.capa if args.capa >= 0 else num_layers - 1} con {num_heads} cabezas"
)


window = sg.Window(
    "Demo cámara atención",
    [[sg.Image(filename="", key="image")]],
    location=(800, 400),
)
# try:
#    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
#
# except:
cap = cv2.VideoCapture(0)
try:
    while True:
        event, values = window.read(timeout=10, timeout_key="timeout")
        if event is None:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.Tensor(img_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = Resize((config["tamano_patch"], config["tamano_patch"]))(
            img_tensor
        ).to(get_device())

        with torch.no_grad():
            attn = modelo.get_last_selfattention(
                img_tensor, layer=args.capa if args.capa >= 0 else num_layers - 1
            )  # (1, heads, tokens, tokens)
            num_tokens = attn.shape[-1]
            grid_size = int(round((num_tokens - 1) ** 0.5))
            w, h = (
                int(grid_size * num_heads / 2 * args.resize),
                int(grid_size * 2 * args.resize),
            )

            attentions = (
                attn[0, :, 0, 1:].reshape(num_heads, grid_size, grid_size).cpu().numpy()
            )
            attentions = np.block(
                [[*attentions[: num_heads // 2]], [*attentions[num_heads // 2 :]]]
            )  # asume numero de cabezas par

            att = attentions - attentions.min()
            att = att / (att.max() + 1e-8)
            att = (att * 255).astype(np.uint8)
            att = cv2.resize(att, (w, h), interpolation=cv2.INTER_NEAREST)
            att = cv2.flip(att, 1)
            att = cv2.applyColorMap(att, cv2.COLORMAP_PARULA)
            imgbytes = cv2.imencode(".png", att)[1].tobytes()

            window["image"].update(data=imgbytes, size=(w, h))
finally:
    cap.release()
    window.close()
