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
args = parser.parse_args()

config = cargar_config_yaml(args.pesos_red, args.directorio)

modelo = load_model(
    weights_path=args.pesos_red,
    arch=config["arch"],
    patch_size=config["tamano_patch"],
    token_size=config["tamano_token"],
    order=config.get("order", "first"),
    shared_u=config.get("shared_u", False),
    shared_dict=config.get("shared_dict", False),
).to(get_device())

window = sg.Window(
    "Aplicación Demo - Integración OpenCV",
    [
        [sg.Image(filename="", key="image")],
    ],
    location=(800, 400),
)
cap = cv2.VideoCapture(1)  # Configura la cámara como dispositivo de captura
while True:
    event, values = window.Read(timeout=20, timeout_key="timeout")
    if event is None:
        break

    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.Tensor(img_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
    # TODO: usar tamano_patch
    img_tensor = Resize((224, 224))(img_tensor).to(get_device())

    with torch.no_grad():
        # TODO: sacar cual es la ultima capa
        attn = modelo.get_last_selfattention(img_tensor, layer=3)

        num_tokens = attn.shape[-1]
        grid_size = int(round((num_tokens - 1) ** 0.5))
        if grid_size * grid_size != (num_tokens - 1):
            raise ValueError(f"Token grid is not square: N={num_tokens}")

        nh = attn.shape[1]
        attentions = attn[0, :, 0, 1:].reshape(nh, grid_size, grid_size).cpu().numpy()
        attentions = np.block([*attentions])

        att = attentions - attentions.min()
        att = att / (att.max() + 1e-8)
        att = (att * 255).astype(np.uint8)
        att = cv2.resize(att, (168 * 10, 28 * 10), interpolation=cv2.INTER_NEAREST)
        att = cv2.applyColorMap(att, cv2.COLORMAP_PINK)
        imgbytes = cv2.imencode(".png", att)[1].tobytes()

        window["image"].update(data=imgbytes)
