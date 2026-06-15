# inspirado en https://www.reddit.com/r/Python/comments/cpymni/7_lines_of_python_code_to_show_your_webcam_in_a/?tl=es-es
import argparse
from src.utils import load_model, cargar_config_yaml, get_device, imagenet_labels
import torch
import cv2
import PySimpleGUI as sg
import numpy as np
from torchvision.transforms import Resize
from time import sleep

parser = argparse.ArgumentParser(
    prog="Demo camara atencion", description="Ver la segmentacion emergente en directo"
)

parser.add_argument("pesos_red")
parser.add_argument("directorio")
parser.add_argument("--resize", default=10, type=int)
parser.add_argument(
    "--resolution",
    default=1,
    type=int,
    help="Resolucion extra (1 = original, maximo = tamano_token)",
)
parser.add_argument(
    "--capa",
    default=-1,
    type=int,
    help="Capa de la que mostrar la atención (-1 para la última)",
)
parser.add_argument(
    "--sleep",
    default=0.01,
    type=float,
    help="Tiempo a dormir entre frames (en segundos) para reducir uso de CPU/GPU",
)

args = parser.parse_args()
config = cargar_config_yaml(args.pesos_red, args.directorio)
tamano_patch = config["tamano_patch"]
tamano_token = config["tamano_token"]
if args.resolution == -1:
    args.resolution = tamano_token
args.resolution = max(1, min(args.resolution, tamano_token))

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

labels = imagenet_labels()

print(
    f"        vamos a ver la capa {args.capa if args.capa >= 0 else num_layers - 1} con {num_heads} cabezas"
)


window = sg.Window(
    "Demo cámara atención",
    [[sg.Image(filename="", key="image")]],
    location=(800, 400),
)

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
        og_img_tensor = Resize((tamano_patch, tamano_patch))(img_tensor)
        stride = tamano_token // args.resolution
        max_position = (args.resolution - 1) * stride
        required_size = max_position + tamano_patch
        if required_size > tamano_patch and args.resolution > 1:
            # print(f"Padding required: {required_size - tamano_patch}. Resizing to {required_size}x{required_size} from {tamano_patch}x{tamano_patch} due to resolution {args.resolution}, stride {stride}, and token size {tamano_token}.")
            pad_amount = required_size - tamano_patch
            img_tensor = torch.nn.functional.pad(
                og_img_tensor, (0, pad_amount, 0, pad_amount)
            ).to(get_device())
        else:
            img_tensor = og_img_tensor.to(get_device())

        with torch.no_grad():
            layer_idx = args.capa if args.capa >= 0 else num_layers - 1
            num_patches = tamano_patch // tamano_token
            fine_grained_size = num_patches * args.resolution
            attentions = np.zeros(
                (num_heads, fine_grained_size, fine_grained_size), dtype=np.float32
            )

            # TODO: check img_tensor shape and model input requirements, maybe need to add batch dimension or permute dimensions
            print(
                f"Salida del modelo: {labels[modelo(og_img_tensor.to(get_device())).cpu().numpy().argmax()]}"
            )
            # >>> img_tensor shape: torch.Size([1, 3, 230, 230])
            # lo cual es raro porque el tamano_patch es 224 ¿el padding lo hace 230? no deberia
            for di in range(args.resolution):
                for dj in range(args.resolution):
                    start_i, start_j = di * stride, dj * stride
                    subimage = img_tensor[
                        :,
                        :,
                        start_i : start_i + tamano_patch,
                        start_j : start_j + tamano_patch,
                    ]
                    attn = modelo.get_last_selfattention(
                        subimage, layer=layer_idx
                    )  # (1, heads, tokens, tokens)
                    num_tokens = attn.shape[-1]
                    grid_size = int(round((num_tokens - 1) ** 0.5))
                    att_map = (
                        attn[0, :, 0, 1:]
                        .reshape(num_heads, grid_size, grid_size)
                        .cpu()
                        .numpy()
                    )
                    attentions[:, di :: args.resolution, dj :: args.resolution] = (
                        att_map
                    )

            w, h = (
                int(fine_grained_size * num_heads / 2 * args.resize),
                int(fine_grained_size * 2 * args.resize),
            )
            attentions = np.block(
                [[*attentions[: num_heads // 2]], [*attentions[num_heads // 2 :]]]
            )  # asume numero de cabezas par

            att = attentions - attentions.min()
            att = att / (att.max() + 1e-8)
            att = (att * 255).astype(np.uint8)
            att = cv2.resize(att, (w, h), interpolation=cv2.INTER_NEAREST)
            att = cv2.flip(att, 1)
            att = cv2.applyColorMap(att, cv2.COLORMAP_PINK)
            imgbytes = cv2.imencode(".png", att)[1].tobytes()

            window["image"].update(data=imgbytes, size=(w, h))
            sleep(args.sleep)
except KeyboardInterrupt:
    cap.release()
    window.close()
    print("\nBoas tardes.          (:'D)")
finally:
    cap.release()
    window.close()
