# Video to Attention Map converter
# Processes an MP4 video and generates attention maps as output video
import argparse
from src.utils import load_model, cargar_config_yaml, get_device, imagenet_labels
import torch
import cv2
import numpy as np
from torchvision.transforms import Resize
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="Video 2 Attention",
    description="Procesar un vídeo y generar mapas de atención",
)

parser.add_argument("pesos_red", help="Ruta a los pesos de la red")
parser.add_argument("directorio", help="Directorio de configuración")
parser.add_argument("video_path", help="Ruta al archivo de vídeo MP4")
parser.add_argument(
    "--output",
    default=None,
    help="Ruta del vídeo de salida (por defecto: video_atention.mp4)",
)
parser.add_argument("--resize", default=10, type=int, help="Factor de redimensión")
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
    "--batch_size",
    default=4,
    type=int,
    help="Número de frames a procesar en batch (default: 4)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Mostrar información de cada frame procesado",
)

args = parser.parse_args()

# Validar que el vídeo existe
if not os.path.exists(args.video_path):
    print(f"Error: El archivo de vídeo '{args.video_path}' no existe")
    exit(1)

# Cargar configuración y modelo
config = cargar_config_yaml(args.pesos_red, args.directorio)
tamano_patch = config["tamano_patch"]
tamano_token = config["tamano_token"]

if args.resolution == -1:
    args.resolution = tamano_token
args.resolution = max(1, min(args.resolution, tamano_token))

device = get_device()

modelo = load_model(
    weights_path=args.pesos_red,
    arch=config["arch"],
    patch_size=config["tamano_patch"],
    token_size=config["tamano_token"],
    num_classes=config.get("num_classes", 2),
    order=config.get("order", "first"),
    shared_u=config.get("shared_u", False),
    shared_dict=config.get("shared_dict", False),
).to(device)

num_layers = modelo.transformer.depth
num_heads = modelo.transformer.heads

labels = imagenet_labels()

print(
    f"Usando capa {args.capa if args.capa >= 0 else num_layers - 1} con {num_heads} cabezas"
)

# Abrir vídeo de entrada
cap = cv2.VideoCapture(args.video_path)

if not cap.isOpened():
    print(f"Error: No se pudo abrir el vídeo '{args.video_path}'")
    exit(1)

# Obtener propiedades del vídeo
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Vídeo de entrada: {args.video_path}")
print(f"  Resolución: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total de frames: {total_frames}")
print(f"  Batch size: {args.batch_size}")

# Calcular tamaño de salida
stride = tamano_token // args.resolution
num_patches = tamano_patch // tamano_token
fine_grained_size = num_patches * args.resolution
output_w = int(fine_grained_size * num_heads / 2 * args.resize)
output_h = int(fine_grained_size * 2 * args.resize)

print(f"  Tamaño de salida: {output_w}x{output_h}")

# Configurar escritor de vídeo de salida
if args.output is None:
    video_base = os.path.basename(args.video_path)
    video_name, video_ext = os.path.splitext(video_base)
    args.output = f"{video_name}_attention.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, (output_w, output_h))

if not out.isOpened():
    print(f"Error: No se pudo crear el vídeo de salida '{args.output}'")
    cap.release()
    exit(1)

print(f"Guardando vídeo en: {args.output}\n")

# ImageNet normalization values (from ImagenetDataset.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_imagenet(img_tensor):
    """Aplica normalización de ImageNet: (x - mean) / std"""
    return (img_tensor - IMAGENET_MEAN) / IMAGENET_STD


def process_batch(batch_padded_tensors, batch_og_tensors, batch_frames_idx, layer_idx):
    """Procesa un batch de frames y escribe los resultados"""
    if len(batch_padded_tensors) == 0:
        return

    with torch.no_grad():
        batch_size_actual = len(batch_padded_tensors)

        # Concatenar batch en una tensor
        img_tensor_batch = torch.cat(
            batch_padded_tensors, dim=0
        )  # (batch_size, 3, H, W)
        og_img_tensor_batch = torch.cat(
            batch_og_tensors, dim=0
        )  # (batch_size, 3, tamano_patch, tamano_patch)

        # Predicciones en batch
        predictions = None
        if args.verbose:
            preds = modelo(og_img_tensor_batch.to(device)).cpu().numpy().argmax(axis=1)
            predictions = preds

        # Procesar atención para cada frame del batch
        for batch_idx in range(batch_size_actual):
            num_patches = tamano_patch // tamano_token
            fine_grained_size = num_patches * args.resolution
            attentions = np.zeros(
                (num_heads, fine_grained_size, fine_grained_size), dtype=np.float32
            )

            if args.verbose and predictions is not None:
                frame_idx = batch_frames_idx[batch_idx]
                print(f"\n  Frame {frame_idx}: {labels[predictions[batch_idx]]}")

            for di in range(args.resolution):
                for dj in range(args.resolution):
                    start_i, start_j = di * stride, dj * stride
                    subimage = img_tensor_batch[
                        batch_idx : batch_idx + 1,
                        :,
                        start_i : start_i + tamano_patch,
                        start_j : start_j + tamano_patch,
                    ]
                    attn = modelo.get_last_selfattention(subimage, layer=layer_idx)
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

            # Visualizar atención
            attentions = np.block(
                [[*attentions[: num_heads // 2]], [*attentions[num_heads // 2 :]]]
            )

            att = attentions - attentions.min()
            att = att / (att.max() + 1e-8)
            att = (att * 255).astype(np.uint8)
            att = cv2.resize(att, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
            att = cv2.flip(att, 1)
            att = cv2.applyColorMap(att, cv2.COLORMAP_PINK)

            # Escribir frame en vídeo de salida
            out.write(att)


# Procesar frames en batch
frame_count = 0
batch_padded_tensors = []
batch_og_tensors = []
batch_frames_idx = []

try:
    layer_idx = args.capa if args.capa >= 0 else num_layers - 1

    with tqdm(total=total_frames, desc="Procesando frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Procesar último batch si queda algo
                if len(batch_padded_tensors) > 0:
                    process_batch(
                        batch_padded_tensors,
                        batch_og_tensors,
                        batch_frames_idx,
                        layer_idx,
                    )
                break

            frame_count += 1

            # Convertir frame a tensor
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.Tensor(img_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
            og_img_tensor = Resize((tamano_patch, tamano_patch))(img_tensor)
            og_img_tensor = normalize_imagenet(og_img_tensor)

            # Aplicar padding si es necesario
            max_position = (args.resolution - 1) * stride
            required_size = max_position + tamano_patch
            if required_size > tamano_patch and args.resolution > 1:
                pad_amount = required_size - tamano_patch
                img_tensor = torch.nn.functional.pad(
                    og_img_tensor, (0, pad_amount, 0, pad_amount)
                ).to(device)
            else:
                img_tensor = og_img_tensor.to(device)

            # Acumular en batch
            batch_padded_tensors.append(img_tensor)
            batch_og_tensors.append(og_img_tensor.to(device))
            batch_frames_idx.append(frame_count)

            # Procesar batch cuando alcanza el tamaño
            if len(batch_padded_tensors) == args.batch_size:
                process_batch(
                    batch_padded_tensors, batch_og_tensors, batch_frames_idx, layer_idx
                )
                batch_padded_tensors = []
                batch_og_tensors = []
                batch_frames_idx = []

            pbar.update(1)

    print(f"\n\nProceso completado: {frame_count} frames procesados")
    print(f"Vídeo guardado en: {args.output}")

except KeyboardInterrupt:
    print("\n\nProceso interrumpido por el usuario")
except Exception as e:
    print(f"\n\nError durante el procesamiento: {e}")
    import traceback

    traceback.print_exc()
finally:
    cap.release()
    out.release()
    print("Recursos liberados")
