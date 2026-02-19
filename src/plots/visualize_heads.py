import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.architectures import * 
from src.data.Online_Dataset import Online_Dataset
from src.utils import cargar_config_yaml, instantiate_model 


def get_device():
    """Get the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu') 

def crear_modelo(config, checkpoint):
    arch = config.get('arch')
    
    tamano_patch = config.get('tamano_patch')
    tamano_token = config.get('tamano_token')
    num_classes = config.get('num_classes', 2)  # Default 2 se non existe
    
    
    print(f"Creando modelo: {arch}")
    print(f"  tamano_patch: {tamano_patch}")
    print(f"  tamano_token: {tamano_token}")
    print(f"  num_classes: {num_classes}")
    
    # Use instantiate_model function
    try:
        modelo = instantiate_model(arch, tamano_patch, tamano_token, num_classes)
    except NotImplementedError:
        print(f"ERRO: Arquitectura '{arch}' non soportada")
        print(" Uso de instantiate_model - ver src/utils/instantiate_model.py para arquitecturas soportadas")
        sys.exit(1)
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Eliminar prefixo 'module.' se existe
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict
    modelo.load_state_dict(state_dict)
    modelo.eval()
    
    depth = modelo.transformer.depth 
    num_heads = modelo.transformer.heads
    
    print(f"Modelo cargado: depth={depth}, heads={num_heads}")
    
    return modelo, depth, num_heads


def cargar_imaxes(dataset_path, tamano_patch, num_images):
    """Cargar imaxes do dataset con metade positivas e metade negativas."""
    dataset = Online_Dataset(drive_dir=dataset_path, tamano_patch=tamano_patch, aumento_datos=False)
    
    imaxes = []
    etiquetas = []
    num_por_clase = num_images // 2
    counts = {1: 0, 0: 0}
    indices = np.random.permutation(len(dataset))
    i = 0

    while counts[1] < num_por_clase or counts[0] < num_por_clase:

        img, label = dataset[int(indices[i])]
        if counts[label] < num_por_clase:
            imaxes.append(img)
            etiquetas.append(label)
            counts[label] += 1
        i += 1
    
    imaxes = torch.stack(imaxes)
    print(f"Cargadas {len(imaxes)} imaxes ({counts[1]} positivas, {counts[0]} negativas)")
    return imaxes, etiquetas


def obter_mapas_atencion_cls(modelo, imaxes, indices_capas, num_heads):
    attention_matrices = {}
    qkv_values = {}

    def make_qkv_hook(layer_idx):
        def hook(module, input, output):
            qkv_values[f'layer_{layer_idx}'] = output.detach()  # [B, N, inner_dim]
        return hook

    hooks = []
    for layer_idx in indices_capas:
        try:
            attn_module = modelo.transformer.layers[layer_idx][0].fn
            hook = attn_module.qkv.register_forward_hook(make_qkv_hook(layer_idx))
            hooks.append(hook)
        except (AttributeError, IndexError) as e:
            print(f"Aviso: {e}")

    with torch.no_grad():
        _ = modelo(imaxes)

    for hook in hooks:
        hook.remove()

    for layer_idx in indices_capas:
        key = f'layer_{layer_idx}'
        if key in qkv_values:
            qkv = qkv_values[key]      # [B, N, inner_dim]
            B, N, inner_dim = qkv.shape
            D_h = inner_dim // num_heads

            # Reshape to heads: [B, N, H, D_h] -> [B, H, N, D_h]
            q = qkv.reshape(B, N, num_heads, D_h).permute(0, 2, 1, 3)

            # In CRATE, K = Q (symmetric), so dots are q @ q^T
            cls = q[:, :, 0:1, :]                          # [B, H, 1, D_h]
            scores = (q @ cls.transpose(-2, -1)).squeeze(-1) * (D_h ** -0.5)  # [B, H, N]
            attn_weights = torch.softmax(scores, dim=-1)   # [B, H, N]

            attention_matrices[key] = attn_weights
        else:
            print(f"Aviso: qkv para {key} non dispoñible")

    return attention_matrices

def obter_mapas_atencion(modelo, imaxes, indices_capas, num_heads):
    """Extraer mapas de atención das capas especificadas."""
    activations = {}
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[f'layer_{layer_idx}'] = output.detach()
        return hook
    
    # Rexistrar hooks
    hooks = []
    for layer_idx in indices_capas:
        layer = modelo.transformer.layers[layer_idx]
        hook = layer[0].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = modelo(imaxes)
    
    # Eliminar hooks
    for hook in hooks:
        hook.remove()
    
    # Procesar activacións para obter saídas das cabezas de atención
    results = {}
    for layer_idx in indices_capas:
        key = f'layer_{layer_idx}'
        if key in activations:
            act = activations[key]  # [B, N, D]
            B, N, D = act.shape
            
            # Reshape para separar cabezas: [B, N, D] -> [B, N, H, D_h] -> [B, H, N, D_h]
            act = act.reshape(B, N, num_heads, D // num_heads)
            act = act.permute(0, 2, 1, 3)
            
            results[key] = act
    
    return results


def visualizar(imaxes, mapas_atencion, indices_capas, num_heads_to_show, 
               tamano_token, output_path, indices_cabezas_por_capa, etiquetas=None):
    """
    Visualize attention heatmaps as per mathematical formulation.
    
    Each element (i,j) of the heatmap corresponds to the m-th component of A_k^ℓ
    where m = (i-1)·√n + j
    """
    num_imaxes = imaxes.shape[0]
    num_capas = len(indices_capas)
    
    # Columnas: imaxe orixinal + num_heads_to_show * num_capas
    num_cols = 1 + (num_heads_to_show * num_capas)
    
    fig, axes = plt.subplots(num_imaxes, num_cols, figsize=(num_cols * 3, num_imaxes * 3))
    
    if num_imaxes == 1:
        axes = axes.reshape(1, -1)
    
    # Calcular dimensións espaciais
    tamano_imaxe = imaxes.shape[-1]
    num_patches = tamano_imaxe // tamano_token
    
    print(f"Tamano imaxe: {tamano_imaxe}, Tamano token: {tamano_token}, Patches por lado: {num_patches}")
    print(f"\nÍndices de cabezas consistentes por capa:")
    for layer_idx in indices_capas:
        print(f"  Capa {layer_idx}: cabezas {indices_cabezas_por_capa[layer_idx]}")
    
    for img_idx in range(num_imaxes):
        col_idx = 0
        
        # Plotear imaxe orixinal
        ax = axes[img_idx, col_idx]
        img = imaxes[img_idx].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.axis('off')
        
        # Crear título con info da etiqueta
        if etiquetas is not None and img_idx < len(etiquetas):
            label = etiquetas[img_idx]
            if isinstance(label, torch.Tensor):
                label_val = label[1].item() if label.numel() == 2 else label.item()
            else:
                label_val = label
            
            color = 'green' if label_val >= 0.5 else 'red'
            rect = Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1, 
                     linewidth=4, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        col_idx += 1
        
        # Plotear cabezas de atención para cada capa
        for layer_idx in indices_capas:
            key = f'layer_{layer_idx}'
            
            if key not in mapas_atencion:
                print(f"Aviso: {key} non está en mapas_atencion")
                for _ in range(num_heads_to_show):
                    ax = axes[img_idx, col_idx]
                    ax.axis('off')
                    col_idx += 1
                continue
            
            # Get attention matrix [B, H, N]
            attn_matrix = mapas_atencion[key]  # [B, H, N]
            B, H, N = attn_matrix.shape
            
            # Usar os índices de cabezas pre-seleccionados para esta capa
            cabezas_seleccionadas = indices_cabezas_por_capa[layer_idx]
            
            for head_idx in cabezas_seleccionadas:
                ax = axes[img_idx, col_idx]
                
                try:
                    # Obter matriz de atención para esta cabeza e imaxe: [N]
                    head_attn = attn_matrix[img_idx, head_idx].cpu().numpy()  # [N]
                    
                    # Eliminar token CLS (primeiro elemento)
                    if N > 1:
                        spatial_attn = head_attn[1:]  # [N-1] - atención aos tokens espaciais
                        num_spatial_tokens = spatial_attn.shape[0]
                    else:
                        spatial_attn = head_attn
                        num_spatial_tokens = N
                    
                    # Reshape a grella espacial: √N × √N
                    actual_patches_per_side = int(np.sqrt(num_spatial_tokens))
                    
                    if actual_patches_per_side * actual_patches_per_side != num_spatial_tokens:
                        raise ValueError(f"Non se pode reshape {num_spatial_tokens} tokens a grella cadrada")
                    
                    # Reshape segundo a fórmula: elemento (i,j) corresponde a m-ésima componente
                    # onde m = (i-1)·√n + j
                    attn_heatmap = spatial_attn.reshape(actual_patches_per_side, actual_patches_per_side)
                    
                    # Upsample ao tamano da imaxe para mellor visualización
                    from scipy.ndimage import zoom
                    zoom_factor = tamano_imaxe / actual_patches_per_side
                    attn_heatmap_up = zoom(attn_heatmap, zoom_factor, order=1)
                    
                    # Plotear heatmap
                    im = ax.imshow(attn_heatmap_up, cmap='copper', interpolation='nearest')
                    ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=10)
                    ax.axis('off')
                    
                    # Engadir colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    
                except Exception as e:
                    print(f"    Erro para Imaxe {img_idx}, Capa {layer_idx}, Cabeza {head_idx}: {e}")
                    ax.text(0.5, 0.5, f'Erro\n{str(e)[:30]}', ha='center', va='center', fontsize=8)
                    ax.axis('off')
                
                col_idx += 1
    
    plt.tight_layout()
    
    if output_path:
        # Crear directorio se non existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualización gardada en: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualizar cabezas de atención de CRATE')
    
    parser.add_argument('checkpoint', type=str, help='Ruta ao checkpoint')
    parser.add_argument('logs_dir', type=str, default='data/runs/', help='Path to the metadata (e.g. data/runs/)')
    parser.add_argument('-cabezas', type=int, default=3,
                        help='Número de cabezas a visualizar por capa')
    parser.add_argument('-capas', '--num-last-layers', type=int, default=1,
                        help='Número de últimas capas a visualizar')
    parser.add_argument('-imaxes', type=int, default=2,
                        help='Número de imaxes a visualizar')
    parser.add_argument('--mode', type=str, default='vainilla', choices=['vainilla', 'cls'],
                        help='Modo de extracción: vainilla (saídas de camadas) o cls (atención desde token CLS)')
    
    args = parser.parse_args()
    
    # Get best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Cargar configuración dende YAML
    config = cargar_config_yaml(args.checkpoint, args.logs_dir)
    
    # Cargar checkpoint
    print(f"Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    print("✓ Checkpoint cargado")
    
    # Crear modelo
    modelo, depth, num_heads = crear_modelo(config, checkpoint)
    modelo = modelo.to(device)
    
    # Determinar capas a visualizar
    indices_capas = []
    
    indices_capas.append(0)
    
    # Engadir últimas n capas
    for i in range(args.num_last_layers):
        layer_idx = depth - 1 - i
        if layer_idx not in indices_capas and layer_idx >= 0:
            indices_capas.append(layer_idx)
    
    indices_capas.sort()
    print(f"Visualizando capas: {indices_capas}")
    
    # PRE-SELECCIONAR índices de cabezas para cada capa (consistente entre todas as imaxes)
    print(f"\nPre-seleccionando {args.cabezas} cabezas por capa...")
    indices_cabezas_por_capa = {}
    for layer_idx in indices_capas:
        selected = np.random.choice(num_heads, min(args.cabezas, num_heads), replace=False)
        indices_cabezas_por_capa[layer_idx] = sorted(selected.tolist())
    
    imaxes, etiquetas = cargar_imaxes(
        dataset_path='data/DRIVE/val',
        tamano_patch=config['tamano_patch'],
        num_images=args.imaxes
    )
    imaxes = imaxes.to(device)
    
    # Use 'cls' mode by default for proper attention matrix visualization
    mapas_atencion = obter_mapas_atencion_cls(modelo, imaxes, indices_capas, num_heads=num_heads)

    plots_dir = Path(args.logs_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = f'{plots_dir /  Path(args.checkpoint).stem}_atencion.png'
    
    visualizar(
        imaxes=imaxes.cpu(),
        mapas_atencion=mapas_atencion,
        indices_capas=indices_capas,
        num_heads_to_show=args.cabezas,
        tamano_token=config['tamano_token'],
        output_path=output_path,
        indices_cabezas_por_capa=indices_cabezas_por_capa,
        etiquetas=etiquetas
    )

if __name__ == '__main__':
    main()
