from einops import rearrange, repeat
import matplotlib.pyplot as plt
import argparse
import numpy as np

import torch

from src.models.architectures import *
from src.models.coding_rate import CodingRate
from src.plots.metrics import *
from src.data.Online_Dataset import Online_Dataset
from src.utils.cargar_config_yaml import cargar_config_yaml


coding_rate_list = []
sparsity_list = []

def forward_hook_codingrate(module, input, output):
    coding_rate_list.append(criterion(rearrange(output, 'b n (h d) -> b h n d', h=model.transformer.heads)))


def forward_hook_sparsity(module, input, output):
    sparsity_list.append(cal_sparsity(output.cpu().numpy(), is_sparse=True))


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("checkpoint_path", type=str, default="checkpoint.pth.tar")
    
    args.add_argument('--logs_dir', type=str, help='Path to the metadata (e.g. data/runs/)')

    # Dataset arguments
    args.add_argument("--directorio_train_base", type=str, default='data/DRIVE/val/', help="Base directory for training data")
    args.add_argument("--overlap_rate", type=float, default=0.0)
    args.add_argument("--batch_size", type=int, default=2048)
    args.add_argument("--workers", type=int, default=4)
    
    
    args = args.parse_args()
    
    config = cargar_config_yaml(args.checkpoint_path, args.logs_dir)
    
    tamano_patch = config.get('tamano_patch')
    print(f'tamano_patch: {tamano_patch}')
    tamano_token = config.get('tamano_token')
    print(f'tamano_token: {tamano_token}')
    sigma = config.get('sigma')
    num_sigmas = config.get('num_sigmas')
    label_mode = config.get('label_mode')

    criterion = CodingRate()
    model = CRATE_tiny(image_size=tamano_patch, patch_size=tamano_token)  # change this if you are not using CRATE_small
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint_path, map_location='cuda')
    new_state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()
    
    # Register hooks
    for layer in model.transformer.layers:
        layer[0].fn.qkv.register_forward_hook(forward_hook_codingrate)
        layer[1].register_forward_hook(forward_hook_sparsity)
    
    # Create dataset and dataloader
    train_dataset = Online_Dataset(
        args.directorio_train_base, 
        tamano_patch=tamano_patch,
        label_mode=label_mode, 
        sigma=sigma, 
        num_sigmas=num_sigmas,
        aumento_datos=False, 
        total_epochs=1, 
        sobrelapamento=args.overlap_rate
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers, 
        pin_memory=True, 
        prefetch_factor=4, 
        persistent_workers=True
    )
    
    all_coding_rates = []  
    all_sparsities = []   

    # Process batches
    print(f"imos procesar imaxes para ver a rede. en concreto faremos un _epoch_ de {len(train_loader)} minibatches")
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):

            # Reset per-batch lists
            coding_rate_list = []
            sparsity_list = []

            # Get images
            if isinstance(batch, dict):
                imgs = batch['imgs'].cuda() if 'imgs' in batch else batch['image'].cuda()
            else:
                imgs = batch[0].cuda() if isinstance(batch, (list, tuple)) else batch.cuda()

            output = model(imgs)

            # Store this batch's layer results
            batch_means = []
            batch_stds = []
            for (mean, std) in coding_rate_list:
                batch_means.append(mean.item())
                batch_stds.append(std.item())

            batch_spars_means = []
            batch_spars_stds = []
            for (mean, std) in sparsity_list:
                batch_spars_means.append(mean)
                batch_spars_stds.append(std)

            all_coding_rates.append((batch_means, batch_stds))
            all_sparsities.append((batch_spars_means, batch_spars_stds))

            print(f"Procesao batch {batch_idx + 1}/{len(train_loader)}")
 
    # Aggregate results
    means = []
    std_devs = []
    for (mean, std) in coding_rate_list:
        means.append(mean.item())
        std_devs.append(std.item())
    
    sparsities = []
    std_sparsities = []
    for (mean, std) in sparsity_list:
        sparsities.append(mean)
        std_sparsities.append(std)
    
    means = [means]
    std_devs = [std_devs]
    sparsities = [sparsities]
    std_sparsities = [std_sparsities]
    
    name = args.checkpoint_path.replace('data/weights','').replace('.pth.tar','') 

    plot_coding_rate(means, std_devs, name)
    plot_sparsity(sparsities, std_sparsities, name)
    
    print("listo :)")
