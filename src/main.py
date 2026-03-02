import argparse
import os
import yaml
import hashlib
import random
import time
import warnings
import math
from dotenv import load_dotenv

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import RandomSampler
from torch.cuda.amp import autocast, GradScaler

from lion_pytorch import Lion

from src.data.Online_Dataset import Online_Dataset
from src.data.Offline_Dataset import Offline_Dataset
from src.data import recorta_dataset, ImageGroupedSampler 
from src.models.architectures import *
from src.utils import init_csv, init_yaml, accuracy, compute_auc, instantiate_model, instantiate_dataset, ProgressMeter, AverageMeter, Summary, print_prediccions, get_device
from src.utils.checkpoint import load_checkpoint, save_checkpoint



load_dotenv('.env')

def get_args_parser():

    parser = argparse.ArgumentParser(
        description='script de pruebas sobre CRATE de Yi Ma',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='CRATE_tiny',
                        choices=model_names,
                        help='arquitectura do modelo: ' +
                            ' | '.join(model_names) +
                            ' (default: CRATE_tiny)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--epochs', default=5000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--label_smoothing', default=0.0, type=float, metavar='L',
                        help='label smoothing coef')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float,
                        metavar='LR', help='initial learning rate (default 0.005)', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--use_amp', action='store_true', help='Toggles the use of mixed precission')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-eval', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--aumento_datos', action='store_true',
                        help='Usar aumento de datos')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('-tp','--tamano_patch', default=48, type=int,
                        help='la subimagen que se recorta de la grande')
    parser.add_argument('-tt', '--tamano_token', default=16, type=int,
                        help='el token del ViT')
    parser.add_argument('-ns', '--num_sigmas', default=4, type=int,
                        help='Numero de escalas para multisalida (usar con --label_mode=multiple, por defecto 4)')
    parser.add_argument('-s', '--sigma', default=3, type=float,
                        help='Sigma para la gausiana de las etiquetas')
    parser.add_argument('-t_dir', '--directorio_train_base', default="data/DRIVE/train", type=str,
                        help='directorio de las imagenes de train')
    parser.add_argument('-v_dir', '--directorio_val_base', default="data/DRIVE/val", type=str,
                        help='directorio de las imagenes de val')
    parser.add_argument('-runs_dir', default="data/runs", type=str,
                        help='a que directorio se van los logs')
    parser.add_argument('-weights_dir', default="data/weights", type=str,
                        help='a que directorio se van los pesos')
    parser.add_argument('--dataset', default="rfmid", type=str,
                        help='Dataset "offline" (defecto) o "online"')
    parser.add_argument('-ca', '--contador_aumento',  default=-1, type=int,
                        help='Cada cantos parches cambiase o aumento de datos da cache para a mesma imaxe, por defecto non cambiase.  (solo ten efecto se usase con --dataset online --aumento_datos)')
    parser.add_argument('-or', '--overlap_rate',  default=0.2, type=float,
                        help='Razon de sobrelapamiento de los parches')
    parser.add_argument('-lm', '--label_mode', default="vainilla", type=str,
                        help='como se fabrican las etiquetas para cada patch')
    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Optimizer to Use.')
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--paciencia', default=600, type=int,
                        help='number of epochs without improving loss before early stopping (default: 20)')
    parser.add_argument('--warmup_steps', default=20, type=int,
                        help='number of epochs ascencing to the base lr before the cosine decay')
    parser.add_argument('--class_weight', default=1.0, type=float,
                        help='class weight for positive class (vessel). 1.0 means no weighting, >1 penalizes vessel misclassification')

    parser.add_argument('--order', default='first', choices=['first', 'second'],
                        help='Order of Neumann approximation (default: first)')
    parser.add_argument('--share_proj', default='none', choices=['none', 'headwise', 'key-value', 'layerwise'],
                        help='Parameter sharing strategy for projection matrices (default: none)')
    parser.add_argument('--project_dim', default=None, type=int,
                        help='Dimension for projection matrices E and F (default: dim_head)')
    parser.add_argument('--no_pos', action='store_true',
                        help='Disable positional embeddings')
    parser.add_argument('--shared_dict', action='store_true',
                        help='Enable shared dictionary for CRATE')
    parser.add_argument('--linformer', action='store_true',
                        help='Enable Linformer efficient attention trick')

    return parser

def main():
    args = get_args_parser().parse_args()


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'wLabelSmoothingChich can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    random_bytes = os.urandom(32)  # 256 bits
    file_name = hashlib.sha256(random_bytes).hexdigest()[:6]

    init_yaml(file_name, args)
    csv_file, csv_writer = init_csv(file_name, args)

    if args.label_mode == 'multiple':
        args.num_classes = args.num_sigmas
    else:
        args.num_classes = 2

    device = get_device()

    model = instantiate_model(
        arch=args.arch,
        image_size=args.tamano_patch,
        patch_size=args.tamano_token,
        num_classes=args.num_classes,
        order=args.order,
        shared_dict=args.shared_dict,
        no_pos=args.no_pos,
        project_dim=args.project_dim,
        share_proj=args.share_proj,
        linformer=args.linformer
    )
    model.to(device)


    # Set up class weights
    if args.class_weight > 1.0:
        class_weights = torch.tensor([1.0, args.class_weight]).to(device)  # [background, vessel]
    else:
        class_weights = None
    
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    ).to(device)

    #criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.999),
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "Lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    # For PyTorch 1.12.1, GradScaler() should be called without arguments
    if torch.cuda.is_available() and args.use_amp:
        scaler = GradScaler()
        print("Adestrando con Precision Mezclada Automatica (AMP)")
    else:
        scaler = None
        if args.use_amp and not torch.cuda.is_available():
            print("VAITES: pediches Precision Mezclada Automatica (AMP) pero non tes NVIDIA. Non esta dispoñible para cpu nin a gráfica do mac.")

    lr_func = lambda step: min((step + 1) / (args.warmup_steps + 1e-8),
                               0.5 * (math.cos(step / args.epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    train_dataset, val_dataset = instantiate_dataset(args)

    train_sampler = ImageGroupedSampler(train_dataset, shuffle=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, 
        pin_memory=True, prefetch_factor=2, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, device)
        return

    best_acc1 = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        p = train_dataset.aug_scheduler.get_probabilidade(epoch) if train_dataset.aug_scheduler is not None else 0.0
        # train for one epoch
        loss, acc1, accx, train_auc = train(train_loader, model, criterion, optimizer, epoch, p, device, args, scaler, scheduler)

        # evaluate on validation set
        val_loss, val_acc1, val_auc = validate(val_loader, model, criterion, args, device)

        csv_writer.writerow({
            'epoch': epoch, 
            'loss': loss,
            'val_loss': val_loss,
            'train_accuracy': acc1.item(), 
            'val_accuracy': val_acc1.item(),
            'train_auc': train_auc.item(),
            'val_auc': val_auc.item()
        })
        csv_file.flush()   

        scheduler.step()

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.paciencia:
            print(f'\n==> Early stopping: No improvement for {args.paciencia} epochs')
            break

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, args, file_name)

def train(train_loader, model, criterion, optimizer, epoch, p, device, args, scaler=None, scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    aug_p= AverageMeter('p(Aug)', ':6.2f')
    lr_meter = AverageMeter('LR', ':6.5f')
    train_auc_meter = AverageMeter('AUC', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, aug_p, lr_meter, train_auc_meter],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    all_outputs = []
    all_targets = []

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # Use AMP only if scaler is provided (CUDA + --use-amp flag)
        # For PyTorch 1.12.1, autocast() without arguments defaults to 'cuda'
        if scaler is not None:
            with autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)


        if (epoch % 10 == 0 ) and ( i == 0):
            print_prediccions(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 1)) # !!! -> que sea (1, 1) pierde
        # el sentido original de acc@5 pero lo dejo asi por si luego lo uso
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        aug_p.update(p)
        
        # Accumulate outputs and targets for AUC
        all_outputs.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())
        
        # Calculate running AUC
        if all_outputs:
            all_outputs_cat = torch.cat(all_outputs, dim=0)
            all_targets_cat = torch.cat(all_targets, dim=0)
            running_auc = compute_auc(all_outputs_cat, all_targets_cat)
            train_auc_meter.update(running_auc.item())
        
        # track learning rate
        if scheduler is not None:
            lr_meter.update(scheduler.get_last_lr()[0])
        else:
            lr_meter.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    # Calculate final AUC over entire epoch
    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        train_auc = compute_auc(all_outputs, all_targets)
    else:
        train_auc = torch.tensor(0.0)

    return losses.avg, top1.avg, top5.avg, train_auc


def validate(val_loader, model, criterion, args, device):

    # Accumulate predictions and targets for AUC calculation
    all_outputs = []
    all_targets = []

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                images = images.to(device, non_blocking=True)
                if args.label_mode == 'gaussian':
                    target = target.to(device, dtype=torch.float32, non_blocking=True)
                else: 
                    target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 1)) # !!! -> que sea (1, 1) pierde
                # el sentido original de acc@5 pero lo dejo asi por si luego lo uso
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                
                # Accumulate for AUC calculation
                all_outputs.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())
                
                # Calculate running AUC
                if all_outputs:
                    all_outputs_cat = torch.cat(all_outputs, dim=0)
                    all_targets_cat = torch.cat(all_targets, dim=0)
                    running_auc = compute_auc(all_outputs_cat, all_targets_cat)
                    val_auc_meter.update(running_auc.item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    val_auc_meter = AverageMeter('Val AUC', ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, val_auc_meter],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    # Calculate final AUC
    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        val_auc = compute_auc(all_outputs, all_targets)
    else:
        val_auc = torch.tensor(0.0)

    return losses.avg, top1.avg, val_auc

if __name__ == '__main__':
    main()
