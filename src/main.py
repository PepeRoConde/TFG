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
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler

from lion_pytorch import Lion

from src.data.Online_Dataset import Online_Dataset
from src.data.Offline_Dataset import Offline_Dataset
from src.data.recorta_dataset import recorta_dataset
from src.models.architectures import *
from src.utils import init_csv, accuracy, ProgressMeter, AverageMeter, Summary, print_prediccions
from src.utils.checkpoint import load_checkpoint, save_checkpoint


model_names = ["vit_tiny", "vit_small", "CRATE_tiny", "CRATE_small", "CRATE_base", "CRATE_large"]

load_dotenv('.env')

def get_args_parser():

    parser = argparse.ArgumentParser(
        description='script de pruebas sobre CRATE de Yi Ma',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='CRATE_tiny',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: CRATE_tiny)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--label_smoothing', default=0.1, type=float, metavar='L',
                        help='label smoothing coef')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float,
                        metavar='LR', help='initial learning rate (default 0.005)', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
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
    parser.add_argument('--dataset', default="offline", type=str,
                        help='Dataset "offline" (defecto) o "online"')
    parser.add_argument('-ca', '--contador_aumento',  default=-1, type=int,
                        help='Cada cantos parches cambiase o aumento de datos da cache para a mesma imaxe, por defecto non cambiase.  (solo ten efecto se usase con --dataset online --aumento_datos)')
    parser.add_argument('-or', '--overlap_rate',  default=0.2, type=float,
                        help='Razon de sobrelapamiento de los parches')
    parser.add_argument('-lm', '--label_mode', default="gaussian", type=str,
                        help='como se fabrican las etiquetas para cada patch')
    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Optimizer to Use.')
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision training')

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


    #file_name = f'a:{args.arch}_tp:{args.tamano_patch}_tt:{args.tamano_token}_s:{args.sigma}_lm:{args.label_mode}'
    random_bytes = os.urandom(32)  # 256 bits
    file_name = hashlib.sha256(random_bytes).hexdigest()[:6]

    yaml_path = f"{args.runs_dir}/metadata/{file_name}.yaml"
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)

    csv_file, csv_writer = init_csv(args.runs_dir + '/' + file_name + '.log')

    if args.label_mode == 'multiple':
        args.num_classes = args.num_sigmas
    else:
        args.num_classes = 2

    print('==> Building model: {}'.format(args.arch))
    if args.arch == 'vit_tiny':
        model = vit_tiny_patch16(global_pool=True)
    elif args.arch == 'vit_small':
        model = vit_small_patch16(global_pool=True)
    elif args.arch == 'CRATE_tiny':
        model = CRATE_tiny(image_size=args.tamano_patch, patch_size=args.tamano_token, num_classes=args.num_classes)
    elif args.arch == "CRATE_small":
        model = CRATE_small(image_size=args.tamano_patch, patch_size=args.tamano_token, num_classes=args.num_classes)
    elif args.arch == "CRATE_base":
        model = CRATE_base(image_size=args.tamano_patch, patch_size=args.tamano_token, num_classes=args.num_classes)
    elif args.arch == "CRATE_large":
        model = CRATE_large(image_size=args.tamano_patch, patch_size=args.tamano_token, num_classes=args.num_classes)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to("cuda")
        device = torch.device("cuda")
        print('Usando la gráfica NVIDIA')

    elif torch.backends.mps.is_available():
        model = model.to("mps")
        device = torch.device("mps")
        print('Usando la gráfica del portatil')

    else:
        print("using CPU, this will be slow")
        device = torch.device("cpu")

    class_weights = torch.tensor([1.0, 7.0]).to(device)  # [background, vessel]
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

    # Initialize GradScaler only for CUDA with AMP enabled
    # For PyTorch 1.12.1, GradScaler() should be called without arguments
    if torch.cuda.is_available() and args.use_amp:
        scaler = GradScaler()
        print("Adestrando con Precision Mezclada Automatica (AMP)")
    else:
        scaler = None
        if args.use_amp and not torch.cuda.is_available():
            print("VAITES: pediches Precision Mezclada Automatica (AMP) pero non tes NVIDIA. Non esta dispoñible para cpu nin a gráfica do mac.")

    warmup_steps = 20
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8),
                               0.5 * (math.cos(step / args.epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


    # Data loading code
    if args.dataset == 'online':
        train_dataset = Online_Dataset(args.directorio_train_base, tamano_patch=args.tamano_patch, label_mode=args.label_mode, 
                                       sigma=args.sigma, num_sigmas=args.num_sigmas, aumento_datos=args.aumento_datos, 
                                       total_epochs=args.epochs, sobrelapamento=args.overlap_rate, contador_aumento=args.contador_aumento)

        val_dataset = Online_Dataset(args.directorio_val_base, tamano_patch=args.tamano_patch, label_mode=args.label_mode, 
                                     sigma=args.sigma, num_sigmas=args.num_sigmas, aumento_datos=args.aumento_datos, 
                                     total_epochs=args.epochs, sobrelapamento=args.overlap_rate, contador_aumento=args.contador_aumento)

    elif args.dataset == 'offline':
       
        # train -------

        directorio_train_cropeado = f'{args.directorio_train_base}_{args.tamano_patch}_{args.overlap_rate}'

        if not os.path.isdir(directorio_train_cropeado):
            print('no hay el conjunto de datos de patches que quieres para entrenar, esperte y te lo hago')
            recorta_dataset(input_dir=args.directorio_train_base, output_dir=directorio_train_cropeado, 
                               patch_size=args.tamano_patch,  overlap_rate=args.overlap_rate,
                               image_start_idx=21, image_end_idx=36)


        train_dataset = Offline_Dataset(directorio_train_cropeado,
                                        label_mode=args.label_mode, sigma=args.sigma, num_sigmas=args.num_sigmas,
                                      data_augmentation=args.aumento_datos, total_epochs=args.epochs)

        # val -----------

        directorio_val_cropeado = f'{args.directorio_val_base}_{args.tamano_patch}_{args.overlap_rate}'

        if not os.path.isdir(directorio_val_cropeado):
            print('no hay el conjunto de datos de patches que quieres para validar, esperte y te lo hago')
            recorta_dataset(input_dir=args.directorio_val_base, output_dir=directorio_val_cropeado, 
                               patch_size=args.tamano_patch,  overlap_rate=args.overlap_rate,
                               image_start_idx=37, image_end_idx=39)

        val_dataset = Offline_Dataset(directorio_val_cropeado,
                                        label_mode=args.label_mode, sigma=args.sigma, num_sigmas=args.num_sigmas,
                                      data_augmentation=False, total_epochs=args.epochs)
        
        train_sampler = None

    else:
        print(f'La opcion {args.dataset} no existe, debe ser o "online" (cortar las imagenes bajo demanda) o "offline" (previamente se espera la ejecucion de src.data.crop_script)')

    print(f"Usando {args.workers} hilos")


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, 
        pin_memory=True, prefetch_factor=4, persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, device)
        return

    best_acc1 = 0

    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        p = train_dataset.aug_scheduler.get_probabilidade(epoch)
        # train for one epoch
        loss, acc1, accx = train(train_loader, model, criterion, optimizer, epoch, p, device, args, scaler)

        # evaluate on validation set
        val_acc1 = validate(val_loader, model, criterion, args, device)

        csv_writer.writerow({'epoch': epoch, 'loss': loss,'train_accuracy': acc1.item(), 'val_accuracy': val_acc1.item() })
        csv_file.flush()   

        scheduler.step()

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

def train(train_loader, model, criterion, optimizer, epoch, p, device, args, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    aug_p= AverageMeter('p(Aug)', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, aug_p],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    print(f'numero batches : {len(train_loader)}')

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        #if args.label_mode == 'gaussian':
        #    target = target.to(device, dtype=torch.float32, non_blocking=True)
        #else: 
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

    return loss.item(), acc1[0], acc5[0]


def validate(val_loader, model, criterion, args, device):

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

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg

if __name__ == '__main__':
    main()
