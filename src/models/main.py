import argparse
import os
import random
import shutil
import time
import warnings
import math
from pathlib import Path
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import RandomSampler
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler

from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
from lion_pytorch import Lion

from data.DRIVE_SSL_Dataset import DriveSSLDataset
from model.crate import *
from model.vit import *


model_names = ["vit_tiny", "vit_small", "CRATE_tiny", "CRATE_small", "CRATE_base", "CRATE_large"]

def get_args_parser():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DIR', default="/path/to/imagenet",
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='CRATE_DRIVE',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: CRATE_tiny)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--label_smooth', default=0.1, type=float, metavar='L',
                        help='label smoothing coef')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help='Label Smoothing for PyTorch CrossEntropy')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='gpu id to use.')
    parser.add_argument('-tp','--tamano_patch', default=48, type=int,
                        help='la subimagen que se recorta de la grande')
    parser.add_argument('-tt', '--tamano_token', default=16, type=int,
                        help='el token del ViT')
    parser.add_argument('-s', '--sigma', default=3, type=float,
                        help='Sigma para la gausiana de las etiquetas')
    parser.add_argument('-logs_dir', default="logs", type=str,
                        help='a que directorio se van los logs')
    parser.add_argument('-weights_dir', default="weights", type=str,
                        help='a que directorio se van los pesos')
    parser.add_argument('-lm', '--label_mode', default="gaussian", type=str,
                        help='como se fabrican las etiquetas para cada patch')
    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Optimizer to Use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision training')
    return parser

#parser = get_args_parser()
best_acc1 = 0


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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    file_name = f'{args.logs_dir}/tp:{args.tamano_patch}_tt:{args.tamano_token}_s:{args.sigma}_lm:{args.label_mode}'
    
    csv_file, csv_writer = init_csv(file_name+'.log')

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, csv_file, csv_writer)



def main_worker(gpu, ngpus_per_node, args, csv_file, csv_writer):

    def CRATE_DRIVE(num_classes=2):
        return CRATE(
            image_size=args.tamano_patch,
            patch_size=args.tamano_token,
            num_classes=num_classes,
            dim=384,
            depth=12,
            heads=6,
            dropout=0.0,
            emb_dropout=0.0,
            dim_head=384 // 6
            )

    global best_acc1
    args.gpu = gpu


    print('==> Building model: {}'.format(args.arch))
    if args.arch == 'vit_tiny':
        model = vit_tiny_patch16(global_pool=True)
    elif args.arch == 'vit_small':
        model = vit_small_patch16(global_pool=True)
    elif args.arch == 'CRATE_tiny':
        model = CRATE_tiny()
    elif args.arch == "CRATE_small":
        model = CRATE_small()
    elif args.arch == "CRATE_base":
        model = CRATE_base()
    elif args.arch == "CRATE_large":
        model = CRATE_large()
    elif args.arch == "CRATE_DRIVE":
        model = CRATE_DRIVE()
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

    print(f'device: {device}')

    #if args.label_mode == 'gaussian':
    #    criterion = torch.nn.MSELoss()
    #else:
    #    #criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smooth).to(device)
    #    #class_weights = get_class_weights(dataset_train).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=).to(device))

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
        print("Using automatic mixed precision (AMP) training")
    else:
        scaler = None
        if args.use_amp and not torch.cuda.is_available():
            print("Warning: AMP requested but CUDA not available. Running without AMP.")

    warmup_steps = 20
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8),
                               0.5 * (math.cos(step / args.epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    transform_simple = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    #train_dataset = datasets.ImageFolder(
    #    traindir,
    #    transform_simple
    #    )

    train_dataset = DriveSSLDataset('data/DRIVE/train', tamano_patch=args.tamano_patch,
                                    label_mode=args.label_mode, sigma=args.sigma)

    val_dataset = DriveSSLDataset('data/DRIVE/val', tamano_patch=args.tamano_patch,
                                  label_mode=args.label_mode, sigma=args.sigma)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = RandomSampler(
            train_dataset, 
            replacement=True, # clave
            num_samples=args.batch_size
        )

        val_sampler = None
    print(f"I am using {args.workers} worker")


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args, device)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss, acc1, accx = train(train_loader, model, criterion, optimizer, epoch, device, args, scaler)

        # evaluate on validation set
        val_acc1 = validate(val_loader, model, criterion, args, device)

        csv_writer.writerow({'epoch': epoch, 'loss': loss,'train_accuracy': acc1.item(), 'val_accuracy': val_acc1.item() })
        csv_file.flush()   

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, args)

def train(train_loader, model, criterion, optimizer, epoch, device, args, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    print(f'numero batches : {len(train_loader)}')

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        print(f'numero de imagenes en batch: {images.shape}')
        print('etiquetas->>',target)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        #if args.label_mode == 'gaussian':
        #    target = target.to(device, dtype=torch.float32, non_blocking=True)
        #else: 
        target = target.to(device, non_blocking=True)

        torch.mps.synchronize()
        print('etiquetas->>',target)
        # compute output
        # Use AMP only if scaler is provided (CUDA + --use-amp flag)
        # For PyTorch 1.12.1, autocast() without arguments defaults to 'cuda'
        if scaler is not None:
            with autocast():
                output = model(images)
                print(output, target)
                if args.label_mode == 'gaussian':
                    output = torch.log_softmax(output, dim=1)
                    print('sofmax->', output)
                loss = criterion(output, target)
        else:
            output = model(images)
            print(f'y_pred -> {output}, y -> {target}')
            #if args.label_mode == 'gaussian':
            #    output = torch.log_softmax(output, dim=1)
            #    print('sofmax->', output, target)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 1)) # !!! -> que sea (1, 1) pierde
        # el sentido original de acc@5 pero lo dejo asi por si luego lo uso
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

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
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    best = f'{args.weights_dir}/tp:{args.tamano_patch}_tt:{args.tamano_token}_s:{args.sigma}_lm:{args.label_mode}'
    if is_best:
        shutil.copyfile(filename, best+'.pth.tar')




if __name__ == '__main__':
    main()
