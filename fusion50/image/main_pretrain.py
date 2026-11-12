# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from functools import partial
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import mae3d.util.misc as misc
from mae3d.util.misc import NativeScalerWithGradNormCount as NativeScaler
from mae3d.datasets import RealVolumeDataset   # we may use it or build a custom dataset
from mae3d.models_mae import MaskedAutoencoderViT3D
from mae3d.engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=True)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--patch_size', default=[8, 12, 12], nargs=3, type=int,
                    help='Patch size: D H W')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='focus3d_mae3d', choices=['focus3d_mae3d'], type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=[32, 96, 96], nargs=3, type=int,
                    help='images input size: D H W')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--npy_axes', choices=['zyx', 'xyz'], default='zyx',
                        help='NPY spatial axis order on disk; optional channel axis must be first. TIFF stays ZYX.')

    parser.add_argument('--output_dir', default='./output/pretrain',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/pretrain',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', '--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    parser.add_argument('--config', type=str, default=None, help='YAML config; CLI values override YAML')
    parser.add_argument('--embed_dim', type=int, default=1008)
    parser.add_argument('--depth', type=int, default=24)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--decoder_embed_dim', type=int, default=768)
    parser.add_argument('--decoder_depth', type=int, default=8)
    parser.add_argument('--decoder_num_heads', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--mlp_ratio', type=float, default=4.)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=0, help='Limit batches per epoch; 0 means all')
    parser.add_argument('--dist_backend', choices=['nccl', 'gloo'], default='nccl')
    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    if args.distributed and device.type == 'cuda':
        device = torch.device('cuda', args.gpu)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    import random
    random.seed(seed)
    cudnn.benchmark = device.type == "cuda"

    if args.log_dir is not None and misc.is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    data_root = Path(args.data_path)
    train_dir = data_root / 'train' if (data_root / 'train').is_dir() else data_root
    dataset_train = RealVolumeDataset(str(train_dir), target_size=args.input_size,
                                      augment=args.augment, normalize=args.normalize,
                                      npy_axes=args.npy_axes)
    sample, _ = dataset_train[0]
    if sample.shape[0] != args.in_chans:
        raise ValueError(f"Volume channels {sample.shape[0]} != configured in_chans {args.in_chans}")

    # data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
    #                         num_workers=args.num_workers, pin_memory=True, persistent_workers=True,prefetch_factor=4)
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    if len(data_loader_train) == 0:
        raise ValueError("No complete batch: reduce batch_size or provide more volumes per rank")
    # define the model
    model = MaskedAutoencoderViT3D(
        img_size=tuple(args.input_size),
        in_chans=args.in_chans,
        norm_pix_loss=args.norm_pix_loss,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    model.to(device)

    model_without_ddp = model
    print("Model: %s; parameters: %s" % (type(model_without_ddp).__name__, sum(p.numel() for p in model.parameters())))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu] if device.type == 'cuda' else None,
            find_unused_parameters=False)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    decay, no_decay = [], []
    for name, param in model_without_ddp.named_parameters():
        if param.requires_grad:
            (no_decay if param.ndim <= 1 or name.endswith('.bias') else decay).append(param)
    param_groups = [{'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler(enabled=args.amp and device.type == "cuda")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if log_writer is not None:
        log_writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    from mae3d.config import parse_config
    args = parse_config(get_args_parser())
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if misc.is_main_process() and int(os.environ.get('RANK', 0)) == 0 and args.output_dir:
        import yaml
        with open(Path(args.output_dir) / 'config.yaml', 'w') as f:
            yaml.safe_dump(vars(args), f, sort_keys=True)
    main(args)
