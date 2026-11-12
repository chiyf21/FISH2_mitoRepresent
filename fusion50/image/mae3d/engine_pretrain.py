# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from typing import Iterable
import torch
from .util import misc, lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    num_steps = min(len(data_loader), args.max_steps) if args.max_steps else len(data_loader)

    # optimizer.zero_grad()
    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step >= num_steps:
            break
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_steps + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=args.amp and device.type == "cuda"):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()


        # All ranks fail together before backward to avoid a DDP deadlock.
        bad = misc.all_reduce_mean(float(not math.isfinite(loss_value)))
        if bad:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={data_iter_step}")

        group_start = (data_iter_step // accum_iter) * accum_iter
        group_size = min(accum_iter, num_steps - group_start)
        update_grad = (data_iter_step + 1) % accum_iter == 0 or data_iter_step + 1 == num_steps
        loss = loss / group_size
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=update_grad)
        if update_grad:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and update_grad:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / num_steps + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
