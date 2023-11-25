import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

os.environ["MKL_NUM_THREADS"] = "2" 
os.environ["NUMEXPR_NUM_THREADS"] = "2" 
os.environ["OMP_NUM_THREADS"] = "2" 

from os import path, makedirs, listdir
import sys
import numpy as np
import random

from contextlib import suppress

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

from torch.optim import SGD

# from adamw import AdamW
# from timm.optim import AdamW
from torch.optim.adamw import AdamW

from losses import iou_round, dice_round, ComboLoss

from sklearn.metrics import mean_squared_error, log_loss

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from models import Timm_Unet

from Dataset import TrainDataset #, ValDataset
from utils import *

from timm.utils.distributed import distribute_bn

from ddp_utils import all_gather, reduce_tensor

import timm

from torch.utils.tensorboard import SummaryWriter

# import warnings
# warnings.filterwarnings("ignore")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amp', default=True, type=bool)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--encoder", default='convnextv2_base.fcmae_ft_in22k_in1k') #  coat_lite_medium
parser.add_argument("--checkpoint", default='convnextv2_base_256_e04_pretrain')  
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument("--checkpoint_path", default='')
parser.add_argument("--continue_best", default=False, type=bool)


args, unknown = parser.parse_known_args()


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

local_rank = 0
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
args.local_rank = local_rank


train_files = sorted(listdir('pretrain_crops'))
val_files = sorted(listdir('val_crops'))

train_data_dir='pretrain_crops'
train_masks_dir='pretrain_masks'

val_data_dir='val_crops'
val_masks_dir='val_masks'

models_folder = 'weights'



def validate(model, val_data_loader, current_epoch, amp_autocast=suppress):
    metrics = [[] for i in range(9)]

    if args.local_rank == 0:
        iterator = tqdm(val_data_loader)
    else:
        iterator = val_data_loader

    with torch.no_grad():
        for i, sample in enumerate(iterator):
            with amp_autocast():
                imgs = sample['img'].cuda(non_blocking=True)
                otps = sample['msk'].cpu().numpy()

                res = model(imgs)
                pred = torch.sigmoid(res).cpu().numpy()

                for j in range(otps.shape[0]):

                    for l in range(1):
                        _truth = otps[j, l] > 0.5
                        _pred = pred[j, l] > 0.5

                        _dice = dice(_truth, _pred)
                        
                        metrics[l].append(_dice)

  
    metrics = [np.asarray(x) for x in metrics]

    if args.distributed:
        metrics = [np.concatenate(all_gather(x)) for x in metrics]

    _dice_mean = np.mean(metrics[0])

    if args.local_rank == 0:
        print("Val Dice: {} Len: {}".format(_dice_mean, len(metrics[0])))


    return _dice_mean


def evaluate_val(val_data_loader, best_score, model, snapshot_name, current_epoch, amp_autocast=suppress):
    model.eval()
    _sc = validate(model, val_data_loader, current_epoch, amp_autocast)

    if args.local_rank == 0:
        if _sc > best_score:
            if args.distributed:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))
            else:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': _sc,
                }, path.join(models_folder, snapshot_name))

            best_score = _sc
        print("Val score: {}\tbest_score: {}".format(_sc, best_score))
    return best_score, _sc



def train_epoch(current_epoch, seg_loss, cce_loss, model, optimizer, scaler, train_data_loader, amp_autocast=suppress):
    losses = [AverageMeter() for i in range(10)]
    metrics = [AverageMeter() for i in range(10)]
    
    if args.local_rank == 0:
        iterator = tqdm(train_data_loader)
    else:
        iterator = train_data_loader

    _lr = optimizer.param_groups[0]['lr']

    model.train()

    for __i, sample in enumerate(iterator):
        with amp_autocast():
            imgs = sample["img"].cuda(non_blocking=True)
            otps = sample["msk"].cuda(non_blocking=True)

            res = model(imgs)

            loss = seg_loss(res, otps)

            if current_epoch < start_epoch + 1:
                loss = loss * 0.05 # warm-up

        _dices = []
        with torch.no_grad():
            _probs = torch.sigmoid(res)
            dice_sc = 1 - dice_round(_probs, otps)
            _dices.append(dice_sc)

        if args.distributed:
            reduced_loss = [reduce_tensor(x.data) for x in [loss]]
            reduced_sc = [reduce_tensor(x) for x in _dices]
        else:
            reduced_loss = [x.data for x in [loss]]
            reduced_sc = _dices

        for _i in range(len(reduced_loss)):
            losses[_i].update(to_python_float(reduced_loss[_i]), imgs.size(0))
        for _i in range(len(reduced_sc)):
            metrics[_i].update(reduced_sc[_i], imgs.size(0)) 

        if args.local_rank == 0:
            iterator.set_description(
                "epoch: {}; lr {:.7f}; Loss: {:.4f}  ({:.4f}) dices: {:.4f} ({:.4f})".format(
                    current_epoch, _lr, losses[0].val, losses[0].avg, metrics[0].val, metrics[0].avg))


        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.999)
            optimizer.step()

        torch.cuda.synchronize()

    if args.local_rank == 0:
        _dice = metrics[0].avg
        writer.add_scalar("Loss/train", losses[0].avg, current_epoch)
        writer.add_scalar("Dice/train", _dice, current_epoch)
        writer.add_scalar("Loss cce/train", losses[1].avg, current_epoch)
        writer.add_scalar("lr", _lr, current_epoch)

        print("epoch: {}; lr {:.7f}; Loss {:.4f} dices: {:.4f};".format(
                    current_epoch, _lr, losses[0].avg, _dice))


start_epoch = 0
            

if __name__ == '__main__':
    t0 = timeit.default_timer()
    
    makedirs(models_folder, exist_ok=True)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()


    fold = args.fold


    if args.local_rank == 0:
        writer = SummaryWriter(comment='__{}_{}'.format(args.checkpoint, fold))
        print(args)
        
    cudnn.benchmark = True

    batch_size = args.batch_size
    val_batch = args.batch_size

    best_snapshot_name = '{}_{}_pre_best'.format(args.checkpoint, fold)
    last_snapshot_name = '{}_{}_pre_last'.format(args.checkpoint, fold)


    data_train = TrainDataset(train_files, data_dir=train_data_dir, masks_dir=train_masks_dir, aug=True)
    data_val = TrainDataset(val_files, data_dir=val_data_dir, masks_dir=val_masks_dir, aug=False)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(data_val, shuffle=False)


    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=8, shuffle=(train_sampler is None), pin_memory=False, sampler=train_sampler, drop_last=True)
    val_data_loader = DataLoader(data_val, batch_size=val_batch, num_workers=8, shuffle=False, pin_memory=False, sampler=val_sampler)

    model = Timm_Unet(name=args.encoder, pretrained=args.pretrained) 


    if args.distributed:
        timm.models.layers.convert_sync_batchnorm(model)

    model = model.cuda()

    params = model.parameters()
    
    lr = 1e-4
    if args.continue_best:
        lr = 1e-5
    
    optimizer = AdamW(params, lr=lr) #, weight_decay=1e-3


    if args.continue_best:
        snap_to_load = best_snapshot_name.format(fold)
        if path.exists(path.join(models_folder, snap_to_load)):
            if args.local_rank == 0:
                print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            model.load_state_dict(loaded_dict)
            if args.local_rank == 0:
                print("loaded checkpoint '{}' (epoch {}, best_score {})"
                    .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))

        start_epoch = checkpoint['epoch'] + 1


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
            output_device=args.local_rank) #, find_unused_parameters=True


    loss_scaler = None
    amp_autocast = suppress
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
        amp_autocast = torch.cuda.amp.autocast

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=False, threshold=0.00001, threshold_mode='abs', cooldown=0, min_lr=1e-08, eps=1e-08)

    seg_loss = ComboLoss({'dice': 1.0, 'bce': 0.1}, per_image=False).cuda() 
    cce_loss = nn.CrossEntropyLoss().cuda()


    # best_score, _sc = evaluate_val(val_data_loader, -1, model, best_snapshot_name, 16, amp_autocast)
    best_score = 0
    for epoch in range(start_epoch, 5): #start_epoch
        torch.cuda.empty_cache()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_epoch(epoch, seg_loss, cce_loss, model, optimizer, loss_scaler, train_data_loader, amp_autocast)

        if args.distributed:
            distribute_bn(model, args.world_size, True)

        # if epoch % 2 != 0:
        #     continue

        torch.cuda.empty_cache()

        best_score, _sc = evaluate_val(val_data_loader, best_score, model, best_snapshot_name, epoch, amp_autocast)
        # _sc = 0
        scheduler.step(_sc)
        
        torch.cuda.empty_cache()

        if args.local_rank == 0:
            writer.flush()

            if args.distributed:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict(),
                    'best_score': best_score,
                }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': best_score,
                }, path.join(models_folder, last_snapshot_name + '_' + str(epoch)))
    
    torch.cuda.empty_cache()
    if args.distributed:
        torch.cuda.synchronize()

    del model

    elapsed = timeit.default_timer() - t0
    if args.local_rank == 0:
        writer.close()
        print('Time: {:.3f} min'.format(elapsed / 60))