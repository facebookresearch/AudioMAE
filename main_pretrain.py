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

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_audio, interpolate_patch_embed_audio
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch
from dataset import AudiosetDataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.8, type=float, 
                        help='Masking ratio (percentage of removed patches).') # 0.75

    #parser.add_argument('--norm_pix_loss', action='store_true',
    #                    help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.add_argument('--norm_pix_loss', type=bool, default=False, help='Use (per-patch) normalized pixels as targets for computing loss')
    

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    # For audioset
    parser.add_argument('--audio_exp', type=bool, default=True, help='audio exp')
    #parser.add_argument("--data_train", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train.json', help="training data json")
    #parser.add_argument("--data_eval", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval.json', help="validation data json")
    parser.add_argument("--data_train", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/train_video.json', help="training data json")
    parser.add_argument("--data_eval", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/datafiles/eval_video.json', help="validation data json")    
    parser.add_argument("--label_csv", type=str, default='/checkpoint/berniehuang/ast/egs/audioset/data/class_labels_indices.csv', help="csv with class labels")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0) # pretraining 0
    parser.add_argument('--timem', help='time mask max length', type=int, default=0) # pretraining 0
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--dataset", type=str, default="audioset", help="dataset", choices=["audioset", "esc50", "speechcommands"])
    parser.add_argument("--use_fbank", type=bool, default=False)
    parser.add_argument("--fbank_dir", type=str, default="/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank", help="fbank dir")
    parser.add_argument("--alpha", type=float, default=0.0, help="contrastive loss weight")
    parser.add_argument("--omega", type=float, default=1.0, help="reconstruction loss weight")    
    parser.add_argument('--mode', default=0, type=int,help='contrastive mode')
    parser.add_argument('--save_every_epoch', default=20, type=int,help='save_every_epoch')
    parser.add_argument('--use_custom_patch', type=bool, default=False, help='use custom patch and override timm PatchEmbed')
    #parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument('--roll_mag_aug', type=bool, default=False, help='use roll_mag_aug')	
    parser.add_argument('--split_pos', type=bool, default=False, help='use splitted pos emb')	
    parser.add_argument('--pos_trainable', type=bool, default=False, help='use trainable pos emb')	
    parser.add_argument('--use_nce', type=bool, default=False, help='use use_nce')
    parser.add_argument('--load_video', type=bool, default=False, help='load video')
    parser.add_argument('--decoder_mode', default=1, type=int,help='decoder mode 0: global attn 1: swined local attn')
    # remove for A-MAE
    #parser.add_argument('--v_weight', default=1.0, type=float, help='reconstruction weight for the visual part')
    #parser.add_argument('--video_only', type=bool, default=False, help='video_only pre-training')
    #parser.add_argument('--cl', type=bool, default=False, help='use pre-text curriculum')
    #parser.add_argument('--n_frm', default=4, type=int,help='how many frames to encode, at least 2 as temporal kernel stride is 2')
    #parser.add_argument('--depth_av', default=3, type=int,help='depth of multimodal fusion encoder')
    parser.add_argument('--mask_t_prob', default=0.7, type=float, help='ratio of masking time')
    parser.add_argument('--mask_f_prob', default=0.3, type=float, help='ratio of masking freq')
    parser.add_argument('--mask_2d', type=bool, default=False, help='use 2d masking')
    parser.add_argument('--init_audio_with_video_mae', type=bool, default=False, help='init_audio_with_video_mae')
    parser.set_defaults(audio_exp=True)
    parser.add_argument('--no_shift', type=bool, default=False, help='no_shift')

    # set norm_pix_loss=True for normal training, norm_pix_loss=False for visualization
    parser.set_defaults(norm_pix_loss=True)
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('======================= starting pretrain =======================')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    if not args.audio_exp:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    else:
        norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
        target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
        multilabel_dataset = {'audioset': True, 'esc50': False, 'k400': False, 'speechcommands': True}
        audio_conf = {'num_mel_bins': 128, 
                      'target_length': target_length[args.dataset], 
                      'freqm': args.freqm,
                      'timem': args.timem,
                      'mixup': args.mixup,
                      'dataset': args.dataset,
                      'mode':'train',
                      'mean':norm_stats[args.dataset][0],
                      'std':norm_stats[args.dataset][1],
                      'multilabel':multilabel_dataset[args.dataset],
                      'noise':False}
        dataset_train = AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf, roll_mag_aug=args.roll_mag_aug,
                                        load_video=args.load_video)
    #print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    if args.audio_exp:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, 	
                                            in_chans=1, audio_exp=True,	
                                            img_size=(target_length[args.dataset],128),	
                                            alpha=args.alpha, mode=args.mode, use_custom_patch=args.use_custom_patch,	
                                            split_pos=args.split_pos, pos_trainable=args.pos_trainable, use_nce=args.use_nce,
                                            decoder_mode=args.decoder_mode, 
                                            mask_2d=args.mask_2d, mask_t_prob=args.mask_t_prob, mask_f_prob=args.mask_f_prob, 
                                            no_shift=args.no_shift,
                                            # remove for A-MAE
                                            #v_weight=args.v_weight, n_frm=args.n_frm, video_only=args.video_only, cl=args.cl, depth_av=args.depth_av,
                                            )
    else:
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print('use distributed!!')
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

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
        if args.output_dir and (epoch % args.save_every_epoch == 0 or epoch + 1 == args.epochs):
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


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
