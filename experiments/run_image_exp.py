import wandb
import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.multiprocessing as mp
import argparse

from monarch_juice.ddp import get_free_port
from monarch_juice.train import construct_and_train_pc

sys.setrecursionlimit(15000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monarch Imagenet32')
    # Model args
    parser.add_argument('-ly', '--layer_type', type=str, default='monarch', help='layer type to use')
    parser.add_argument('-hs',  '--hidden_size',     type=int,   default=1024,    help='hidden size of model; must be 2^{2n} for positive integer n')
    # Training args
    parser.add_argument('-bs',  '--batch_size',     type=int,   default=250,help='batch size')
    parser.add_argument('-ms', '--minibatch_schedule', type = str, default='20000,20',help='minibatch EM schedule')
    parser.add_argument('-ds', '--dataset', type = str, default='imagenet32', help='dataset (either imagenet32 or imagenet64)')
    # Dataset args
    parser.add_argument("-ll", "--lossless", action='store_true', help='whether to train on lossless (YCoCg-R) or lossy (YCoCg) transformed data')
    parser.add_argument("-ps", "--patch_size", type = int, default = 8, help = "image patch sizes for training")
    # Misc args
    parser.add_argument('-sloc', '--save_location', type=str, default='model.jpc', help='save location')
    parser.add_argument('-lloc', '--load_location', type=str, default=None, help='load location')
    parser.add_argument('-dv', '--devices', type=str, default='0', help='devices to use, comma separated')
    

    args = parser.parse_args()
    assert args.patch_size in [1,2,4,8, 16, 32, 64], "Invalid patch size"

    torch.multiprocessing.set_start_method('spawn')
    device_list = args.devices.split(',')
    world_size = len(device_list)

    port = get_free_port()
    args.port = port

    if world_size == 1:
        construct_and_train_pc(0, world_size, args)
    else:
        mp.spawn(
            construct_and_train_pc,
            args = (world_size, args),
            nprocs = world_size,
        )  # nprocs - total number of processes - # gpus
    
    if args.activate_wandb:
        wandb.finish()
