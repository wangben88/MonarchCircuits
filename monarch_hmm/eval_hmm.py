import os
import argparse
import random
import math
import numpy

import torch
import torch.nn as nn
import torch.distributed as dist
import datasets

from tqdm import tqdm
from monarch import *

torch.backends.cuda.matmul.allow_tf32 = True


def ll_to_bpc(ll, seq_len):
    return - ll / seq_len / math.log(2)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--init_model_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--batch_size', default=32, type=int)

    args = arg_parser.parse_args()

    dist.init_process_group('nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f'cuda:{rank}'

    print(f'loading {args.dataset}...')
    dataset = torch.load(args.dataset, weights_only=False)

    test_data = dataset['test_one_line']
    print(f'test_data size: {test_data.shape}')
    test_data = test_data.view(args.batch_size, -1)
    print(f'reshaped test size: {test_data.shape}')

    hmm_model = HMM.from_pretrained(f'{args.init_model_path}', map_location='cpu').to(device)

    with torch.no_grad():
        ll = torch.sum(hmm_model(test_data.to(hmm_model.beta.device))).item()
        print(ll_to_bpc(ll, test_data.shape[0] * test_data.shape[-1]))