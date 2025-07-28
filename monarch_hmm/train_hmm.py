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
from hmm import *

torch.backends.cuda.matmul.allow_tf32 = True

def loglikelihood(model, data_loader):
    device = model.beta.device

    with torch.no_grad():
        ll = torch.tensor([0.0], device=device)
        data_size = torch.tensor([0.0], device=device)
        for input_ids_batch in data_loader:
            input_ids_batch = input_ids_batch.to(device)
            probs_ = model(input_ids_batch)
            ll += torch.sum(probs_)
            data_size += input_ids_batch.shape[0]

    dist.all_reduce(ll, op=dist.ReduceOp.SUM)
    dist.all_reduce(data_size, op=dist.ReduceOp.SUM)

    avg_ll = ll.item() / data_size.item()

    return avg_ll


def ll_to_bpc(ll, seq_len):
    return - ll / seq_len / math.log(2)


def eval_ll_and_log(hmm_model, seq_len, train_loader, dev_loader, test_loader, rank, world_size, ckpt, decay, log_file):
    # evaluate ll
    if train_loader != []:
        train_ll = loglikelihood(hmm_model, train_loader)
        train_ll = ll_to_bpc(train_ll, seq_len)
    else:
        train_ll = -1

    dev_ll = loglikelihood(hmm_model, dev_loader)
    test_ll = loglikelihood(hmm_model, test_loader)

    if rank == 0:
        dev_ll = ll_to_bpc(dev_ll, seq_len)
        test_ll = ll_to_bpc(test_ll, seq_len)
        msg = f'{ckpt}\t{train_ll}\t{dev_ll}\t{test_ll}\t{decay:.9f}'
        print(msg)
        with open(log_file, 'a+') as fout:
            fout.write(msg + '\n')


def generate_decay_schedule(num_steps, decay_start, decay_end, decay_fn):
    if num_steps > 1:
        if decay_fn.startswith('sigmoid'):
            decay_schedule = torch.arange(0, num_steps, dtype=torch.float32)
            a = -torch.log(torch.tensor(1.0/decay_start - 1.0))
            b = -torch.log(torch.tensor(1.0/decay_end - 1.0))
            decay_schedule = a * ((num_steps - 1 - decay_schedule) / (num_steps - 1)) + b * (decay_schedule / (num_steps - 1))
            decay_schedule = torch.sigmoid(decay_schedule).tolist()

        if decay_fn == 'linear':
            decay_schedule = torch.arange(0, num_steps)
            a = torch.pow(torch.tensor(decay_start), 1.0)
            b = torch.pow(torch.tensor(decay_end), 1.0)
            decay_schedule = a * ((num_steps - 1 - decay_schedule) / (num_steps - 1)) + b * (decay_schedule / (num_steps - 1))
            decay_schedule = torch.pow(decay_schedule, 1.0).tolist()
    else:
        decay_schedule = [decay_start]

    return decay_schedule


def generate_data_loader(rank, world_size, dataset, batch_size, drop_last=False):
    if drop_last:
        num_per_process = dataset.shape[0] // world_size
    else:
        num_per_process = math.ceil(dataset.shape[0] / world_size)

    dataset = dataset[rank * num_per_process: min(dataset.shape[0], (rank+1) * num_per_process)]

    data_loader = []
    for batch_idx in range(0, dataset.shape[0], batch_size):
        batch_size_ = min(dataset.shape[0] - batch_idx, batch_size)
        if batch_size_ == batch_size or (not drop_last):
            data_loader.append(dataset[batch_idx: batch_idx+batch_size_].clone().detach())

    return data_loader


def train_hmm(rank, world_size,
    init_model_path, model_path, checkpoint, save_per_steps, eval_per_steps,
    train_data, dev_data, test_data,
    num_train_epochs, num_train_steps,
    batch_size, grad_accum_iters, sample_length,
    decay_schedule, log_file, dropout, pseudocount,
    hidden_states, block_size, vocab_size, sep_token_id,
    eval_only=False, monarch=False):

    device = f'cuda:{rank}'
    seq_len = train_data.shape[1]

    dev_loader = generate_data_loader(rank, world_size, dev_data, batch_size, drop_last=False)
    test_loader = generate_data_loader(rank, world_size, test_data, batch_size, drop_last=False)

    if init_model_path != '':
        if monarch:
            hmm_model = MonarchHMM.from_pretrained(f'{init_model_path}', map_location='cpu').to(device)
        else:
            hmm_model = HMM.from_pretrained(f'{init_model_path}', map_location='cpu').to(device)
    else:
        checkpoint = 0
        if monarch:
            hmm_model = MonarchHMM(hidden_states, block_size, vocab_size, sep_token_id).to(device)
        else:
            hmm_model = HMM(hidden_states, vocab_size, sep_token_id).to(device)

    grad_update_cnt = 0
    hmm_model.zero_grad()

    # eval at step 0
    eval_ll_and_log(hmm_model, seq_len, [], dev_loader, test_loader,
                                rank, world_size, checkpoint, -1, log_file)

    if eval_only:
        return

    for epoch_cnt in range(num_train_epochs):
        # shuffle train data every epoch
        train_data = train_data[torch.randperm(train_data.shape[0])].contiguous()
        train_loader = generate_data_loader(rank, world_size, train_data, batch_size, drop_last=True)

        for iter_cnt, train_data_batch in enumerate(tqdm(train_loader, desc="Training")):
            train_data_batch = train_data_batch.to(device)
            log_probs = hmm_model(train_data_batch)
            loss = torch.sum(log_probs)
            loss.backward()

            # update params every grad_accum_iters
            if (iter_cnt + 1) % grad_accum_iters == 0:
                decay = decay_schedule[grad_update_cnt]

                hmm_model.step(lr=decay)

                hmm_model.zero_grad()

                # update step count (# of gradient updates)
                grad_update_cnt += 1

                if grad_update_cnt % eval_per_steps == 0 or grad_update_cnt == 1 or grad_update_cnt == num_train_steps:
                    eval_ll_and_log(hmm_model, seq_len, [train_data_batch], dev_loader, test_loader,
                                    rank, world_size, checkpoint + grad_update_cnt, decay, log_file)

                if grad_update_cnt % save_per_steps == 0 and rank == 0:
                    hmm_model.save_pretrained(f'{model_path}/checkpoint-{checkpoint + grad_update_cnt}')

                if grad_update_cnt >= num_train_steps:
                    break

        if rank == 0:
            hmm_model.save_pretrained(f'{model_path}/checkpoint-{checkpoint + grad_update_cnt}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--init_model_path', default='', type=str)
    arg_parser.add_argument('--checkpoint', default=50, type=int)
    arg_parser.add_argument('--save_per_steps', default=10000000, type=int)
    arg_parser.add_argument('--eval_per_steps', default=10, type=int)

    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--batch_size', default=32, type=int)
    arg_parser.add_argument('--grad_accum_iters', default=1, type=int)
    arg_parser.add_argument('--sample_length', default=None, type=int)
    arg_parser.add_argument('--em_schedule', type=str)
    arg_parser.add_argument('--dropout', default=0.001, type=float)
    arg_parser.add_argument('--pseudocount', default=0.001, type=float)
    arg_parser.add_argument('--hidden_states', default=32, type=int)
    arg_parser.add_argument('--block_size', default='', type=str)
    arg_parser.add_argument('--hidden_states_block_size', default='', type=str)
    arg_parser.add_argument('--eval_only', action='store_true')
    arg_parser.add_argument('--monarch', action='store_true')

    arg_parser.add_argument('--log_file', default='', type=str)

    args = arg_parser.parse_args()

    dist.init_process_group('nccl')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f'loading {args.dataset}...')
    dataset = torch.load(args.dataset, weights_only=False)
    train_data, dev_data, test_data = dataset["train"], dataset["dev"], dataset["test"]
    if (args.dataset).find('text8') != -1:
        vocab_size = 27
        sep_token_id = 0
    elif (args.dataset).find('lm1b') != -1:
        vocab_size = 8192
        sep_token_id = 6
    else:
        raise NotImplementedError

    if rank == 0:
        with open(args.log_file, 'a+') as fout:
            fout.write(str(vars(args)) + '\n')

    num_update_steps_per_epoch = (train_data.shape[0] // world_size) // (args.grad_accum_iters * args.batch_size)

    em_schedule = [x.split(',') for x in args.em_schedule.split(';') if x != '']
    if em_schedule[0][0] == "epoch":
        em_schedule = [(int(x[1]) * num_update_steps_per_epoch, float(x[2]), float(x[3]), x[4]) for x in em_schedule]
    elif em_schedule[0][0] == "step":
        em_schedule = [(int(x[1]), float(x[2]), float(x[3]), x[4]) for x in em_schedule]
    else:
        raise NotImplementedError

    decay_schedule = []
    for num_steps, decay_start, decay_end, decay_fn in em_schedule:
        decay_schedule.extend(generate_decay_schedule(num_steps, decay_start, decay_end, decay_fn))

    num_train_steps = len(decay_schedule)
    num_train_epochs = math.ceil(num_train_steps / num_update_steps_per_epoch)

    if em_schedule[0][0] == "epoch":
        assert num_train_epochs == sum([int(x[1]) for x in em_schedule[1:]])

    print("#####################################################")
    print(f'train size: {train_data.shape}')
    print(f'dev size: {dev_data.shape}')
    print(f'test size: {test_data.shape}')
    print("# of training examples", len(train_data))
    print("Mini batch size", args.batch_size)
    print("World size", world_size)
    print("Gradient accumulate", args.grad_accum_iters)
    print("Effective batch size", args.batch_size * world_size * args.grad_accum_iters)
    print("# of steps per epoch", num_update_steps_per_epoch)
    print("total # of epochs", num_train_epochs)
    print("total # of steps", num_train_steps)

    if args.hidden_states_block_size != '':
        hidden_states_block_size = [int(x) for x in args.hidden_states_block_size.split('_')]
        hidden_states = hidden_states_block_size[0]
        block_size = tuple(hidden_states_block_size[1:])
        print(f'hidden_states: {hidden_states}')
        print(f'block_size: {block_size}')
    else:
        hidden_states = args.hidden_states
        if args.block_size != '':
            block_size = tuple(int(x) for x in args.block_size.split('_'))
            if len(block_size) == 1:
                block_size = block_size[0]
            print('block_size', block_size)
        else:
            block_size = args.block_size
    print("#####################################################")

    train_hmm(rank, world_size,
        args.init_model_path, args.model_path, args.checkpoint, args.save_per_steps, args.eval_per_steps,
        train_data, dev_data, test_data,
        num_train_epochs, num_train_steps,
        args.batch_size, args.grad_accum_iters, args.sample_length,
        decay_schedule, args.log_file, args.dropout, args.pseudocount,
        hidden_states, block_size, vocab_size, sep_token_id,
        args.eval_only, args.monarch)
