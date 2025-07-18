import os
import sys
sys.path.append(os.getcwd())
import math
import torch

import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data.distributed import DistributedSampler

import pyjuice as juice

from pc_mini_em.src.utils import instantiate_from_config, collect_data_from_dsets, ProgressBar
from monarch_juice.layers.monarchlayer import create_monarch_layers, create_dense_layer
from monarch_juice.structures.HCLTMonarch import HCLTGeneral
from monarch_juice.ddp import ddp_setup

sys.setrecursionlimit(15000)

def load_data(dataset, patch_size, lossless, batch_size):
    data_config = f"pc_mini_em/configs/data/imagenet32{'_nopatch' if (patch_size == 32 and dataset == 'imagenet32') or (patch_size == 64 and dataset == 'imagenet64') else ''}_{'lossless' if lossless else 'lossy'}.yaml"
    data_config = OmegaConf.load(data_config)
    if dataset == 'imagenet32':
        pass
    elif dataset == 'imagenet64':
        data_config['params']['train']['target'] = 'pc_mini_em.src.data.ImageNet64Train'
        data_config['params']['validation']['target'] = 'pc_mini_em.src.data.ImageNet64Validation'
    if batch_size > 0:
        data_config["params"]["batch_size"] = batch_size
    if (dataset == 'imagenet32' and patch_size < 16) or (dataset == 'imagenet64'):
        for i, transform in enumerate(data_config['params']['train']['params']['transform_fns']):
            if 'Patchify' in transform['target']:
                data_config['params']['train']['params']['transform_fns'][i]['params']['patch_size'] = patch_size
                data_config['params']['train']['params']['transform_fns'][i]['target'] = 'pc_mini_em.src.transforms.Patchify'
        for i, transform in enumerate(data_config['params']['validation']['params']['transform_fns']):
            if 'Patchify' in transform['target']:
                print('patchified')
                data_config['params']['validation']['params']['transform_fns'][i]['params']['patch_size'] = patch_size
                data_config['params']['validation']['params']['transform_fns'][i]['target'] = 'pc_mini_em.src.transforms.Patchify'
    dsets = instantiate_from_config(data_config)
    dsets.prepare_data()
    dsets.setup()

    train_sampler = DistributedSampler(dsets.datasets["train"], shuffle = True) 
    train_loader = dsets._train_dataloader(sampler = train_sampler)
    valid_sampler = DistributedSampler(dsets.datasets["validation"], shuffle = True)
    val_loader = dsets._val_dataloader(sampler=valid_sampler)

    return dsets, train_loader, val_loader

def construct_model(dsets, lossless, layer_type, hidden_size):
    model_config = f"pc_mini_em/configs/model/hclt_256{'_lossless' if lossless else ''}.yaml"
    model_config = OmegaConf.load(model_config)
    model_kwargs = {}
    for k, v in model_config["params"].items():
        if isinstance(v, str) and v.startswith("__train_data__:"):
            num_samples = int(v.split(":")[1])
            data = collect_data_from_dsets(dsets, num_samples = num_samples, split = "train")
            model_config["params"].pop(k, None)
            model_kwargs[k] = data.cuda()
    print("> Constructing PC...")
    print(model_config)
    x_shape = model_kwargs['x'].shape[1:]
    model_config['params']['num_latents'] = hidden_size
    if layer_type == 'dense':
        model_config['params']['block_size'] = model_config['params']['num_latents']
    elif layer_type == 'monarch':
        model_config['params']['block_size'] = int(model_config['params']['num_latents'] ** 0.5)
    match layer_type:
        case 'dense':
            layer_fn = create_dense_layer 
        case 'monarch':
            layer_fn = create_monarch_layers
    model_config['params']['homogeneous_inputs'] = True
    if 'input_dist' in model_config['params']:
        input_dist = instantiate_from_config(model_config['params']['input_dist'])
        del model_config['params']['input_dist']
    else:
        input_dist = None
    ns = HCLTGeneral(**model_config["params"], **model_kwargs, 
                        input_dist=input_dist,
                        layer_fn=layer_fn)
    return ns

def construct_and_train_pc(rank, world_size, args):
    device_list = args.devices.split(',')
    torch.set_num_threads(8)

    store = dist.FileStore('filestore')
    ddp_setup(rank, world_size, args.port, store=store)

    device = torch.device(f'cuda:{device_list[rank]}')

    dsets, train_loader, val_loader = load_data(args.dataset, args.patch_size, args.lossless, args.batch_size)

    if rank == 0:
        if args.load_location is not None:
            ns = juice.load(args.load_location)
        else:
            ns = construct_model(dsets, args.lossless,
                                args.layer_type, args.hidden_size)
        store.set('filename', args.save_location)
        
    dist.barrier()

    if rank != 0:
        ns = juice.load(store.get('filename').decode("utf-8"))

    dist.barrier()

    pc = juice.compile(ns)
    if rank == 0:
        pc.print_statistics()
    pc.to(device)

    dist.barrier()

    schedule = [int(x) for x in args.minibatch_schedule.split(",")]
    sizes, epochs = schedule[::2], schedule[1::2]
    tot_epochs = sum(epochs)
    era, epochs_in_era = 0, 0

    if rank == 0:
        progress_bar = ProgressBar(tot_epochs, len(train_loader), ["LL"], cumulate_statistics = True)

    pc.init_param_flows(flows_memory = 0.0)
    step_count = 0

    # initial validation
    val_ll = 0.0
    for x in val_loader:
        x = x.to(device)
        with torch.cuda.device(f'cuda:{device_list[rank]}'):
            lls = pc(x)
        val_ll += lls.mean().detach().cpu().numpy().item()
    stats = torch.tensor([val_ll]).to(device)
    dist.all_reduce(stats, op = dist.ReduceOp.SUM)
    val_ll = stats[0].item() / world_size / len(val_loader)
    print('initial val ll', val_ll)


    for epoch in range(tot_epochs):
        if rank == 0:
            progress_bar.new_epoch_begin()
        if sizes[era] == 0:
            niters_per_update = len(train_loader) # full batch
        else:
            niters_per_update = sizes[era] // world_size // args.batch_size
        update_count = 0

        for x in train_loader:
            x = x.to(device)
            with torch.cuda.device(f'cuda:{device_list[rank]}'):
                lls = pc(x,propagation_alg = "LL")
                pc.backward(x, flows_memory = 1.0, allow_modify_flows = False,
                            propagation_alg = "LL", logspace_flows = True)
            curr_ll = lls.mean().detach().cpu().numpy().item()
            stats = torch.tensor([curr_ll]).to(device)
            dist.all_reduce(stats, op = dist.ReduceOp.SUM)
            if rank == 0:
                curr_ll = stats[0].item() / world_size
                progress_bar.new_batch_done([curr_ll])
            
            step_count += 1
            if step_count >= niters_per_update:
                step_count = 0
                update_count += 1

                dist.barrier()
                dist.all_reduce(pc.param_flows, op = dist.ReduceOp.SUM)
                for layer in pc.input_layer_group:
                    dist.all_reduce(layer.param_flows, op = dist.ReduceOp.SUM)
                dist.barrier()
                current_time = (epoch + update_count * sizes[era]/len(train_loader.dataset)) / tot_epochs
                # Cosine LR decay
                step_size = 0.5 * (1 + math.cos(math.pi*current_time))
                with torch.cuda.device(f'cuda:{device_list[rank]}'):
                    pc.mini_batch_em(step_size = step_size, pseudocount = 1e-8)
                pc.init_param_flows(flows_memory = 0.0)
        
        epochs_in_era = epochs_in_era + 1
        if epochs_in_era > epochs[era]:
            epochs_in_era = 0
            era = era + 1

        val_ll = 0.0
        for x in val_loader:
            x = x.to(device)
            with torch.cuda.device(f'cuda:{device_list[rank]}'):
                lls = pc(x)
            val_ll += lls.mean().detach().cpu().numpy().item()
        stats = torch.tensor([val_ll]).to(device)
        dist.all_reduce(stats, op = dist.ReduceOp.SUM)

        if rank == 0:
            train_ll = progress_bar.epoch_ends()[0]
            val_ll = stats[0].item() / world_size / len(val_loader)
            print(f"[Epoch {epoch+1}/{tot_epochs}][train LL: {train_ll:.2f}; val LL: {val_ll:.2f}]")
        dist.barrier()
    if rank == 0:
        if args.save_location is not None:
            juice.save(f'{args.save_location}', pc)