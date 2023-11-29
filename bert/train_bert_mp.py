from common.utils import HDF5Dataset
from torch.utils.data import DataLoader
from equations.PDEs import *
import random
import wandb
from datetime import datetime
import argparse
from MaskedWrapper import MaskedWrapper
from models import BERT

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os

def setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def sync_tensor_across_gpus(t):
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t) 
    return torch.cat(gather_t_tensor, dim=0)

def cleanup():
    destroy_process_group()

def get_data(u_super, steps, tw):
    """
    Get data for training
    Args:
        x (torch.Tensor): input data
        steps (list): list of timesteps
        tw (int): number of timesteps in one trajectory
    Returns:
        x (torch.Tensor): input data
        x_masked (torch.Tensor): masked input data
        x_masked_out (torch.Tensor): masked output data
    """
    random_steps = random.choices(steps, k=u_super.shape[0])

    x = torch.Tensor()
    for i, step in enumerate(random_steps):
        d = u_super[i, step:step + tw, :]
        x = torch.cat((x, d[None, :]), dim=0)
    x = torch.flatten(x, start_dim=1, end_dim=2).unsqueeze(-1)
    return x

def train(args: argparse,
          epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
    Returns:
        None
    """
    model.train()
    t_res, x_res = args.base_resolution
    tw = args.num_segments
    batch_size = args.batch_size

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.
    steps = [t for t in range(0, t_res - tw)]
    for i in range(t_res):
        losses = []
        grad_norms = []
        loader.sampler.set_epoch(epoch*t_res + i)
        for (u_base, u_super, x, variables) in loader:
            # Reset gradients
            optimizer.zero_grad()

            # Create data to be masked
            x = get_data(u_super, steps, tw).to(device)

            # Forward pass
            loss = model(x)

            # Backward pass
            loss.backward()
            losses.append(loss.detach() / batch_size)
            optimizer.step()

            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()
            grad_norms.append(grad_norm / batch_size)

            scheduler.step()

        losses = torch.stack(losses)
        grad_norms = torch.stack(grad_norms)
        losses_out, grad_norms_out = torch.mean(sync_tensor_across_gpus(losses.unsqueeze(0))), torch.mean(sync_tensor_across_gpus(grad_norms.unsqueeze(0)))
        
        if device == 0:
            print(f'Training Loss (progress: {i / t_res:.2f}): {losses_out}')
            wandb.log({"train/loss": losses_out,
                        "metrics/grad_norm": grad_norms_out})
  
def prepare_data(rank, world_size, path, pde, args, mode='train'):
    dataset = HDF5Dataset(path, pde=pde, mode=mode, base_resolution=args.base_resolution, super_resolution=args.super_resolution)
    shuffle = False
    if(mode == 'train'):
         shuffle = True
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=False, num_workers=0, shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size, args: argparse):
    setup(rank, world_size)
    if rank == 0:
        run = wandb.init(project="bert-pde",
                config=vars(args))
        
    pde = CE(device=rank)
    train_string = "/home/cmu/anthony/gpt-mp-solver/data/largeCE_train_E3.h5"
    train_loader = prepare_data(rank, world_size, train_string, pde, args, mode='train')

    pde.tmin = train_loader.dataset.tmin
    pde.tmax = train_loader.dataset.tmax
    pde.grid_size = args.base_resolution
    pde.dt = train_loader.dataset.dt

    t_res, x_res = args.base_resolution

    bert = BERT(d_in=args.d_in,
             d_out = args.d_out,
             d_model = args.d_model,
             nhead = args.nhead,
             num_layers= args.num_layers,
             segment_len = args.segment_len,
             num_segments = args.num_segments,
             )
    model = MaskedWrapper(net=bert,
                            mask_prob = args.mask_prob,
                            replace_prob = args.replace_prob,
                            random_token_prob = args.random_token_prob).to(rank)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of BERT parameters: {params}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr, betas=(args.beta1, args.beta2), fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.max_lr, 
                                                        steps_per_epoch= t_res * len(train_loader), 
                                                        epochs=args.num_epochs, 
                                                        pct_start=args.pct_start, 
                                                        anneal_strategy='cos', 
                                                        final_div_factor=args.max_lr/args.min_lr)

    # Multiprocessing
    model = DDP(model, device_ids=[rank])

    ## Training
    min_val_loss = 10e10
    num_epochs = args.num_epochs

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    save_path= f'models/BERT_{args.experiment}_time{timestring}.pt'
    save_path_log = f'logs/log_{args.experiment}_time{timestring}.txt'
    verbose = True

    if rank == 0:
        with open(save_path_log, 'w') as f:
            print(vars(args), file=f)

    for epoch in range(num_epochs):
        train(args, model=model, optimizer=optimizer, scheduler=scheduler, loader=train_loader, device=rank)
        # Save model
        if rank == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at {save_path}\n")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # BERT parameters
    parser.add_argument('--batch_size', type=int, default=32,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='Number of training epochs')
    parser.add_argument('--d_in', type=int, default=1,
            help='Input dimension of BERT')
    parser.add_argument('--d_out', type=int, default=1,
            help='Output dimension of BERT')
    parser.add_argument('--d_model', type=int, default=128,
            help='Model dimension of BERT')
    parser.add_argument('--nhead', type=int, default=4,
            help='Number of heads in BERT')
    parser.add_argument('--num_layers', type=int, default=2,
            help='Number of layers in BERT')
    parser.add_argument('--segment_len', type=int, default=100,
            help='Length of each PDE segment')
    parser.add_argument('--num_segments', type=int, default=5,
            help='Number of PDE timesteps/segments')
    parser.add_argument('--mask_prob', type=float, default=0.15,
            help='Probability to mask out a token')
    parser.add_argument('--replace_prob', type=float, default=0.9,
            help='Probability to replace a masked token with 0')
    parser.add_argument('--random_token_prob', type=float, default=0.1,
            help='Probability to replace a masked token with a random token')
    parser.add_argument('--min_lr', type=float, default=1e-4,
            help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
            help='Maximum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
            help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
            help='Adam beta2')
    parser.add_argument('--pct_start', type=float, default=0.1,
            help='Percentage of training to increase learning rate')
    

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 100], help="PDE base resolution on which network is applied")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")

    args = parser.parse_args()
    world_size = 8
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    ) 