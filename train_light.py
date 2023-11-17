from common.utils import HDF5Dataset
from torch.utils.data import DataLoader
from experiments.models_gpt import GPT
from common.utils import GraphCreator
from equations.PDEs import *
import random
from experiments.models_gnn import MP_PDE_Solver
import wandb
from experiments.train_helper import *
from datetime import datetime
import torch_geometric
import argparse

import os

def train_pf(args: argparse,
          epoch: int,
          model_gnn: torch.nn.Module,
          optimizer_gnn: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator,
          criterion: torch.nn.modules.loss,
          model_gpt: torch.nn.Module = None,
          optimizer_gpt: torch.optim = None,
          scheduler_gpt: torch.optim.lr_scheduler = None,
          curriculum: Curriculum = None,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is done over each batch in the dataloader for each timestep sequentially
    This allows for curriculum learning on autoregressively generated datapoints
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    if model_gpt is not None:
        model_gpt.train()
    model_gnn.train()

    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    training_loop_pf(model_gnn, unrolling, args.batch_size, optimizer_gnn, loader, graph_creator, criterion, model_gpt, optimizer_gpt, scheduler_gpt, epoch, curriculum, device)


def train(args: argparse,
          epoch: int,
          model_gnn: torch.nn.Module,
          optimizer_gnn: torch.optim,
          loader: DataLoader,
          graph_creator: GraphCreator,
          criterion: torch.nn.modules.loss,
          model_gpt: torch.nn.Module = None,
          optimizer_gpt: torch.optim = None,
          scheduler_gpt: torch.optim.lr_scheduler = None,
          device: torch.cuda.device="cpu") -> None:
    """
    Training loop.
    Loop is over the mini-batches and for every batch we pick a random timestep.
    This is done for the number of timesteps in our training sample, which covers a whole episode.
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    print(f'Starting epoch {epoch}...')
    if model_gpt is not None:
        model_gpt.train()
    model_gnn.train()

    max_unrolling = epoch if epoch <= args.unrolling else args.unrolling
    unrolling = [r for r in range(max_unrolling + 1)]

    # Loop over every epoch as often as the number of timesteps in one trajectory.
    # Since the starting point is randomly drawn, this in expectation has every possible starting point/sample combination of the training data.
    # Therefore in expectation the whole available training information is covered.

    for i in range(graph_creator.t_res):
        losses, norms_gnn, norms_gpt, lr_gpt = training_loop(model_gnn, unrolling, args.batch_size, optimizer_gnn, loader, graph_creator, criterion, model_gpt, optimizer_gpt, scheduler_gpt, device)
        losses_out, gnn_norms_out, gpt_norms_out, lr_gpt_out = torch.mean(losses), torch.mean(norms_gnn), torch.mean(norms_gpt), torch.mean(lr_gpt)
        
        print(f'Training Loss (progress: {i / graph_creator.t_res:.2f}): {losses_out}')
        print(f'GNN Norm: {gnn_norms_out}, GPT Norm: {gpt_norms_out}')
        print(f"Learning rate GPT: {lr_gpt_out}")
        wandb.log({"train/loss": losses_out,
                    "metrics/grad_gnn_norm": gnn_norms_out,
                    "metrics/grad_gpt_norm": gpt_norms_out,
                    "metrics/lr_gpt": lr_gpt_out,})

def test(args: argparse,
         model_gnn: torch.nn.Module,
         loader: DataLoader,
         graph_creator: GraphCreator,
         criterion: torch.nn.modules.loss,
         model_gpt: torch.nn.Module = None,
         device: torch.cuda.device="cpu",
         verbose: bool=False) -> torch.Tensor:
    """
    Test routine
    Both step wise and unrolled forward losses are computed
    and compared against low resolution solvers
    step wise = loss for one neural network forward pass at certain timepoints
    unrolled forward loss = unrolling of the whole trajectory
    Args:
        args (argparse): command line inputs
        pde (PDE): PDE at hand [CE, WE, ...]
        model (torch.nn.Module): neural network PDE solver
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: unrolled forward loss
    """
    model_gnn.eval()
    if model_gpt is not None:
        model_gpt.eval()

   # first we check the losses for different timesteps (one forward prediction array!)
    steps = [t for t in range(graph_creator.tw, graph_creator.t_res-graph_creator.tw + 1)]
    losses = test_timestep_losses(model_gnn=model_gnn,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  model_gpt=model_gpt,
                                  device=device,
                                  verbose=verbose)

    # next we test the unrolled losses
    losses = test_unrolled_losses(model_gnn=model_gnn,
                                  steps=steps,
                                  batch_size=args.batch_size,
                                  nr_gt_steps=args.nr_gt_steps,
                                  nx_base_resolution=args.base_resolution[1],
                                  loader=loader,
                                  graph_creator=graph_creator,
                                  criterion=criterion,
                                  model_gpt= model_gpt,
                                  device=device,
                                  verbose=verbose)

    return torch.mean(losses)

def prepare_data(path, pde, args, mode='train'):
    dataset = HDF5Dataset(path, pde=pde, mode=mode, base_resolution=args.base_resolution, super_resolution=args.super_resolution)
    shuffle = False
    if(mode == 'train'):
         shuffle = True
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=shuffle)
    return dataloader

def main(args: argparse):
    use_gpt = args.use_gpt
    run = wandb.init(project="gpt-mp-pde-solver",
            config=vars(args))

    ## GNN Initialization
    device = 'cuda'
    pde = CE(device=device)
    train_string = f'data/{pde}_train_{args.experiment}.h5'
    valid_string = f'data/{pde}_valid_{args.experiment}.h5'
    test_string = f'data/{pde}_test_{args.experiment}.h5'

    train_loader = prepare_data(train_string, pde, args, mode='train')
    valid_loader = prepare_data(valid_string, pde, args, mode='valid')
    test_loader = prepare_data(test_string, pde, args, mode='test')

    pde.tmin = train_loader.dataset.tmin
    pde.tmax = train_loader.dataset.tmax
    pde.grid_size = args.base_resolution
    pde.dt = train_loader.dataset.dt

    eq_variables = {}
    if args.experiment == 'E2':
            print('Beta parameter added to the GNN solver')
            eq_variables['beta'] = 0.2
    elif args.experiment  == 'E3':
            print('Alpha, beta, and gamma parameter added to the GNN solver')
            eq_variables['alpha'] = 3.
            eq_variables['beta'] = 0.4
            eq_variables['gamma'] = 1.
    elif (args.experiment  == 'WE3'):
            print('Boundary parameters added to the GNN solver')
            eq_variables['bc_left'] = 1
            eq_variables['bc_right'] = 1

    graph_creator = GraphCreator(   pde=pde,
                                    neighbors=args.neighbors,
                                    time_window=args.time_window,
                                    t_resolution=args.base_resolution[0],
                                    x_resolution=args.base_resolution[1],
                                    use_gpt=use_gpt).to(device)

    model_gnn = MP_PDE_Solver(pde=pde,
                            time_window=graph_creator.tw,
                            hidden_features=args.hidden_features,
                            hidden_layer=args.hidden_layer,
                            eq_variables=eq_variables,
                            use_gpt=use_gpt).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model_gnn.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of GNN parameters: {params}')

    ## GPT Initialization
    if(use_gpt):
        model_gpt = GPT(vars(args)).to(device)
    else:
        model_gpt = None
        optimizer_gpt = None
        scheduler_gpt = None

    optimizer_gnn = torch.optim.AdamW(model_gnn.parameters(), lr=args.lr_gnn, fused=True)
    scheduler_gnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_gnn, milestones=[args.unrolling, 5, 10, 15], gamma=args.lr_gnn_decay)

    if(use_gpt):
        optimizer_gpt = torch.optim.AdamW(model_gpt.parameters(), lr=args.min_lr_gpt, betas=(args.beta1, args.beta2), fused=True)
        scheduler_gpt = torch.optim.lr_scheduler.OneCycleLR(optimizer_gpt, 
                                                            max_lr=args.lr_gpt, 
                                                            steps_per_epoch=graph_creator.t_res * len(train_loader), 
                                                            epochs=args.num_epochs, 
                                                            pct_start=args.pct_start, 
                                                            anneal_strategy='cos', 
                                                            final_div_factor=args.lr_gpt/args.min_lr_gpt)
        curriculum = Curriculum(args.num_epochs*graph_creator.t_res, args.delta)


    ## Training
    min_val_loss = 10e10
    criterion = torch.nn.MSELoss(reduction="sum")
    num_epochs = args.num_epochs

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'

    save_path_gpt = f'models/GPT_{args.experiment}_time{timestring}.pt'
    save_path_gnn = f'models/GNN_{args.experiment}_time{timestring}.pt'
    save_path_log = f'logs/log_{args.experiment}_time{timestring}.txt'

    verbose = True

    with open(save_path_log, 'w') as f:
        print(vars(args), file=f)

    for epoch in range(num_epochs):
        train_pf(args, epoch, model_gnn, optimizer_gnn, train_loader, graph_creator, criterion, model_gpt, optimizer_gpt, scheduler_gpt, curriculum, device=device)
        print("Evaluation on validation dataset:")

        val_loss = test(args, model_gnn, valid_loader, graph_creator, criterion, model_gpt, device=device, verbose=verbose)
        val_loss_out = torch.mean(val_loss)

        print(f"Validation Loss: {val_loss_out}\n")
        wandb.log({
            "valid/loss": val_loss_out,
        })

        if(val_loss_out < min_val_loss):
            print("Evaluation on test dataset:")
            test_loss = test(args, model_gnn, test_loader, graph_creator, criterion, model_gpt, device=device, verbose=verbose)
            test_loss_out = torch.mean(test_loss)
            print(f"Test Loss: {test_loss_out}\n")
            wandb.log({
                "test/loss": test_loss,
            })
            # Save model
            torch.save(model_gnn.state_dict(), save_path_gnn)
            print(f"Saved model at {save_path_gnn}\n")
            if(use_gpt):
                torch.save(model_gpt.state_dict(), save_path_gpt)
                print(f"Saved model at {save_path_gpt}\n")
            min_val_loss = val_loss_out
        scheduler_gnn.step()
    print(f"Test loss: {test_loss}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # GPT Model
    parser.add_argument('--model', type=str, default='GNN',
                        help='Model used as PDE solver: [GNN, BaseCNN]')
    parser.add_argument('--use_gpt', type=bool, default=True,
                    help='Flag to use gpt or not')

    # GNN parameters
    parser.add_argument('--batch_size', type=int, default=32,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='Number of training epochs')
    parser.add_argument('--lr_gnn', type=float, default=1e-4,
            help='Learning rate')
    parser.add_argument('--lr_gnn_decay', type=float,
                        default=0.4, help='multistep lr decay')
    parser.add_argument('--hidden_features', type=int,
                    default=128, help='hidden features for GNN')
    parser.add_argument('--hidden_layer', type=int,
                    default=6, help='Number of hidden layers for GNN')
    parser.add_argument('--parameter_ablation', type=eval, default=False,
                        help='Flag for ablating MP-PDE solver without equation specific parameters')

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 100], help="PDE base resolution on which network is applied")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")
    parser.add_argument('--neighbors', type=int,
                        default=3, help="Neighbors to be considered in GNN solver")
    parser.add_argument('--time_window', type=int,
                        default=25, help="Time steps to be considered in GNN solver")
    parser.add_argument('--unrolling', type=int,
                        default=1, help="Unrolling which proceeds with each epoch")
    parser.add_argument('--nr_gt_steps', type=int,
                        default=2, help="Number of steps done by numerical solver")
    
    # GPT params
    parser.add_argument('--n_x', type=int, default=100,
                        help='Number of discretization points')
    parser.add_argument('--block_size', type=int, default=250,
                        help='Block size for GPT model')
    parser.add_argument('--n_layer', type=int, default=4,
                        help='Number of layers for GPT model')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of heads for GPT model')
    parser.add_argument('--n_embd', type=int, default=256,
                        help='Hidden dimension for GPT model')
    parser.add_argument('--n_gnn', type=int, default=100,
                        help='Output dimension to input as GNN embeddings')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout for GPT model')
    parser.add_argument('--bias', type=bool, default=False,
                        help='Flag to use bias for LayerNorm and Linear Layers')
    parser.add_argument('--lr_gpt', type=float, default=1e-3,
                        help='Max learning rate for GPT model')
    parser.add_argument('--pct_start', type=float, default=0.1,
                        help='Pct of steps to use as warmup for GPT model')
    parser.add_argument('--min_lr_gpt', type=float, default=1e-4,
                        help='Min learning rate for GPT model')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for GPT model')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='Beta2 for GPT model')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Gradient clipping for GPT model")
    parser.add_argument('--delta', type=float, default=0.2,
                        help="Delta for curriculum learning. Determines steepness of autogressive/teacher forcing")

    # Misc
    parser.add_argument('--print_interval', type=int, default=20,
            help='Interval between print statements')
    parser.add_argument('--log', type=eval, default=False,
            help='pip the output to log file')

    args = parser.parse_args()
    main(args)