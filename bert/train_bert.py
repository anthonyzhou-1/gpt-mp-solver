from common.utils import HDF5Dataset
from torch.utils.data import DataLoader
from equations.PDEs import *
import random
import wandb
from datetime import datetime
import argparse
from MaskedWrapper import MaskedWrapper
from models import BERT
from PDETokenizer import PDETokenizer

def get_data(u_super, steps, tw):
    """
    Get data for training
    Args:
        x (torch.Tensor): input data in shape (batch_size, t_res, x_res)
        steps (list): list of possible starting points for the window
        tw (int): window size
    Returns:
        x (torch.Tensor): input data in shape (batch_size, tw, x_res)
    """
    random_steps = random.choices(steps, k=u_super.shape[0])

    x = torch.Tensor()
    for i, step in enumerate(random_steps):
        d = u_super[i, step:step + tw, :]
        x = torch.cat((x, d[None, :]), dim=0)
    return x

def dict2tensor(d: dict) -> torch.Tensor:
    """
    Converts a dictionary to a tensor
    Args:
        d (dict): dictionary
    Returns:
        t (torch.Tensor): tensor
    """
    tensors = []
    for k, v in d.items():
        tensors.append(v.unsqueeze(0))
    return torch.cat(tensors, dim=0)

def train(args: argparse,
          model: torch.nn.Module,
          optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          loader: DataLoader,
          tokenizer: PDETokenizer,
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
        reconstruction_losses = []
        sysID_losses = []
        grad_norms = []
        for (u_base, u_super, x, variables) in loader:
            # Reset gradients
            optimizer.zero_grad()

            # Create random window of data (batch_size, tw, nx)
            u_window = get_data(u_super, steps, tw)
            # Tokenize data to shape (batch_size, num_tokens, d_in)
            u_tokens = tokenizer.forward(u_window).to(device)

            # Create labels for system ID in shape (batch_size, num_vars)
            var_labels = torch.transpose(dict2tensor(variables), 0, 1).to(device)

            # Forward pass
            loss, reconstruction_loss, sysID_loss = model(u_tokens, variables = var_labels)

            # Backward pass
            loss.backward()
            losses.append(loss.detach() / batch_size)
            reconstruction_losses.append(reconstruction_loss.detach() / batch_size)
            sysID_losses.append(sysID_loss.detach() / batch_size)
            
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
        losses_out, grad_norms_out = torch.mean(losses), torch.mean(grad_norms)
        print(f'Training Loss (progress: {i / t_res:.2f}): {losses_out}')
        #wandb.log({"train/loss": losses_out,
                    #"metrics/grad_norm": grad_norms_out})
  
def prepare_data(path, pde, args, mode='train'):
    dataset = HDF5Dataset(path, pde=pde, mode=mode, base_resolution=args.base_resolution, super_resolution=args.super_resolution)
    shuffle = False
    if(mode == 'train'):
         shuffle = True
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=shuffle)
    return dataloader

def main(args: argparse):
    #run = wandb.init(project="bert-pde",
    #        config=vars(args))
    device = 'cuda'
    pde = CE(device=device)
    train_string = "/home/cmu/anthony/gpt-mp-solver/data/largeCE_train_E3.h5"
    train_loader = prepare_data(train_string, pde, args, mode='train')

    pde.tmin = train_loader.dataset.tmin
    pde.tmax = train_loader.dataset.tmax
    pde.grid_size = args.base_resolution
    pde.dt = train_loader.dataset.dt

    t_res, x_res = args.base_resolution
    
    if args.mode == 'constant':
        assert args.segment_len== args.num_tokens/args.num_segments, f'Segment length {args.segment_len} must be equal to number of tokens {args.num_tokens} divided by number of segments {args.num_segments}' 
        assert args.d_in == args.num_segments*x_res/args.num_tokens, f'Input dimension {args.d_in} must be equal to number of segments {args.num_segments} times number of grid points {x_res} divided by number of tokens {args.num_tokens}'
    
    bert = BERT(d_in=args.d_in,
             d_model = args.d_model,
             nhead = args.nhead,
             num_layers= args.num_layers,
             segment_len = args.segment_len,
             num_segments = args.num_segments,
             dropout = args.dropout,
             )
    
    reconstruction_net = nn.Linear(in_features=args.d_model, out_features=args.d_in)
    variable_net = nn.Linear(in_features=args.d_model, out_features=args.n_vars)

    model = MaskedWrapper(net=bert,
                          reconstruction_net=reconstruction_net,
                          variable_net=variable_net,
                          gamma = args.gamma,
                            mask_prob = args.mask_prob,
                            replace_prob = args.replace_prob,
                            random_token_prob = args.random_token_prob).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of BERT parameters: {params}')
    tokenizer = PDETokenizer(num_tokens=args.num_tokens, nx=x_res, mode=args.mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.min_lr, betas=(args.beta1, args.beta2), fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=args.max_lr, 
                                                        steps_per_epoch= t_res * len(train_loader), 
                                                        epochs=args.num_epochs, 
                                                        pct_start=args.pct_start, 
                                                        anneal_strategy='cos', 
                                                        final_div_factor=args.max_lr/args.min_lr)

    ## Training
    min_val_loss = 10e10
    num_epochs = args.num_epochs

    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    save_path= f'models/BERT_{args.experiment}_time{timestring}.pt'
    save_path_log = f'logs/log_{args.experiment}_time{timestring}.txt'
    verbose = True

    with open(save_path_log, 'w') as f:
        print(vars(args), file=f)

    for epoch in range(num_epochs):
        train(args, model=model, optimizer=optimizer, scheduler=scheduler, loader=train_loader, tokenizer=tokenizer, device=device)
        # Save model
        torch.save(model.state_dict(), save_path)
        print(f"Saved model at {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an PDE solver')

    # PDE
    parser.add_argument('--experiment', type=str, default='',
                        help='Experiment for PDE solver should be trained: [E1, E2, E3, WE1, WE2, WE3]')

    # BERT parameters
    parser.add_argument('--batch_size', type=int, default=16,
            help='Number of samples in each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='Number of training epochs')
    parser.add_argument('--d_in', type=int, default=5,
            help='Input dimension of BERT')
    parser.add_argument('--d_model', type=int, default=128,
            help='Model dimension of BERT')
    parser.add_argument('--nhead', type=int, default=4,
            help='Number of heads in BERT')
    parser.add_argument('--num_layers', type=int, default=2,
            help='Number of layers in BERT')
    parser.add_argument('--segment_len', type=int, default=20,
            help='Length of each PDE segment')
    parser.add_argument('--num_segments', type=int, default=25,
            help='Number of PDE timesteps/segments')
    parser.add_argument('--dropout', type=float, default=0,
                help='Dropout probability in BERT')
    
    # Reconstruction parameters
    parser.add_argument('--mask_prob', type=float, default=0.15,
            help='Probability to mask out a token')
    parser.add_argument('--replace_prob', type=float, default=0.9,
            help='Probability to replace a masked token with 0')
    parser.add_argument('--random_token_prob', type=float, default=0.1,
            help='Probability to replace a masked token with a random token')
    
    # SysID parameters
    parser.add_argument('--gamma', type=float, default=0.1,
            help='Weight of variable loss')
    parser.add_argument('--n_vars', type=int, default=3,
            help='Number of variables in PDE')
    
    # Optimizer parameters
    parser.add_argument('--min_lr', type=float, default=1e-4,
            help='Minimum learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
            help='Maximum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
            help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98,
            help='Adam beta2')
    
    # Scheduler parameters
    parser.add_argument('--pct_start', type=float, default=0.1,
            help='Percentage of training to increase learning rate')
    
    # Tokenizer parameters
    parser.add_argument('--num_tokens', type=int, default=500,
            help='Number of tokens to generate from a PDE time/spatial sequence')
    parser.add_argument('--mode', type=str, default='constant',
                help='Mode for tokenization: [constant, bicubic, fourier]')
    

    # Base resolution and super resolution
    parser.add_argument('--base_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 100], help="PDE base resolution on which network is applied")
    parser.add_argument('--super_resolution', type=lambda s: [int(item) for item in s.split(',')],
            default=[250, 200], help="PDE super resolution for calculating training and validation loss")

    args = parser.parse_args()
    main(args)