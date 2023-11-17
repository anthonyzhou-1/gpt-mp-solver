import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from common.utils import *
from equations.PDEs import *
import wandb
from tqdm import tqdm

def training_loop_pf(model_gnn: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer_gnn: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  model_gpt: torch.nn.Module = None,  
                  optimizer_gpt: torch.optim = None,
                  scheduler_gpt: torch.optim.lr_scheduler = None,
                  epoch: int = 0,
                  curriculum: Curriculum = None,
                  device: torch.cuda.device="cpu",
                  mode = "GNN",) -> torch.Tensor:
    """
    One training epoch starting at the beginning of a sequence and progressing forward in time
    Loops through each training sample ~t_res times using curriculum learning
    Args:
        model (torch.nn.Module): neural network PDE solver
        gpt (torch.nn.Module): GPT PDE context generator
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """
    for (u_base, u_super, x, variables) in tqdm(loader, desc="epoch: " + str(epoch)):
        losses = []
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        
        # Create cache of autoregressive predictions
        cache = torch.clone(u_super).to(device)

        # Initialize a counter for curriculum learning. Equal across batches, increases as the epoch and timestep increases
        step_c = epoch*graph_creator.t_res
        
        for step in tqdm(steps, desc="Steps"):
            # form list of steps for each batch entry
            same_steps = [step] * batch_size

            # Reset gradients
            optimizer_gnn.zero_grad()
            if model_gpt is not None:
                optimizer_gpt.zero_grad()

            data, labels = graph_creator.create_data(u_super, same_steps)

            # Create first graph to propagate at t
            if mode == 'GNN':
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
            else:
                data, labels = data.to(device), labels.to(device)
            
            #Create first data to generate embeddings at t
            if model_gpt is not None:
                curriculum_prob = curriculum.get_prob(step_c)
                data_gpt = generate_gpt_data(u_super, cache, same_steps, curriculum_prob, graph_creator.tw).to(device)

            # Unrolling of the equation which serves as input at the current step
            # This is the pushforward trick!!!
            with torch.no_grad():
                for _ in range(unrolled_graphs):

                    # Forward pass of GPT model to make an embedding at t. Adds embedding to graph
                    if model_gpt is not None:
                        embeddings = model_gpt(data_gpt)
                        graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)

                    # Increments steps by tw and creates ground truth for t+1
                    same_steps = [rs + graph_creator.tw for rs in same_steps]
                    _, labels = graph_creator.create_data(u_super, same_steps)

                    if mode == 'GNN':
                        # Makes a prediction for state at t+1 and creates new graph with predicted state at t+1 as input and true state at t+2 as label
                        pred = model_gnn(graph)
                        graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)

                        # Updates data for GPT model at t+1 and puts prediction into cache
                        pred_tensor = graph2tensor(pred, batch_size, graph_creator.x_res)
                        cache[:, same_steps[0]-graph_creator.tw:same_steps[0], :] = pred_tensor.detach()
                        data_gpt = update_gpt_data(data_gpt, pred_tensor, same_steps, graph_creator.tw).to(device)
                    else:
                        data = model_gnn(data)
                        labels = labels.to(device)

            # Forward pass of GPT model to make an embedding at t+1. Adds embedding to graph
            if model_gpt is not None:
                embeddings = model_gpt(data_gpt)
                graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)

            # Makes a prediction for state at t+2 and computes loss with true state at t+2
            # Caches prediction to use for autoregression
            if mode == 'GNN':
                pred = model_gnn(graph)
                loss = criterion(pred, graph.y)
                pred_tensor = graph2tensor(pred, batch_size, graph_creator.x_res)
                cache[:, same_steps[0]: same_steps[0] + graph_creator.tw, :] = pred_tensor.detach()
            else:
                pred = model_gnn(data)
                loss = criterion(pred, labels)

            # Backpropagation and stepping the GNN optimizer
            loss = torch.sqrt(loss)
            loss.backward()
            losses.append(loss.detach() / batch_size)
            optimizer_gnn.step()
            # Stepping the GPT optimizer and scheduler
            if model_gpt is not None:
                optimizer_gpt.step()
                scheduler_gpt.step()
            
            # increment counter for curriculum
            step_c = step_c + 1

        losses = torch.mean(torch.stack(losses))
        wandb.log({"train/loss": losses})

def training_loop(model_gnn: torch.nn.Module,
                  unrolling: list,
                  batch_size: int,
                  optimizer_gnn: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  model_gpt: torch.nn.Module = None,  
                  optimizer_gpt: torch.optim = None,
                  scheduler_gpt: torch.optim.lr_scheduler = None,
                  device: torch.cuda.device="cpu",
                  mode = "GNN",) -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        gpt (torch.nn.Module): GPT PDE context generator
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = []
    gnn_norms = []
    gpt_norms = []
    for (u_base, u_super, x, variables) in loader:
        # Reset gradients
        optimizer_gnn.zero_grad()
        if model_gpt is not None:
            optimizer_gpt.zero_grad()
        # Randomly choose number of unrollings
        unrolled_graphs = random.choice(unrolling)
        steps = [t for t in range(graph_creator.tw,
                                  graph_creator.t_res - graph_creator.tw - (graph_creator.tw * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
        random_steps = random.choices(steps, k=batch_size)
        data, labels = graph_creator.create_data(u_super, random_steps)

        # Create first graph to propagate at t
        if mode == 'GNN':
            graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)
        else:
            data, labels = data.to(device), labels.to(device)
        
        #Create first data to generate embeddings at t
        if model_gpt is not None:
            data_gpt = generate_gpt_data(u_super, random_steps).to(device)

        # Unrolling of the equation which serves as input at the current step
        # This is the pushforward trick!!!
        with torch.no_grad():
            for _ in range(unrolled_graphs):

                # Forward pass of GPT model to make an embedding at t. Adds embedding to graph
                if model_gpt is not None:
                    embeddings = model_gpt(data_gpt)
                    graph = graph_creator.add_embeddings(graph, embeddings, random_steps).to(device)

                # Increments steps by tw and creates ground truth for t+1
                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u_super, random_steps)

                if mode == 'GNN':
                    # Makes a prediction for state at t+1 and creates new graph with predicted state at t+1 as input and true state at t+2 as label
                    pred = model_gnn(graph)
                    graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)

                    # Updates data for GPT model at t+1
                    pred_tensor = graph2tensor(pred, batch_size, graph_creator.x_res)
                    data_gpt = update_gpt_data(data_gpt, pred_tensor, random_steps, graph_creator.tw).to(device)
                else:
                    data = model_gnn(data)
                    labels = labels.to(device)

        # Forward pass of GPT model to make an embedding at t+1. Adds embedding to graph
        if model_gpt is not None:
            embeddings = model_gpt(data_gpt)
            graph = graph_creator.add_embeddings(graph, embeddings, random_steps).to(device)

        # Makes a prediction for state at t+2 and computes loss with true state at t+2
        if mode == 'GNN':
            pred = model_gnn(graph)
            loss = criterion(pred, graph.y)
        else:
            pred = model_gnn(data)
            loss = criterion(pred, labels)

        # Backpropagation and stepping the GNN optimizer
        loss = torch.sqrt(loss)
        loss.backward()
        losses.append(loss.detach() / batch_size)
        optimizer_gnn.step()

        grads_gnn = [
            param.grad.detach().flatten()
            for param in model_gnn.parameters()
            if param.grad is not None
        ]
        norm_gnn = torch.cat(grads_gnn).norm()
        gnn_norms.append(norm_gnn / batch_size)

        # Stepping the GPT optimizer and scheduler
        if model_gpt is not None:
            optimizer_gpt.step()
            scheduler_gpt.step()
            grads_gpt = [
                param.grad.detach().flatten()
                for param in model_gpt.parameters()
                if param.grad is not None
            ]
            norm_gpt = torch.cat(grads_gpt).norm()
            gpt_norms.append(norm_gpt / batch_size)

    losses = torch.stack(losses)
    gnn_norms = torch.stack(gnn_norms)
    gpt_norms = torch.stack(gpt_norms)
    return losses, gnn_norms, gpt_norms, scheduler_gpt.get_last_lr()[0]

def test_timestep_losses(model_gnn: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         model_gpt: torch.nn.Module = None,
                         device: torch.cuda.device = "cpu",
                         mode = "GNN",
                         verbose = False) -> None:
    """
    Loss for one neural network forward pass at certain timepoints on the validation/test datasets
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        batch_size (int): batch size
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        None
    """
    # Set models to eval mode
    model_gnn.eval()
    if model_gpt is not None:
        model_gpt.eval()

    # Loop over steps [0, tw, 2*tw, ...]
    for step in steps:

        # Condition to skip steps that are not spaced by tw
        if (step != graph_creator.tw and step % graph_creator.tw != 0):
            continue

        losses = []
        # Loop over every data sample for a given time window
        for (u_base, u_super, x, variables) in loader:
            with torch.no_grad():
                # Create data and labels for current time window in shape [batch_size, tw, x_res]
                same_steps = [step]*batch_size
                data, labels = graph_creator.create_data(u_super, same_steps)

                if mode == 'GNN':
                    # Creates graph at current time window
                    graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)

                    # Create data to input to GPT model. Zeroes out data in the future, although not necessary due to causal attention
                    # Generates embedding and adds it to graph
                    if(model_gpt is not None):
                        data_gpt = torch.zeros_like(u_super).to(device)
                        data_gpt[:, :same_steps[0], :] = u_super[:, :same_steps[0], :]
                        embeddings = model_gpt(data_gpt)
                        graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)

                    # Makes a prediction for state at next time window and computes loss with true state at next time window
                    pred = model_gnn(graph)
                    loss = criterion(pred, graph.y)
                else:
                    data, labels = data.to(device), labels.to(device)
                    pred = model_gnn(data)
                    loss = criterion(pred, labels)
                losses.append(loss / batch_size)

        losses = torch.stack(losses)
        if verbose:
            print(f'Step {step}, mean loss {torch.mean(losses)}')


def test_unrolled_losses(model_gnn: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         nx_base_resolution: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         model_gpt: torch.nn.Module,
                         device: torch.cuda.device = "cpu",
                         mode = "GNN",
                         verbose = False) -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    losses_base = []
    # Loop over every data sample
    for (u_base, u_super, x, variables) in loader:
        losses_tmp = []
        losses_base_tmp = []
        with torch.no_grad():
            # Start at t=tw*nr_gt_steps. Possibility to backprop at test time if nr_gt_steps>1 (future work)
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size

            # Creates data at t
            data, labels = graph_creator.create_data(u_super, same_steps)

            if mode == 'GNN':
                # Creates graph at t
                graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)

                # Creates initial data for GPT model at t, embeds data, and adds embedding to graph
                if(model_gpt is not None):
                    data_gpt = torch.zeros_like(u_super).to(device)
                    data_gpt[:, :same_steps[0], :] = u_super[:, :same_steps[0], :]
                    embeddings = model_gpt(data_gpt)
                    graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)
                
                # Makes prediction for state at t+1 and computes loss with true state at t+1
                pred = model_gnn(graph)
                loss = criterion(pred, graph.y) / nx_base_resolution
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model_gnn(data)
                loss = criterion(pred, labels) / nx_base_resolution

            losses_tmp.append(loss / batch_size)

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
                # Increments same step to t+1
                same_steps = [step] * batch_size

                # Creates true state at t+2 as a label
                _, labels = graph_creator.create_data(u_super, same_steps)
                if mode == 'GNN':
                    # Creates graph at t+1 with predicted state at t+1 as input and true state at t+2 as label
                    graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)

                    # Updates gpt data by adding prediciont at t+1, embeds data, and adds to graph
                    if(model_gpt is not None):
                        pred_tensor = graph2tensor(pred, batch_size, nx_base_resolution)
                        data_gpt = update_gpt_data(data_gpt, pred_tensor, same_steps, graph_creator.tw).to(device)
                        embeddings = model_gpt(data_gpt)
                        graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)

                    # Makes a prediction for state at t+2 with embedding and prediction at t+1, computes loss with true state at t+2
                    pred = model_gnn(graph)
                    loss = criterion(pred, graph.y) / nx_base_resolution
                else:
                    labels = labels.to(device)
                    pred = model_gnn(pred)
                    loss = criterion(pred, labels) / nx_base_resolution
                losses_tmp.append(loss / batch_size)

            # Losses for numerical baseline
            for step in range(graph_creator.tw * nr_gt_steps, graph_creator.t_res - graph_creator.tw + 1,
                                graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels_super = graph_creator.create_data(u_super, same_steps)
                _, labels_base = graph_creator.create_data(u_base, same_steps)
                loss_base = criterion(labels_super, labels_base) / nx_base_resolution
                losses_base_tmp.append(loss_base / batch_size)

        losses.append(torch.sum(torch.stack(losses_tmp)))
        losses_base.append(torch.sum(torch.stack(losses_base_tmp)))

    losses = torch.stack(losses)
    losses_base = torch.stack(losses_base)
    if verbose:
        print(f'Unrolled forward losses {torch.mean(losses)}')
        print(f'Unrolled forward base losses {torch.mean(losses_base)}')

    return losses
