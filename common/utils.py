import os
import h5py
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph
from equations.PDEs import *
from collections import OrderedDict
import re 
import matplotlib.pyplot as plt


class HDF5Dataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(self,
                 path: str,
                 pde: PDE,
                 mode: str,
                 base_resolution: list=None,
                 super_resolution: list=None,
                 load_all: bool=False) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: string of PDE ('CE' or 'WE')
            mode: [train, valid, test]
            base_resolution: base resolution of the dataset [nt, nx]
            super_resolution: super resolution of the dataset [nt, nx]
            load_all: load all the data into memory
        Returns:
            None
        """
        super().__init__()
        f = h5py.File(path, 'r')
        self.mode = mode
        self.pde = pde
        self.dtype = torch.float64
        self.data = f[self.mode]
        self.base_resolution = (250, 100) if base_resolution is None else base_resolution
        self.super_resolution = (250, 200) if super_resolution is None else super_resolution
        self.dataset_base = f'pde_{self.base_resolution[0]}-{self.base_resolution[1]}'
        self.dataset_super = f'pde_{self.super_resolution[0]}-{self.super_resolution[1]}'

        ratio_nt = self.data[self.dataset_super].shape[1] / self.data[self.dataset_base].shape[1]
        ratio_nx = self.data[self.dataset_super].shape[2] / self.data[self.dataset_base].shape[2]
        assert (ratio_nt.is_integer())
        assert (ratio_nx.is_integer())
        self.ratio_nt = int(ratio_nt)
        self.ratio_nx = int(ratio_nx)

        self.nt = self.data[self.dataset_base].attrs['nt']
        self.dt = self.data[self.dataset_base].attrs['dt']
        self.dx = self.data[self.dataset_base].attrs['dx']
        self.x = self.data[self.dataset_base].attrs['x']
        self.tmin = self.data[self.dataset_base].attrs['tmin']
        self.tmax = self.data[self.dataset_base].attrs['tmax']

        if load_all:
            data = {self.dataset_super: self.data[self.dataset_super][:]}
            f.close()
            self.data = data


    def __len__(self):
        return self.data[self.dataset_super].shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: numerical baseline trajectory
            torch.Tensor: downprojected high-resolution trajectory (used for training)
            torch.Tensor: spatial coordinates
            list: equation specific parameters
        """
        if(f'{self.pde}' == 'CE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            left = u_super[..., -3:-1]
            right = u_super[..., 1:3]
            u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
            weights = torch.tensor([[[[0.2]*5]]])
            u_super = F.conv2d(u_super_padded, weights, stride=(1, self.ratio_nx)).squeeze().numpy()
            x = self.x

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['alpha'] = self.data['alpha'][idx]
            variables['beta'] = self.data['beta'][idx]
            variables['gamma'] = self.data['gamma'][idx]

            return u_base, u_super, x, variables

        elif(f'{self.pde}' == 'WE'):
            # Super resolution trajectories are downprojected via kernel which averages of neighboring cell values
            # No padding is possible due to non-periodic boundary conditions
            weights = torch.tensor([[[[1./self.ratio_nx]*self.ratio_nx]]])
            u_super = self.data[self.dataset_super][idx][::self.ratio_nt][None, None, ...]
            u_super = F.conv2d(torch.tensor(u_super), weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # To match the downprojected trajectories, also coordinates need to be downprojected
            x_super = torch.tensor(self.data[self.dataset_super].attrs['x'][None, None, None, :])
            x = F.conv2d(x_super, weights, stride=(1, self.ratio_nx)).squeeze().numpy()

            # Base resolution trajectories (numerical baseline) and equation specific parameters
            u_base = self.data[self.dataset_base][idx]
            variables = {}
            variables['bc_left'] = self.data['bc_left'][idx]
            variables['bc_right'] = self.data['bc_right'][idx]
            variables['c'] = self.data['c'][idx]

            return u_base, u_super, x, variables

        else:
            raise Exception("Wrong experiment")


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE,
                 neighbors: int = 2,
                 time_window: int = 5,
                 t_resolution: int = 250,
                 x_resolution: int =100,
                 use_gpt = True,
                 ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution
        self.use_gpt = use_gpt

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d[None, :]), 0)
            labels = torch.cat((labels, l[None, :]), 0)

        return data, labels


    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list,) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor of shape (batch_size, time_window, nx)
            embeddings (torch.Tensor): embeddings from GPT model (batch_size, nt, embedding_size)
            labels (torch.Tensor): label tensor of shape (batch_size, time_window, nx)
            x (torch.Tensor): spatial coordinates tensor of shape (batch_size, nx)
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            # data_batch is of shape (time_window, nx)
            # labels_batch is of shape (time_window, nx)

            # constructs list of shape [time_window, (nx,1)] and concatenates along time axis to make tensor of shape [time_window, nx]
            # transposes to make tensor of shape [nx, time_window]
            # concatenates along batch axis to make tensor of shape [batch_size*nx, time_window]
            u = torch.cat((u, torch.transpose(torch.cat([d[None, :] for d in data_batch]), 0, 1)), )  
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            x_pos = torch.cat((x_pos, x[0]), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
            batch = torch.cat((batch, torch.ones(nx) * b), )

        # Calculate the edge_index
        # Note, investigate knn or constructing edge indices based on adaptive proximity (only need to do once per problem)

        if f'{self.pde}' == 'CE':
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        elif f'{self.pde}' == 'WE':
            edge_index = knn_graph(x_pos, k=self.n, batch=batch.long(), loop=False) 

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        graph.pos = torch.cat((t_pos[:, None], x_pos[:, None]), 1)
        graph.batch = batch.long()

        # Equation specific parameters
        if f'{self.pde}' == 'CE':
            alpha, beta, gamma = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                alpha = torch.cat((alpha, torch.tensor([variables['alpha'][i]])[:, None]), )
                beta = torch.cat((beta, torch.tensor([variables['beta'][i]*(-1.)])[:, None]), )
                gamma = torch.cat((gamma, torch.tensor([variables['gamma'][i]])[:, None]), )

            graph.alpha = alpha
            graph.beta = beta
            graph.gamma = gamma

        elif f'{self.pde}' == 'WE':
            bc_left, bc_right, c = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i in batch.long():
                bc_left = torch.cat((bc_left, torch.tensor([variables['bc_left'][i]])[:, None]), )
                bc_right = torch.cat((bc_right, torch.tensor([variables['bc_right'][i]])[:, None]), )
                c = torch.cat((c, torch.tensor([variables['c'][i]])[:, None]), )

            graph.bc_left = bc_left
            graph.bc_right = bc_right
            graph.c = c

        return graph

    def add_embeddings(self,                             
                            graph: Data,
                             embeddings: torch.Tensor,
                             steps: list) -> Data:
        """
        Adds embeddings to current graph
        Args:
            graph (Data): Pytorch geometric data object
            steps (list): list of different starting points for each batch entry
            embeddings (torch.Tensor): embeddings from GPT model (batch_size, nt, embedding_size)
        Returns:
            Data: Pytorch Geometric data graph
        """
        e = torch.Tensor()
        for e_batch, step in zip(embeddings, steps):
            e_batch = e_batch[step-self.tw:step].cpu()  # selects relevant embeddings
            e = torch.cat((e, torch.transpose(torch.cat([e_b[None, :] for e_b in e_batch]), 0, 1)), )
        graph.e = e
        return graph

    def create_next_graph(self,
                             graph: Data,
                             pred: torch.Tensor,
                             labels: torch.Tensor,
                             steps: list) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]
        nt = self.pde.grid_size[0]
        nx = self.pde.grid_size[1]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor(),
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat((y, torch.transpose(torch.cat([l[None, :] for l in labels_batch]), 0, 1)), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y
        graph.pos[:, 0] = t_pos

        return graph
class Curriculum:
    '''
    Class to get probability of using autoregressive input vs. teacher forcing
    '''

    def __init__(self, total_epochs, delta):
        self.total_epochs = total_epochs
        self.delta = delta

        epoch = torch.arange(total_epochs)
        alpha = (epoch/total_epochs - 0.5)/delta

        k = total_epochs/2 * torch.tanh(alpha)
        k = k - torch.min(k)
        k = k/torch.max(k)

        self.probs = k

    def get_prob(self, epoch):
        return self.probs[epoch]

def process_dict(state_dict: OrderedDict) -> OrderedDict:
    '''
    Processes state dict to remove the 'module.' prefix'''

    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    
    return model_dict

def plot_traj(traj, title):
    '''
    Plots PDE trajectory, with lighter colors for later time steps
    Traj is expected in shape: [nt, nx]
    '''
    N = traj.shape[0]
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))

    fig, ax = plt.subplots()
    x = torch.linspace(0, 2, traj.shape[1])
    for i in range(N):
        ax.plot(x, traj[i])
    plt.title(title)
    plt.show()


def unroll_traj(u_super, x, variables, model_gnn, model_gpt, graph_creator, batch_size, device, nr_gt_steps = 1):
    '''
    Unrolls a trajectory given by u_super, x, and variables
    u_super is expected in [1, t_res, nx]
    '''

    with torch.no_grad():
        same_steps = [graph_creator.tw*nr_gt_steps] * batch_size
        data, labels = graph_creator.create_data(u_super, same_steps)
        nx_base_resolution = u_super.shape[2]
        
        if(model_gpt is not None):
            data_gpt = torch.zeros_like(u_super).to(device)
            data_gpt[:,:same_steps[0],:] = u_super[:,:same_steps[0],:]
            embeddings = model_gpt(data_gpt)

        graph = graph_creator.create_graph(data, labels, x, variables, same_steps).to(device)
        if(model_gpt is not None):
            graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)
        traj = graph.x
        pred = model_gnn(graph)
        traj = torch.cat((traj, pred), dim=1)

        # Unroll trajectory and add losses which are obtained for each unrolling
        for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_res - graph_creator.tw + 1, graph_creator.tw):
            same_steps = [step] * batch_size
            _, labels = graph_creator.create_data(u_super, same_steps)
            graph = graph_creator.create_next_graph(graph, pred, labels, same_steps).to(device)
            if model_gpt is not None:
                data_gpt[:, step-graph_creator.tw:step, :] = graph2tensor(graph.x, batch_size, nx_base_resolution)
                embeddings = model_gpt(data_gpt)
                graph = graph_creator.add_embeddings(graph, embeddings, same_steps).to(device)
            pred = model_gnn(graph)
            traj = torch.cat((traj, pred), dim=1)

    traj = torch.transpose(traj, 0,1)
    traj_true = u_super

    return traj, traj_true

def graph2tensor(input, batch_size, nx_base_resolution):
    '''
    Converts representation of graph from [batch_size*nx, time_window] to [batch_size, time_window, nx]
    '''

    output = torch.zeros((batch_size, input.shape[1], nx_base_resolution)).to(input.device)
    batch = 0
    for i in range(0, input.shape[0], nx_base_resolution):
        output[batch, :, :] = torch.transpose(input[i:i+nx_base_resolution, :], 0,1)
        batch += 1
    return output

def generate_gpt_data(u_super, cache, steps, curriculum_prob, tw):
    '''
    Generates data for GPT model
    data is expected in shape [batch_size, t_res, nx]
    '''
    data_gpt = torch.zeros_like(u_super)

    # Context length subtracts the initial sequence given
    context_len = steps[0] - tw

    # k is the number of autoregressively generated timesteps to add
    k = torch.floor(curriculum_prob * context_len)
    
    # Seed data_gpt with the initial sequence
    data_gpt[:, :steps[0], :] = u_super[:, :steps[0], :]

    # Mix context in data_gpt with autoregressively generated data according to curriculum
    start = int(steps[0] - k)
    data_gpt[:, start:steps[0], :] = cache[:, start:steps[0], :]

    return data_gpt

def update_gpt_data(data_gpt, update_data, steps, tw):
    '''
    Appends new data to data_gpt
    update_data is a model output in shape [batch_size, time_window, nx]
    data_gpt is expected in shape [batch_size, t_res, nx]
    '''
    for i in range(len(steps)):
        data_gpt[i, steps[i]-tw:steps[i], :] = update_data[i]
    return data_gpt