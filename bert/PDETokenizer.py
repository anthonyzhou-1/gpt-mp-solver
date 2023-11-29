import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import numpy as np

class PDETokenizer:
    def __init__(self,
                 num_tokens: int,
                 nx: int,
                 x_len: float = 1.0,
                 mode: str = 'constant',
                 fourier_modes: int = 5):
        self.num_tokens = num_tokens
        self.nx = nx
        self.mode = mode
        if mode != 'constant':
            self.stencil = self.assemble_stencil(nx, x_len/nx)
    
    def assemble_stencil(self, nx, dx):
        '''
        Assembles stencil for finite difference approximation
        '''
        stencil = torch.zeros((nx, nx))
        stencil[0, 0] = 1
        stencil[0, 1] = -1
        stencil[nx-1, nx-1] = -1
        stencil[nx-1, nx-2] = 1
        for i in range(1, nx-1):
            stencil[i, i-1] = 1/2
            stencil[i, i+1] = -1/2
        stencil = stencil/dx
        return stencil

    def generate_tokens(self, sequence):
        '''
        Generates tokens from sequence based on Structured Hierarchical Clustering
        '''
        batch_size, num_segments, nx = sequence.shape

        # Transform stencil to size (batch_size, nx, nx)
        stencil_batch = self.stencil.expand(batch_size, -1, -1)

        # Batch matmul (batch, nx, nx) * (batch, nx, num_segments) -> (batch, nx, num_segments)
        derivatives = torch.bmm(stencil_batch, torch.transpose(sequence, 1, 2))
        # Transform back to original shape
        derivatives = torch.transpose(derivatives, 1, 2)

    def forward(self, sequence):
        '''
        Tokenizes sequence based on mode
        Sequence is given in shape (batch_size, num_segments, nx)
        Returns sequence in shape (batch_size, num_tokens, d_in)
        '''
        batch_size, num_segments, nx = sequence.shape
        d_in = int(num_segments*nx/self.num_tokens)
        if self.mode == "constant":
            assert (num_segments*nx)%self.num_tokens == 0, f'Number of tokens {self.num_tokens} must be a multiple of the number of segments {num_segments} times the number of grid points {nx}'
            return sequence.view(batch_size, self.num_tokens, d_in)
        elif self.mode == "bicubic":
            pass
        elif self.mode == "fourier":
            pass
        else:
            assert False, f'Mode {self.mode} not recognized'