import math
from random import random

import torch
from torch import nn


"Masking functions from: https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py"

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_seq(seq, mask_prob=0.15, replace_prob=0.9, random_token_prob=0.1):

    mask_token_id = 0

    mask = prob_mask_like(seq, mask_prob)
    mask_out = mask.clone().detach()

    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)

    masked_seq = seq.clone().detach()

    idx = torch.randperm(seq.numel())
    random_tokens = torch.flatten(seq)[idx].view(seq.size())

    random_token_prob = prob_mask_like(seq, random_token_prob) & mask
    masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

    # remove tokens that were substituted randomly from being [mask]ed later
    mask = mask & ~random_token_prob

    # [mask] input

    replace_prob = prob_mask_like(seq, replace_prob)
    masked_seq = masked_seq.masked_fill(mask * replace_prob, mask_token_id)

    return masked_seq, mask_out

class MaskedWrapper(nn.Module):
    '''
    Wrapper around a nn.Module that:
        - masks the input
        - computes a forward pass with the model
        - calculates a MSE loss 
    '''
    

    def __init__(
        self,
        net,
        reconstruction_net,
        variable_net = None,
        gamma = 0.1, # Weight of variable loss
        mask_prob = 0.15, # Probability to mask out a token
        replace_prob = 0.9, # Probability to replace a masked token with 0
        random_token_prob = 0.1 # Probability to replace a masked token with a random token
    ):
        super().__init__()
        self.net = net
        self.reconstruction_net = reconstruction_net
        self.variable_net = variable_net
        self.gamma = gamma
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        self.mse = nn.MSELoss()

    def remove_sep(self, x):
        '''
        Remove [SEP] tokens from the input
        '''
        num_segments = self.net.num_segments
        segment_len = self.net.segment_len

        # x is in shape [batch_size, seq_len+num_segments, d_model]
        segments = []
        for i in range(num_segments):
            start = i*(segment_len+1)
            end = (i+1)*(segment_len+1)
            segments.append(x[:, start:end-1, :]) # Remove last element in segment

        out = torch.cat(segments, dim=1)
        return out

    def forward(
        self,
        x,
        mode = 'train',
        variables = None
    ):
        '''
        Inputs:
            x: input sequence in shape [batch_size, num_tokens, d_in]
            mode: train or eval
            variables: PDE variables in shape [batch_size, num_vars]
        '''

        orig_seq = x.clone()

        # Mask input
        x_masked, mask = mask_seq(x, self.mask_prob, self.replace_prob, self.random_token_prob)

        # Forward pass
        x_pred = self.net(x_masked)
        cls_token = torch.clone(x_pred[:, 0, :])
        x_pred = x_pred[:, 1:, :] # remove cls token
        x_pred = self.remove_sep(x_pred) # remove sep tokens
        x_pred = self.reconstruction_net(x_pred)

        if mode == "train":
            if variables is None:
                # Reconstruction Loss
                loss = self.mse(x_pred[mask], orig_seq[mask])
                return loss
            else:
                # Reconstruction Loss
                reconstruction_loss = self.mse(x_pred[mask], orig_seq[mask])

                # Variable Loss
                variables_pred = self.variable_net(cls_token)
                sysID_loss = self.gamma*self.mse(variables, variables_pred)

                loss = reconstruction_loss + sysID_loss

                return loss, reconstruction_loss, sysID_loss
        
        else:
            return x_pred, orig_seq, x_masked