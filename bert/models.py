import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from x_transformers import Encoder, ContinuousTransformerWrapper

class BERT(nn.Module):

    def __init__(self, 
                 d_in: int, 
                 d_out: int, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 dropout: float = 0.1):

        '''
        Arguments:
            d_in: input dimension of signal data.
            d_model: model dimension 
            nhead: number of heads in each self-attention layer
            d_hid: hidden dimension in self-attention layer
            nlayers: number of transformer layers
            dropout: probability of dropout
        
        Transformer model that outputs encodings projected down to input dimension for a reconstruction task
        '''

        super().__init__()
        self.model_type = 'BERT'
        self.encoder = ContinuousTransformerWrapper(
            max_seq_len = 512,
            dim_out = d_out,
            use_abs_pos_emb = False,
            attn_layers = Encoder(
                dim = d_model,
                depth = num_layers,
                heads = nhead,
                rotary_pos_emb = True,
                attn_dropout = dropout,
                ff_dropout = dropout,
                attn_flash = True,
                )
            )
        self.embedding = nn.Linear(d_in, d_model)
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.rand(1, d_model))

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()
    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_in]``

        Returns:
            output: Tensor of shape ``[batch_size, seq_len, d_out]``
        """
        batch_size, seq_len, d_in = src.shape

        # Embeds input data w/ and MLP and scale by model dimension
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Copies cls token across batch dimension and prepends to sequence
        src = torch.column_stack([self.cls_token.expand(batch_size, -1, -1), src])

        # Forward pass through transformer encoder
        output = self.encoder(src)

        return output