import math
import torch
from torch import nn, Tensor
from x_transformers import Encoder, ContinuousTransformerWrapper

class BERT(nn.Module):

    def __init__(self, 
                 d_in: int, 
                 d_model: int, 
                 nhead: int, 
                 num_layers: int, 
                 segment_len: int = 100,
                 num_segments: int = 2,
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
            max_seq_len = num_segments*segment_len + num_segments + 1, # +1 for cls token
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
        self.cls_token = nn.Parameter(torch.randn(1, d_model))
        self.sep_token = nn.Parameter(torch.ones(1, d_model))
        
        self.segment_len = segment_len
        self.num_segments = num_segments
        self.segment_ids = self.generate_segment_ids()
        self.segment_id_embedding = nn.Embedding(num_embeddings=num_segments, embedding_dim=d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()

    def generate_segment_ids(self, batch_size: int = 64) -> Tensor:
        # Builds segment ids once 
        segment_ids = []
        for i in range(self.num_segments):
                if i == 0:
                    segment_ids.append(torch.full((batch_size, self.segment_len+2), i, dtype=torch.long))
                else:
                    segment_ids.append(torch.full((batch_size, self.segment_len+1), i, dtype=torch.long))
        segment_ids = torch.cat(segment_ids, dim=1)
        return segment_ids

    def token_embedding(self, src: Tensor) -> Tensor:
        '''
        Embeds input data w/ and MLP and scale by model dimension
        Adds CLS and SEP tokens
        '''

        batch_size, seq_len, d_in = src.shape

        assert seq_len <= self.num_segments*self.segment_len, f'Sequence length {seq_len} is longer than maximum sequence length {2*self.segment_len}'

        # Embeds input data w/ and MLP and scale by model dimension
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Copies cls token across batch dimension and prepends to sequence
        src = torch.column_stack([self.cls_token.expand(batch_size, -1, -1), src])

        # Adds sep token to end of each segment
        segments = []
        for i in range(self.num_segments):

            # Start at 0 -> segment_len + 1 (for cls), then segment_len + 1 -> 2*self.segment_len + 1, etc.
            if i == 0:
                start = 0
            else:
                start = i*self.segment_len+1
            end = (i+1)*self.segment_len+1 #hello anthony beeeee
            temp = torch.column_stack([src[:, start:end, :], self.sep_token.expand(batch_size, -1, -1)])
            segments.append(temp)

        # Concatenates segments into src
        src = torch.cat(segments, dim=1)

        return src
    
    def segment_embedding(self, src: Tensor) -> Tensor:
        '''
        Embeds segment tokens and adds sep tokens
        '''
        batch_size = src.shape[0]
        # Embeds segment ids
        segment_embedding = self.segment_id_embedding(self.segment_ids[batch_size].to(src.device))
        # Adds segment embedding to input
        src = src + segment_embedding

        return src

    
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_in]``

        Returns:
            output: Tensor of shape ``[batch_size, seq_len, d_out]``
        """
        # Token embedding projects input to model dimension, adds cls token, and adds sep tokens
        src = self.token_embedding(src)
        # Segment embedding adds segment tokens
        src = self.segment_embedding(src)

        # Forward pass through transformer encoder. Automatically adds positional embeddings
        output = self.encoder(src)

        return output