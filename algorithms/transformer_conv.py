""""""
"""
   Copyright (c) 2021 Olivier Sprangers as part of Airlab Amsterdam

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Transformer with Causal Convolutional attention
 Source code: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
              https://nlp.seas.harvard.edu/2018/04/03/attention.html
 Paper: 
   Transformer: https://arxiv.org/abs/1706.03762
   Transformer with Causal Conv: http://arxiv.org/abs/1907.00235

 Changes: 
  No original code from authors available, so implemented from paper. Unclear how the LogSparse attention
  is included but as full convolutional attention provides better accuracy that is not a problem in the experiments 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
#%% Transformer with Causal Convolutional attention
# Source code: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
#              https://nlp.seas.harvard.edu/2018/04/03/attention.html
# Paper: 
#   Transformer: https://arxiv.org/abs/1706.03762
#   Transformer with Causal Conv: http://arxiv.org/abs/1907.00235
#
# Changes: 
#  No original code from authors available, so implemented from paper. Unclear how the LogSparse attention
#  is included but as full convolutional attention provides better accuracy that is not a problem in the experiments      

class FeedForward(nn.Module):
    def __init__(self, d_hidden, d_ff, p_dropout):
        super(FeedForward, self).__init__()    
        # Initialize layers
        self.lin1 = nn.Linear(d_hidden, d_ff)
        self.lin2 = nn.Linear(d_ff, d_hidden)
        self.drop = nn.Dropout(p_dropout)
                
    def forward(self, x):
        # Input dim: [d_seqlen, d_batch, d_hidden]
        # Output dim: [d_seqlen, d_batch, d_hidden]        
        output = F.relu(self.lin1(x))
        output = self.drop(self.lin2(output))
        
        return output   

class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CausalConv, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        
    def forward(self, x):
        return self.conv(x.permute(1, 2, 0))[:, :, :-self.padding].permute(2, 0, 1)

class ConvolutionalMultiheadAttention(nn.Module):
    def __init__(self, d_hidden, h, dropout, kernel_size):
        super(ConvolutionalMultiheadAttention, self).__init__()
        # Initialize variables
        self.h = h
        self.d_hidden = d_hidden
        self.sqrt_d_h = np.sqrt(self.d_hidden)
        # Initialize linear layers for Q, V and K
        self.Qlin = CausalConv(d_hidden, d_hidden * h, kernel_size, padding=(kernel_size - 1))
        self.Klin = CausalConv(d_hidden, d_hidden * h, kernel_size, padding=(kernel_size - 1))
        self.Vlin = nn.Linear(d_hidden, d_hidden * h)
        # Initialize output layer, dropout and normalization layer
        self.linout = nn.Linear(d_hidden * h, d_hidden)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, attn_mask=None):
        d_batch = Q.shape[1]       
        
        # Apply linear layer
        Q, K, V = self.Qlin(Q), self.Klin(K), self.Vlin(V)
        # Split into h heads
        Q = Q.view(-1, d_batch, self.h, self.d_hidden).permute(1, 2, 0, 3)
        K = K.view(-1, d_batch, self.h, self.d_hidden).permute(1, 2, 0, 3)
        V = V.view(-1, d_batch, self.h, self.d_hidden).permute(1, 2, 0, 3)
        # Calculate Scaled Dot Product attention per sample and per head       
        output = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.sqrt_d_h        
        # Apply mask to avoid looking forward (if required)
        if attn_mask is not None:
            output += attn_mask
        attn_scores = F.softmax(output, dim = -1)       
        attn_scores = self.drop(attn_scores)
        # Calculate output
        output = torch.matmul(attn_scores, V)
        # Reverse dimensions; add contiguous because PyTorch otherwise complains...       
        output = output.permute(2, 0, 1, 3).contiguous()
        # Compress last two dimensions, i.e. effectively concatenating the heads
        output = output.view(-1, d_batch, self.h * self.d_hidden)        
        # Calculate linear output
        output = self.linout(output)
        
        return output, attn_scores

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
  
class DecoderLayer(nn.Module):
    # Class constructor
    def __init__(self, d_hidden, p_dropout, h, d_ff, kernel_size):
        super(DecoderLayer, self).__init__()
        # Initialize multi-head attention layers
        self.mha = ConvolutionalMultiheadAttention(d_hidden, h, p_dropout, kernel_size)
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        # Feedforward layer
        self.ff = FeedForward(d_hidden, d_ff, p_dropout)
        
    def forward(self, x, mask):
        # MHA
        h1 = self.norm1(x)
        h1, _ = self.mha(h1, h1, h1, attn_mask = mask)
        h1 = h1 + x
        
        # FF
        h2 = self.norm2(h1)
        h2 = self.ff(h2)
        h2 = h2 + h1

        return h2   

class Decoder(nn.Module):
    # Class constructor
    def __init__(self, d_hidden, N, h, d_ff, p_dropout, kernel_size):
        super(Decoder, self).__init__()    
        self.N = N  
        self.dec_layers = get_clones(DecoderLayer(d_hidden, p_dropout, h, d_ff, kernel_size), N)
    
    def forward(self, x, mask):
        h = x
        for i in range(self.N):
            h = self.dec_layers[i](h, mask)
               
        return h

class transformer_conv(nn.Module):
    # Class constructor
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, kernel_size, dim_maxseqlen):
        super(transformer_conv, self).__init__()
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        # Embedding layer for positions
        self.emb_pos = nn.Embedding(dim_maxseqlen, d_emb_tot)
        emb_pos_idx = torch.arange(dim_maxseqlen)
        self.register_buffer('emb_pos_idx', emb_pos_idx) 
        # Initialize Decoder
        d_ff = 4 * d_hidden
        h = 8
        self.decoder = Decoder(d_lag + d_cov + d_emb_tot, N, h, d_ff, dropout, kernel_size)
        mask = torch.triu(torch.ones((dim_maxseqlen, dim_maxseqlen)), diagonal=1) * -1e9
        self.register_buffer('mask', mask)
        # Output layer
        self.loc = nn.Linear(d_lag + d_cov + d_emb_tot, d_output)
        self.scale = nn.Linear(d_lag + d_cov + d_emb_tot, d_output)

    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):             
        # Time series ID and position embeddings
        emb_pos = self.emb_pos(self.emb_pos_idx[:x_idx.shape[0]]).unsqueeze(1)
        emb_id = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            emb_id.append(out)
        emb_id = torch.cat(emb_id, -1)
        x_emb = emb_id + emb_pos
        # Concatenate inputs
        dim_seq = x_lag.shape[0]
        h = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        # Decoder
        h = self.decoder(h, self.mask[:h.shape[0], :h.shape[0]])        
        # Output layer - location & scale of distribution        
        loc = F.softplus(self.loc(h[-d_outputseqlen:]))
        scale = F.softplus(self.scale(h[-d_outputseqlen:])) + 1e-6
    
        return loc, scale