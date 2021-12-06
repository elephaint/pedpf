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

TCN 
 Source code: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
 Paper: https://arxiv.org/abs/1803.01271
 Changes: default conv initialization of Pytorch used.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dropout):
        super(TBlock, self).__init__()
        self.padding = padding
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Create residual
        res = x if self.downsample is None else self.downsample(x)
        # Two-layer TCN
        h = self.drop1(F.relu(self.conv1(x)[:, :, :-self.padding]))
        h = self.drop2(F.relu(self.conv2(h)[:, :, :-self.padding]))
        return F.relu(h + res)

class tcn(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, kernel_size):
        super(tcn, self).__init__()
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        # TCN       
        layers = nn.ModuleList([TBlock(
                    d_lag+d_cov+d_emb_tot if i == 0 else d_hidden, d_hidden, 
                    kernel_size, padding=(kernel_size-1) * 2**i, 
                    dilation = 2**i, dropout=dropout) for i in range(N)])  
        self.tcn = nn.Sequential(*layers)    
        # Output layer
        self.loc = nn.Linear(d_hidden, d_output)
        self.scale = nn.Linear(d_hidden, d_output)
        
    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):       
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            x_emb.append(out)
        x_emb = torch.cat(x_emb, -1)
        # Concatenate inputs
        dim_seq = x_lag.shape[0]
        h = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        # Apply TCN
        h = self.tcn(h.permute(1, 2, 0)).permute(2, 0, 1)
        # Output layers - location & scale of the distribution
        loc = self.loc(h[-d_outputseqlen:])
        scale = F.softplus(self.scale(h[-d_outputseqlen:])) + 1e-6
        return loc, scale