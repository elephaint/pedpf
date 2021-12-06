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

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn import Parameter
import numpy as np
#%% BiTCN 

# This implementation of causal conv is faster than using normal conv1d module
class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1, mode='backward', groups=1):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels // groups, kernel_size))
        bias_data = -k + 2 * k * torch.rand((out_channels))
        self.weight = Parameter(weight_data, requires_grad=True)
        self.bias = Parameter(bias_data, requires_grad=True)  
        self.dilation = dilation
        self.groups = groups
        if mode == 'backward':
            self.padding_left = padding
            self.padding_right= 0
        elif mode == 'forward':
            self.padding_left = 0
            self.padding_right= padding            

    def forward(self, x):
        xp = F.pad(x, (self.padding_left, self.padding_right))
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation, groups=self.groups)

class tcn_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, mode, groups, dropout):
        super(tcn_cell, self).__init__()
        self.conv1 = weight_norm(CustomConv1d(in_channels, out_channels, kernel_size, padding, dilation, mode, groups))
        self.conv2 = weight_norm(CustomConv1d(out_channels, in_channels * 2, 1))
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        h_prev, out_prev = x
        h = self.drop(F.gelu(self.conv1(h_prev)))
        h_next, out_next = self.conv2(h).chunk(2, 1)
        return (h_prev + h_next, out_prev + out_next)

class bitcn_noforward(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, Nl, kernel_size):
        super(bitcn_noforward, self).__init__()
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        self.upscale_lag = nn.Linear(d_lag + d_emb_tot + d_cov, d_hidden)
        self.drop_lag = nn.Dropout(dropout)
        # tcn
        layers_bwd = nn.ModuleList([tcn_cell(
                    d_hidden, d_hidden * 4, 
                    kernel_size, padding=(kernel_size-1)*2**i, 
                    dilation=2**i, mode='backward', 
                    groups=1, 
                    dropout=dropout) for i in range(Nl)])
        self.net_bwd = nn.Sequential(*layers_bwd)
        # Output layer
        self.loc_scale = nn.Linear(d_hidden, d_output * 2)
        self.epsilon = 1e-6
        
    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):       
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            x_emb.append(out)
        x_emb = torch.cat(x_emb, -1)
        # Concatenate inputs
        dim_seq = x_lag.shape[0]
        h_lag = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        h_lag = self.drop_lag(self.upscale_lag(h_lag)).permute(1,2,0)
        # Apply bitcn
        _, out = self.net_bwd((h_lag, 0))
        # Output layers - location & scale of the distribution
        output = out[:, :, -d_outputseqlen:].permute(2, 0, 1)
        loc, scale = F.softplus(self.loc_scale(output)).chunk(2, -1)
        return loc, scale + self.epsilon