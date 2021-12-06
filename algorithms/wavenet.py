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

WaveNet 
 Paper: https://arxiv.org/pdf/1609.03499.pdf

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# This implementation of causal conv is faster than using normal conv1d module
class CustomConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(CustomConv1d, self).__init__()
        k = np.sqrt(1 / (in_channels * kernel_size))
        weight_data = -k + 2 * k * torch.rand((out_channels, in_channels, kernel_size))
        bias_data = -k + 2 * k * torch.rand((out_channels))
        self.weight = nn.Parameter(weight_data, requires_grad=True)
        self.bias = nn.Parameter(bias_data, requires_grad=True)
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        xp = F.pad(x, (self.padding, 0))
        return F.conv1d(xp, self.weight, self.bias, dilation=self.dilation)

class wavenet_cell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(wavenet_cell, self).__init__()
        self.conv_dil = CustomConv1d(in_channels, out_channels * 2, kernel_size, padding, dilation)
        self.conv_skipres = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, x):
        h_prev, skip_prev = x
        f, g = self.conv_dil(h_prev).chunk(2, 1)
        h_next, skip_next = self.conv_skipres(torch.tanh(f) * torch.sigmoid(g)).chunk(2, 1)
        
        return (h_prev + h_next, skip_prev + skip_next)

class wavenet(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, Nl, kernel_size):
        super(wavenet, self).__init__()
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        self.upscale = nn.Linear(d_lag + d_cov + d_emb_tot, d_hidden)
        # Wavenet
        wnet_layers = nn.ModuleList([wavenet_cell(
                    d_hidden, d_hidden, 
                    kernel_size, padding=(kernel_size-1) * 2**i, 
                    dilation = 2**i) for i in range(Nl)])  
        self.wnet = nn.Sequential(*wnet_layers)
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
        h = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)        
        h = self.upscale(h)
        # Apply wavenet
        _, h = self.wnet((h.permute(1, 2, 0), 0))
        # Output layers - location & scale of the distribution
        output = h[:, :, -d_outputseqlen:].permute(2, 0, 1)
        loc, scale = F.softplus(self.loc_scale(output)).chunk(2, -1)
        return loc, scale + self.epsilon