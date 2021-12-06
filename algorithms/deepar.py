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

DeepAR
 Relevant source code (many thanks): 
       https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/model/deepar/_network.py
       https://github.com/zhykoties/TimeSeries/blob/master/model/net.py
       https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py

 DeepAR has two changes with respect to original DeepAR:
 - Dropout on LSTM layers instead of Zoneout on (h,c) pairs in each cell
 - No residual connections within each cell
 The reason is this allows to use cuDNN LSTMs, which leads to much faster training - in discussion with David these modifications were deemed fine.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
import math
from typing import List
from torch.nn import Parameter

class deepar(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N):
        super(deepar, self).__init__()
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        # Network   
        lstm = [nn.LSTM(d_lag + d_cov + int(d_emb_tot), d_hidden)]
        for i in range(N - 1):
            lstm += [nn.LSTM(d_hidden, d_hidden)]    
        self.lstm = nn.ModuleList(lstm)
        self.drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N)])
        # Output distribution layers
        self.loc = nn.Linear(d_hidden * N, d_output)
        self.scale = nn.Linear(d_hidden * N, d_output)
        self.epsilon = 1e-6
        
    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            x_emb.append(out)
        x_emb = torch.cat(x_emb, -1)
        # Concatenate x_lag, x_cov and time series ID
        dim_seq = x_lag.shape[0]
        inputs = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)
        # DeepAR network       
        h = []
        for i, layer in enumerate(self.lstm):
            outputs, _ = layer(inputs)
            outputs = self.drop[i](outputs)
            inputs = outputs
            h.append(outputs)
        h = torch.cat(h, -1)
        # Output layers - location and scale of distribution
        loc = self.loc(h[-d_outputseqlen:])
        scale = F.softplus(self.scale(h[-d_outputseqlen:]))
        return loc, scale + self.epsilon

## DeepAR following the GluonTS implementation (https://github.com/awslabs/gluon-ts/blob/master/src/gluonts/model/deepar/_network.py) - slow because we need to use JIT instead of cuDNN
## NB: inference is highly inefficient because model does not allow input of current states - this was done to keep structure similar to parallel seq2seq models. 
#class deepar(nn.Module):
#    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, device):
#        super(deepar, self).__init__()
#        # Embedding layer for time series ID
#        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
#        d_emb_tot = d_emb[:, 1].sum()
#        # Network
#        lstm = [LSTMLayer(LSTMCell, d_lag + d_cov + d_emb_tot, d_hidden, dropout, False)]
#        for i in range(N - 1):
#            lstm += [LSTMLayer(LSTMCell, d_hidden, d_hidden, dropout, True)]       
#        self.lstm = nn.ModuleList(lstm)
#        self.h0 = torch.zeros((1, d_hidden)).to(device)
#        self.c0 = torch.zeros((1, d_hidden)).to(device)
#        # Output distribution layers
#        self.loc_scale = nn.Linear(d_hidden * N, d_output * 2)
#        self.epsilon = 1e-6
#     
#    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):
#        # Embedding layers
#        x_emb = []
#        for i, layer in enumerate(self.emb):
#            out = layer(x_idx[:, :, i])
#            x_emb.append(out)
#        x_emb = torch.cat(x_emb, -1)
#        # Concatenate x_lag, x_cov and time series ID
#        dim_seq = x_lag.shape[0]
#        inputs = torch.cat((x_lag, x_cov[:dim_seq], x_emb[:dim_seq]), dim=-1)      
#        # DeepAR network - store total h in list       
#        h = []
#        for layer in self.lstm:
#            outputs = layer(inputs, self.h0, self.c0)
#            inputs = outputs
#            h.append(outputs)
#        h = torch.cat(h, -1)
#        # Output layers - location and scale of distribution
#        loc, scale = F.softplus(self.loc_scale(h[-d_outputseqlen:])).chunk(2, -1)
#        return loc, scale + self.epsilon
#    
## Credit goes to: 
## https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
## https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
#class LSTMCell(jit.ScriptModule):
#    __constants__ = ['residual']
#    
#    def __init__(self, input_size, hidden_size, dropout, residual):
#        super(LSTMCell, self).__init__()
#        k = math.sqrt(1 / hidden_size)
#        self.weight_ih = Parameter(-k + 2 * k * torch.rand(4 * hidden_size, input_size))
#        self.weight_hh = Parameter(-k + 2 * k * torch.rand(4 * hidden_size, hidden_size))
#        self.bias_ih = Parameter(-k + 2 * k * torch.rand(4 * hidden_size))
#        self.bias_hh = Parameter(-k + 2 * k * torch.rand(4 * hidden_size))
#        self.drop_h = nn.Dropout(dropout)
#        self.drop_c = nn.Dropout(dropout)
#        self.residual = residual
#
#    @jit.script_method
#    def forward(self, input, hx, cx):
#        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
#                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
#        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#        ingate = torch.sigmoid(ingate)
#        forgetgate = torch.sigmoid(forgetgate)
#        cellgate = torch.tanh(cellgate)
#        outgate = torch.sigmoid(outgate)
#
#        cy = (forgetgate * cx) + (ingate * cellgate)
#        hy = outgate * torch.tanh(cy)
#        # Residual connection
#        if self.residual:
#            hy = hy + input
#        # Zone-out
#        hy = torch.where(self.drop_h(hy)==0, hx, hy)
#        cy = torch.where(self.drop_c(cy)==0, cx, cy)
#        
#        return hy, cy
#
#class LSTMLayer(jit.ScriptModule):
#    def __init__(self, cell, *cell_args):
#        super(LSTMLayer, self).__init__()
#        self.cell = cell(*cell_args)
#
#    @jit.script_method
#    def forward(self, inputs, h, c):
#        inputs = inputs.unbind(0)
#        outputs = torch.jit.annotate(List[Tensor], [])
#        for i in range(len(inputs)):
#            h, c = self.cell(inputs[i], h, c)
#            outputs += [h]
#        return torch.stack(outputs)