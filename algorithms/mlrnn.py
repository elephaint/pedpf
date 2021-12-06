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

ML-RNN
 https://arxiv.org/pdf/1711.11053.pdf
 Inspiration from: https://github.com/tianchen101/MQRNN/blob/master/Decoder.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class mlrnn(nn.Module):
    def __init__(self, d_lag, d_cov, d_emb, d_output, d_hidden, dropout, N, d_inputseqlen, d_outputseqlen, d_maxseqlen):
        super(mlrnn, self).__init__()
        self.d_outputseqlen = d_outputseqlen
        self.d_hidden = d_hidden
        # Embedding layer for time series ID
        self.emb = nn.ModuleList([nn.Embedding(d_emb[i, 0], d_emb[i, 1]) for i in range(len(d_emb))])
        d_emb_tot = d_emb[:, 1].sum()
        # LSTM Encoder
        self.lstm = nn.LSTM(d_lag + d_cov + int(d_emb_tot), d_hidden,  num_layers=N)
        # Global MLP decoder
        self.mlp_global = nn.Sequential(
                            nn.Linear(d_hidden + d_cov * (d_maxseqlen - d_inputseqlen), d_hidden * (d_outputseqlen + 1)),
                            nn.ReLU(),
                            nn.Linear(d_hidden * (d_outputseqlen + 1), d_hidden * (d_outputseqlen + 1)),
                            nn.ReLU(),
                            nn.Linear(d_hidden * (d_outputseqlen + 1), d_hidden * (d_outputseqlen + 1)))
        # Local MLP decoder
        self.mlp_local = nn.Sequential(
                            nn.Linear(d_hidden + d_hidden + d_cov, d_hidden + d_hidden + d_cov),
                            nn.ReLU(),
                            nn.Linear(d_hidden + d_hidden + d_cov, d_hidden + d_hidden + d_cov),
                            nn.ReLU(),
                            nn.Linear(d_hidden + d_hidden + d_cov, 2))        
        self.epsilon = 1e-6


    def forward(self, x_lag, x_cov, x_idx, d_outputseqlen):
        batch_size = x_lag.shape[1]
        # Embedding layers
        x_emb = []
        for i, layer in enumerate(self.emb):
            out = layer(x_idx[:, :, i])
            x_emb.append(out)
        x_emb = torch.cat(x_emb, -1)
        # Concatenate x_lag, x_cov and time series ID up to inputseqlen
        d_inputseqlen = x_lag.shape[0] - d_outputseqlen
        inputs = torch.cat((x_lag[:d_inputseqlen], x_cov[:d_inputseqlen], x_emb[:d_inputseqlen]), dim=-1)
        # Encoder network for everything up to inputseqlen
        _, (h_enc, _) = self.lstm(inputs)
        ht = h_enc[-1]
        # Global decoder of future covariates and encoder output
        x_cov_future = x_cov[d_inputseqlen:].permute(1, 0, 2)
        x_cov_future = x_cov_future.reshape(batch_size, -1)
        input_global = torch.cat((ht, x_cov_future), dim=-1)
        h_global = self.mlp_global(input_global)
        h_global = h_global.reshape(batch_size, self.d_hidden, self.d_outputseqlen + 1)
        h_global = h_global.permute(2, 0, 1)
        # Local decoder
        y = torch.zeros((d_outputseqlen, batch_size, 2), dtype=torch.float32, device=x_lag.device)
        for t in range(d_outputseqlen):
            input_local = torch.cat((h_global[t], h_global[-1], x_cov[d_inputseqlen + t]), dim=-1)
            y[t] = self.mlp_local(input_local)
        
        # Output layers - location and scale of distribution
        loc, scale = F.softplus(y).chunk(2, -1)
        return loc, scale + self.epsilon