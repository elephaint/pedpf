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
import random
import numpy as np
import pandas as pd
import torch
from lib.loss_metrics import RMSE, ND, QuantileLoss, MAPE, sMAPE, NRMSE
import importlib
import time
#%% Helper functions

# Quantile function for StudenT(2) distribution
def StudentT2icdf(loc, scale, quantile):
    alpha = 4 * quantile * (1 - quantile)
    Q = 2 * (quantile - 0.5) * (2 / alpha).sqrt()
    
    return Q * scale + loc

# Fix seed
def fix_seed(seed):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Calculate and show metrics
def calc_metrics(yhat, y, quantiles):
    df = pd.DataFrame(columns={'RMSE','NRMSE','ND','MAPE','sMAPE','QuantileLoss','Quantile'})
    df.loc[:, 'Quantile'] = quantiles
    for q, quantile in enumerate(quantiles):
        df.loc[q, 'RMSE'] = RMSE(y, yhat[q])
        df.loc[q, 'NRMSE'] = NRMSE(y, yhat[q])
        df.loc[q, 'ND'] = ND(y, yhat[q])
        df.loc[q, 'MAPE'] = MAPE(y, yhat[q])
        df.loc[q, 'sMAPE'] = sMAPE(y, yhat[q])
        df.loc[q, 'QuantileLoss'] = QuantileLoss(y, yhat[q], quantile)
    q = 4
    print(f"         RMSE/NRMSE/ND/MAPE/sMAPE loss: {df['RMSE'][q]:.2f}/{df['NRMSE'][q]:.2f}/{df['ND'][q]:.3f}/{df['MAPE'][q]:.3f}/{df['sMAPE'][q]:.3f}")
    print(f"         p10/p50/p90/mp50 loss: {df['QuantileLoss'][0]:.3f}/{df['QuantileLoss'][4]:.3f}/{df['QuantileLoss'][8]:.3f}/{df['QuantileLoss'].mean():.3f}")
    
    return df

# Instantiate model based on string algorithm input
def instantiate_model(algorithm):
    model_class = importlib.import_module('algorithms.'+algorithm)
    model = getattr(model_class, algorithm)
    
    return model

# Count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Read experiment csv
def read_table(filename):
    for x in range(0, 10):
        try:
            table = pd.read_csv(filename, sep=';')
            error = None
        except Exception as error:
            pass
        
        if error:
            time.sleep(5)
        else:
            break
    
    return table

# Embedding dimensions for each dataset
def get_emb(dataset_name):
    if dataset_name == 'uci_electricity':
        d_emb = np.array([[370, 20]])
    elif dataset_name == 'uci_traffic':
        d_emb = np.array([[963, 20]])
    elif dataset_name == 'kaggle_favorita':
        d_emb = np.array([[3093, 8], [55, 3]])
    elif dataset_name == 'kaggle_webtraffic':
        d_emb = np.array([[10000, 20]])
    elif dataset_name == 'kaggle_m5':
        d_emb = np.array([[3049, 20], [7, 3], [3, 3], [10, 3], [3, 2], [5, 3], [3, 2]])        
    
    return d_emb
