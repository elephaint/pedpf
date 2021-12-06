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
#%% Import packages
import torch
import torch.optim as optim
import numpy as np
from joblib import Parallel, delayed
from lib.utils import fix_seed, instantiate_model, count_parameters, read_table, get_emb
from lib.train import loop
from data.datasets import timeseries_dataset
import pandas as pd
torch.backends.cudnn.benchmark = False
num_cores = 2
torch.set_num_threads(2)
#%% Initialize parameters for datasets
datasets = ['uci_electricity','uci_traffic','kaggle_favorita','kaggle_webtraffic']
dim_inputseqlens = [168, 168, 90, 90]
dim_outputseqlens = [24, 24, 30, 30]
dim_maxseqlens = [500, 500, 150, 150]
#%% Initiate experiment
dataset_id = 0
cuda = 0
seed = 0
fix_seed(seed)
num_samples_test = 10000
scaling = True
#%% Load data
dataset_name = datasets[dataset_id]
experiment_dir = 'experiments/'+dataset_name
dim_inputseqlen = dim_inputseqlens[dataset_id] # Input sequence length
dim_outputseqlen = dim_outputseqlens[dataset_id]  # Output prediction length
dim_maxseqlen = dim_maxseqlens[dataset_id]
# Import data
dset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen)
# test_set = dset.load('test')
# Initialize sample sets
id_samples_test = torch.randperm(len(test_set))[:num_samples_test]
#%% Algorithm parameters
file_experiments = experiment_dir + f'/experiments_gaussian_{dataset_name}.csv'
table = read_table(file_experiments)
d_emb = get_emb(dataset_name)
while table[table['in_progress'] == -1].isnull()['test_score'].sum() > 0:
    # Read experiment table, set hyperparameters
    idx = table[table['in_progress'] == -1].isnull()['test_score'].idxmax()
    algorithm = table.loc[idx, 'algorithm']
    learning_rate = table.loc[idx, 'learning_rate']
    batch_size = int(table.loc[idx, 'batch_size'])
    d_hidden = int(table.loc[idx, 'd_hidden'])
    # Following paper of TransformerConv, hidden dimension is defined by covariates, lags and embedding dims
    if algorithm == 'transformer_conv':
        d_hidden = test_set.d_cov + test_set.d_lag + d_emb[:, 1].sum()
    kernel_size = int(table.loc[idx, 'kernel_size'])
    N = int(table.loc[idx, 'N'])
    dropout = table.loc[idx, 'dropout']
    seed = table.loc[idx, 'seed']
    device = torch.device(cuda)
    params = eval(table.loc[idx, 'params_test'])
    # Test loop
    filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}_gaussian"
    fix_seed(seed)
    n_batch_test = (len(id_samples_test) + batch_size - 1) // batch_size
    if 'model' in locals(): del model
    model = instantiate_model(algorithm)(*params) 
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer=None
    _, loss_test, yhat_tot, y_tot, x_tot, df_test = loop(model, test_set, optimizer, batch_size, id_samples_test, train=False, metrics=True, scaling=scaling)    
    df_test.to_csv(filename + '_test.csv')
    # Write table
    table.loc[idx, 'test_score'] = loss_test / n_batch_test
    table.loc[idx, 'in_progress'] = -1
    table.to_csv(file_experiments, sep=';', index=False)
#%% Combine results
n_seeds = 5
file_experiments = experiment_dir + f'/experiments_gaussian_{dataset_name}.csv'
df_dataset = read_table(file_experiments)
df_dataset['test_smape'] = 0
df_dataset['test_nrmse'] = 0
df_dataset['test_q50'] = 0
df_dataset['test_mean_quantile_loss'] = 0
for idx in range(len(table)):
    algorithm = df_dataset.loc[idx, 'algorithm']
    learning_rate = df_dataset.loc[idx, 'learning_rate']
    batch_size = int(df_dataset.loc[idx, 'batch_size'])
    d_hidden = int(df_dataset.loc[idx, 'd_hidden'])
    seed  = int(df_dataset.loc[idx, 'seed'])
    filename = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}_gaussian_test.csv"
    df_test_current = pd.read_csv(filename)       
    df_dataset.loc[idx, 'test_smape']  = df_test_current[df_test_current.Quantile == 0.5]['sMAPE'].values
    df_dataset.loc[idx, 'test_nrmse']= df_test_current[df_test_current.Quantile == 0.5]['NRMSE'].values
    df_dataset.loc[idx, 'test_q50']= df_test_current[df_test_current.Quantile == 0.5]['QuantileLoss'].values
    df_dataset.loc[idx, 'test_mean_quantile_loss'] = df_test_current['QuantileLoss'].mean()
        
df_mean = df_dataset.groupby(['algorithm'])[['test_smape', 'test_nrmse', 'test_q50','test_mean_quantile_loss']].mean()
df_std = df_dataset.groupby(['algorithm'])[['test_smape', 'test_nrmse', 'test_q50','test_mean_quantile_loss']].std()