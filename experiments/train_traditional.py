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
import time
import torch
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
from lib.utils import fix_seed, calc_metrics
from data.datasets import timeseries_dataset
num_cores = 2
#%% Initialize parameters for datasets
datasets = ['uci_electricity','uci_traffic','kaggle_favorita', 'kaggle_webtraffic', 'kaggle_m5']
dim_inputseqlens = [168, 168, 90, 90, 90]
dim_outputseqlens = [24, 24, 28, 30, 28]
dim_maxseqlens = [500, 500, 150, 150, 119]
#%% Initiate experiment
dataset_id = 4
seed = 0
fix_seed(seed)
num_samples_test = 10000
#%% Load data
dataset_name = datasets[dataset_id]
experiment_dir = 'experiments/'+dataset_name
dim_inputseqlen = dim_inputseqlens[dataset_id] # Input sequence length
dim_outputseqlen = dim_outputseqlens[dataset_id]  # Output prediction length
dim_maxseqlen = dim_maxseqlens[dataset_id]
# Import data
dset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen)
test_set = dset.load('test')
# Initialize sample sets
id_samples_test = torch.randperm(len(test_set))[:num_samples_test]
#%% Run traditional models on each separate time series
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.theta import ThetaModel

data_subset = torchdata.Subset(test_set, id_samples_test)
num_samples = len(id_samples_test)
data_generator = torchdata.DataLoader(data_subset)
frequencies = ['H', 'H', 'D', 'D', 'D']
seasonalities = [24, 24, 7, 7, 7]
seasonality = seasonalities[dataset_id]
frequency = frequencies[dataset_id]
quantiles = np.arange(1, 10) / 10
yhat_tot_test = np.zeros((3, len(quantiles), dim_outputseqlen, num_samples_test), dtype='float32')
y_tot_test = np.zeros((dim_outputseqlen, num_samples_test))
data_max = 1.0 if dataset_name== 'uci_traffic' else 1e12
data_min = 0.0
alphas = [0.2, 0.4, 0.6, 0.8, 1]

t_ets, t_snaive, t_theta = 0, 0, 0
for i, (X, y) in enumerate(data_generator):
    print(f'Sample {i + 1}/{len(data_generator)}')
    y_train = X[0, :dim_inputseqlen + 1, -1].double().numpy()
    y_test =  X[0, dim_inputseqlen + 1:dim_inputseqlen + dim_outputseqlen + 1, -1]
    assert torch.allclose(y_test, y.squeeze())
    # Fill y_test
    y_tot_test[:, i] = y_test.numpy()
    df_train = pd.Series(y_train, index=pd.date_range('1979-01-01', periods=len(y_train), freq=frequency))
    # ETS model
    a = time.perf_counter()
    ets_model = ETSModel(df_train,  error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=seasonality)
    fit_ets = ets_model.fit(maxiter=10000)
    b = time.perf_counter()
    t_ets += b - a
    simulated = fit_ets.simulate(anchor="end", nsimulations=dim_outputseqlen, repetitions=100000).clip(data_min, data_max)
    yhat_tot_test[0, :, :, i] = np.quantile(simulated, quantiles, axis=1)
    # Seasonal naive model
    a = time.perf_counter()
    yhat_snaive = X[0, dim_inputseqlen + 1 - dim_outputseqlen:dim_inputseqlen + 1, -1].numpy()
    yhat_tot_test[1, :, :, i] = yhat_snaive
    b = time.perf_counter()
    t_snaive += b-a    
    # Theta model
    a = time.perf_counter()
    theta_model = ThetaModel(df_train, period=seasonality)
    fit_theta = theta_model.fit()
    b = time.perf_counter()
    t_theta += b-a
    for j, alpha in enumerate(alphas):
        pi = fit_theta.prediction_intervals(dim_outputseqlen, alpha=alpha)
        yhat_tot_test[2, j, :, i] = pi.lower.clip(data_min, data_max)
        yhat_tot_test[2, -1 - j, :, i] = pi.upper.clip(data_min, data_max)
#%%
algorithms = ['ets', 'seasonalnaive', 'theta']
for a, algorithm in enumerate(algorithms):
    filename = f"{experiment_dir}/{algorithm}/{algorithm}"
    df_test = calc_metrics(yhat_tot_test[a], y_tot_test, quantiles)
    df_test.to_csv(filename + '_test.csv')
timings = pd.DataFrame(np.array([t_ets, t_snaive, t_theta])).T
timings.columns = algorithms
timings.to_csv(f"{experiment_dir}/timings_traditional.csv")