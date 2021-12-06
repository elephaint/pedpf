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
#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'font.weight': 'normal'})
datasets = ['uci_electricity','uci_traffic','kaggle_favorita','kaggle_webtraffic']
#%% Running time
dataset = datasets[0]
df = pd.read_csv(f'experiments/{dataset}/experiments_allresults_{dataset}.csv', sep=';', parse_dates=[13, 14], dayfirst=True)
df = df[df['test_smape_std'] > 0]
df = df.reset_index(drop=True)
df['load_time'] = df['load_time'].astype('float')

algorithms = ['deepar', 'transformer_conv', 'tcn', 'wavenet', 'bitcn', 'mlrnn']
learning_rate = df[df['algorithm'] == 'deepar']['learning_rate'].min()
baseline_loadtime = df[(df['algorithm'] == 'deepar') & (df['learning_rate'] == learning_rate) & (df['batch_size'] == 64)]['load_time'].values
baseline_energy = df[(df['algorithm'] == 'deepar') & (df['learning_rate'] == learning_rate) & (df['batch_size'] == 64)]['energy'].values
tickers = ['^','o','s','x','P','v']
fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex=True, sharey=True)
for i, algorithm in enumerate(algorithms):
    subset = df[df['algorithm'] == algorithm]    
    x_time = subset['load_time'] / baseline_loadtime    
    x_energy = subset['energy'] / baseline_energy
    y = subset['test_mean_quantile_loss_mean']   
    axs[0].plot(x_time, y, tickers[i], label=algorithm)
    axs[1].plot(x_energy, y, tickers[i], label=algorithm)

axs[0].legend()
axs[1].legend()
axs[0].set_ylabel('Mean Quantile loss')
axs[0].set(xlabel = 'Running time')
axs[1].set(xlabel = 'Energy cost')
#axs[0].set_title('Running time')
#axs[1].set_title(f'Energy cost')

#%% Running time
#dataset = datasets[3]
datasets = ['uci_electricity','uci_traffic','kaggle_favorita','kaggle_webtraffic']
algorithms = ['deepar', 'transformer_conv', 'tcn', 'wavenet', 'bitcn', 'mlrnn']
algorithms_traditional = ['seasonalnaive', 'ets', 'theta', 'lightgbm']
fig, axs = plt.subplots(nrows = len(datasets), ncols = 2, sharex=True)
for i, dataset in enumerate(datasets):
    df = pd.read_csv(f'experiments/{dataset}/experiments_allresults_{dataset}.csv', sep=';', parse_dates=[13, 14], dayfirst=True)
    df = df[df['test_mean_quantile_loss_std'] > 0]
    # df = df[df['test_mean_quantile_loss_mean'] > 0]
    # df = df[df['algorithm'] != 'bitcn_noforward']
    df = df.reset_index(drop=True)
    df['load_time'] = df['load_time'].astype('float')
    learning_rate = df[df['algorithm'] == 'deepar']['learning_rate'].min()
    batch_size = df[df['algorithm'] == 'deepar']['batch_size'].min()
    baseline_loadtime = df[(df['algorithm'] == 'deepar') & (df['learning_rate'] == learning_rate) & (df['batch_size'] == batch_size)]['load_time'].values
    baseline_energy = df[(df['algorithm'] == 'deepar') & (df['learning_rate'] == learning_rate) & (df['batch_size'] == batch_size)]['energy'].values
    # Neural
    tickers = ['^','o','s','x','P','v']
    for j, algorithm in enumerate(algorithms):
        subset = df[df['algorithm'] == algorithm]    
        x_time = subset['load_time'] / baseline_loadtime    
        x_energy = subset['energy'] / baseline_energy
        y = subset['test_mean_quantile_loss_mean']   
        axs[i, 0].plot(x_time, y, tickers[j], label=algorithm)
        axs[i, 1].plot(x_energy, y, tickers[j], label=algorithm)
    
    # Traditional
    tickers = ['<','>','8','p']
    df_time = pd.read_csv(f'experiments/{dataset}/timings_traditional.csv')
    for j, algorithm in enumerate(algorithms_traditional):
        df_test = pd.read_csv(f'experiments/{dataset}/{algorithm}/{algorithm}_test.csv')
        if algorithm == 'lightgbm': df_time = pd.read_csv(f'experiments/{dataset}/{algorithm}/timings.csv')
        x_time = df_time[algorithm] / baseline_loadtime    
        y = df_test['QuantileLoss'].mean()   
        axs[i, 0].plot(x_time, y, tickers[j], label=algorithm)
    
    axs[i, 0].legend(ncol=2)
    axs[i, 1].legend(ncol=2)
    axs[i, 0].set_ylabel('Mean Quantile loss')

axs[-1, 0].set(xlabel = 'Training time')
axs[-1, 1].set(xlabel = 'Energy cost')
#axs[0].set_title('Running time')
#axs[1].set_title(f'Energy cost')