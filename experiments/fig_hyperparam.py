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
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'font.weight': 'normal'})
datasets = ['uci_electricity','uci_traffic','kaggle_favorita','kaggle_webtraffic']
#%% Plot
def create_plot(dataset, ylim):
    experiment_dir = 'experiments/'+dataset
    file_experiments = experiment_dir + f'/experiments_{dataset}.csv'
    df = pd.read_csv(file_experiments, sep=';')    
    algorithms = ['deepar', 'transformer_conv', 'tcn', 'wavenet',  'mlrnn', 'bitcn']
    learning_rates = [0.001, 0.0005, 0.0001]
    n_seeds = 1
    n_epochs = 100
    seeds = np.arange(n_seeds)
    plots = ['r-','g--','b-.']
    fig, axs = plt.subplots(nrows = len(algorithms), ncols = len(learning_rates), sharex=True, sharey=True)
    x = np.arange(n_epochs)
    # Loop over algorithms and learning rates
    for i, algorithm in enumerate(algorithms):
        for j, learning_rate in enumerate(learning_rates):
            batch_sizes = np.sort(df[df.algorithm == algorithm]['batch_size'].unique()).tolist()
            # This is to catch omittance of this single expceriment for MLRNN which kept giving errors
            if (dataset == 'uci_traffic') & (algorithm == 'mlrnn') & (learning_rate == 0.0001):
                batch_sizes = [64, 256]
                axs[i, j].grid()
            Y = np.zeros((len(batch_sizes), len(seeds), n_epochs))
            for k, batch_size in enumerate(batch_sizes):
                for l, seed in enumerate(seeds):
                    df_current = df[(df.algorithm == algorithm) & (df.seed == seed) & (df.batch_size == batch_size) & (df.learning_rate == learning_rate)]
                    d_hidden = int(df_current['d_hidden'])
                    filename_loss = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}_loss.csv"
                    df_loss_current = pd.read_csv(filename_loss, usecols=['Validation_loss'])
                    df_loss_current.loc[df_loss_current['Validation_loss'] == 0] = np.nan
                    Y[k, l, :] = df_loss_current['Validation_loss'].values
                # Plot
                axs[i, j].plot(x, Y[k].mean(axis=0), plots[k], label=f'{batch_size}', linewidth=3, )
                axs[i, j].legend(loc = 'upper right')
                axs[i, j].grid()
#                axs[i, j].xaxis.set_ticks(np.arange(min(x), max(x)+1, 5))                
                axs[i, j].yaxis.set_tick_params(which='both', labelbottom=True)
                axs[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
                axs[i, j].set_title(f'{algorithm} / {learning_rate}')
                axs[i, j].set_ylim(ylim[0], ylim[1])
                # Only plot ylabel for first column
                if j == 0:
                    axs[i, j].set_ylabel('Validation loss')
                # Only plot xlabel for last row
                if i == len(algorithms) - 1:
                    axs[i, j].set(xlabel='Epochs')
#%% Electricity
dataset = datasets[0]
ylim = [-1.2, -0.2]
create_plot(dataset, ylim)
#%% Traffic
dataset = datasets[1]
ylim = [-4.8, -2]
create_plot(dataset, ylim)
#%% Favorita
dataset = datasets[2]
ylim = [-3.4, 0]
create_plot(dataset, ylim)
#%% Webtraffic
dataset = datasets[3]
ylim = [0.2, 1.5]
create_plot(dataset, ylim)