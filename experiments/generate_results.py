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
import pandas as pd
import numpy as np
#%% Load & process GPU-Z sensor log
df = pd.read_csv('experiments/gpuz_sensor_log.txt', encoding='ansi', sep=',', parse_dates=[0])
column_names = df.columns.tolist()
column_names = [name.strip() for name in column_names]
df.columns = column_names
df.set_index('Date', inplace=True)
# Keep only relevant columns
df = df[['GPU Clock [MHz]','Memory Clock [MHz]', 'GPU Temperature [°C]', 'GPU Load [%]' , 
         'Board Power Draw [W]','Power Consumption (%) [% TDP]']]
# Some processing
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Save file
df.to_csv('experiments/gpuz_sensor_log_processed.csv')
#%% Calculate load and processing time per experiment based on GPU-Z log file
n_epochs = 1
df_gpuz = pd.read_csv('experiments/gpuz_sensor_log_processed.csv', parse_dates=[0], dayfirst=True)
datasets = ['uci_electricity','uci_traffic','kaggle_favorita','kaggle_webtraffic']
# datasets = ['kaggle_webtraffic']

for dataset in datasets:
    # Load dataset
    df_dataset = pd.read_csv(f'experiments/{dataset}/experiments_epochtime_{dataset}.csv', sep=';', parse_dates=[13, 14], dayfirst=True)
    df_dataset.drop(columns=['seed', 'score', 'test_score', 'in_progress', 'params_train','params_test'], inplace=True)
    experiment_dir = 'experiments/'+dataset
    df_dataset_exp = pd.read_csv(f'experiments/{dataset}/experiments_{dataset}.csv', sep=';')
    # Create new columns
    df_dataset['load_time_per_epoch'] = 0
    df_dataset['idle_time_per_epoch'] = 0
    df_dataset['energy_per_epoch'] = 0
    df_dataset['n_epochs_avg'] = 0
    df_dataset['load_time'] = 0
    df_dataset['energy'] = 0
    df_dataset['validation_loss'] = 0
    df_dataset['test_smape_mean'] = 0
    df_dataset['test_smape_std'] = 0
    df_dataset['test_nrmse_mean'] = 0
    df_dataset['test_nrmse_std'] = 0
    df_dataset['test_q50_mean'] = 0
    df_dataset['test_q50_std'] = 0
    df_dataset['test_mean_quantile_loss_mean'] = 0
    df_dataset['test_mean_quantile_loss_std'] = 0
    # Loop over all experimental settings
    for i in range(len(df_dataset)):
        # Start and end date of experiment
        start_date = df_dataset['start_time'][i]    
        end_date = df_dataset['end_time'][i]
        # Select right data from GPU-Z log file. 
        df_current = df_gpuz[(df_gpuz.Date >= start_date) & (df_gpuz.Date <= end_date)]
        train_time = len(df_current)  # the total training time in seconds for the 5 epochs is the length of this dataframe
        # Ignore the idle time between epochs
        df_current = df_current[df_current['GPU Load [%]'] > 0]
        # Calculate time and energy consumption per epoch
        load_time_per_epoch = len(df_current) / n_epochs
        idle_time_per_epoch = (train_time - len(df_current)) / n_epochs
        energy_per_epoch = df_current['Board Power Draw [W]'].sum() / n_epochs
        # Save to file
        df_dataset.loc[i, 'load_time_per_epoch'] = load_time_per_epoch
        df_dataset.loc[i, 'idle_time_per_epoch'] = idle_time_per_epoch
        df_dataset.loc[i, 'energy_per_epoch'] = energy_per_epoch
        # Calculate average no. epochs
        n_epochs_avg = 0
        validation_loss = 0
        # Find number of seeds
        algorithm = df_dataset['algorithm'].iloc[i]
        learning_rate = df_dataset['learning_rate'].iloc[i]
        batch_size = df_dataset['batch_size'].iloc[i]
        n_seeds = len(df_dataset_exp[(df_dataset_exp['algorithm'] == algorithm) & 
                                 (df_dataset_exp['learning_rate'] == learning_rate) &
                                 (df_dataset_exp['batch_size'] == batch_size)])
        test_smape = np.zeros(n_seeds)
        test_nrmse = np.zeros(n_seeds)
        test_q50 = np.zeros(n_seeds)
        test_mean_quantile_loss = np.zeros(n_seeds)
        batch_size = df_dataset.loc[i, 'batch_size']
        d_hidden = df_dataset.loc[i, 'd_hidden']
        learning_rate = df_dataset.loc[i, 'learning_rate']
        algorithm = df_dataset.loc[i, 'algorithm']
        for seed in range(n_seeds):  
            filename_loss = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}_loss.csv"
            df_loss_current = pd.read_csv(filename_loss, usecols=['Validation_loss'])
            n_epochs_avg_seed = df_loss_current[df_loss_current['Validation_loss'] != 0].index.max() + 1
            # Calculate metrics
            n_epochs_avg += n_epochs_avg_seed / n_seeds 
            validation_loss += df_loss_current[df_loss_current['Validation_loss'] != 0].min().values / n_seeds 
            # Calculate test losses
            filename_test = f"{experiment_dir}/{algorithm}/{algorithm}_seed={seed}_hidden={d_hidden}_lr={learning_rate}_bs={batch_size}_test.csv"
            df_test_current = pd.read_csv(filename_test, index_col=[0])
            test_smape[seed] = df_test_current[df_test_current.Quantile == 0.5]['sMAPE'].values
            test_nrmse[seed] = df_test_current[df_test_current.Quantile == 0.5]['NRMSE'].values
            test_q50[seed] = df_test_current[df_test_current.Quantile == 0.5]['QuantileLoss'].values
            test_mean_quantile_loss[seed] = df_test_current['QuantileLoss'].mean()
            
            
        df_dataset.loc[i, 'n_epochs_avg'] = n_epochs_avg
        # Calculate total average load time and energy per experimental setting
        df_dataset.loc[i, 'load_time'] = n_epochs_avg * load_time_per_epoch
        df_dataset.loc[i, 'energy'] = n_epochs_avg * energy_per_epoch
        # Write other metrics
        df_dataset.loc[i, 'validation_loss'] = validation_loss
        df_dataset.loc[i, 'test_smape_mean'] = test_smape.mean()
        df_dataset.loc[i, 'test_smape_std'] = test_smape.std()
        df_dataset.loc[i, 'test_nrmse_mean'] = test_nrmse.mean()
        df_dataset.loc[i, 'test_nrmse_std'] = test_nrmse.std()
        df_dataset.loc[i, 'test_q50_mean'] = test_q50.mean()
        df_dataset.loc[i, 'test_q50_std'] = test_q50.std()
        df_dataset.loc[i, 'test_mean_quantile_loss_mean'] = test_mean_quantile_loss.mean()
        df_dataset.loc[i, 'test_mean_quantile_loss_std'] = test_mean_quantile_loss.std()     
    
    # Write to csv
    df_dataset.to_csv(f'experiments/{dataset}/experiments_allresults_{dataset}.csv', sep=';', index=False)
