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
import numpy as np
import pandas as pd
#%% Read neural methods
dataset = 'kaggle_webtraffic'
df = pd.read_csv(f'experiments/{dataset}/experiments_allresults_{dataset}.csv', sep=';')
cols = ['algorithm', 'test_smape_mean', 'test_nrmse_mean',
        'test_q50_mean', 'test_mean_quantile_loss_mean', 'test_smape_std', 'test_nrmse_std', 'test_q50_std',
        'test_mean_quantile_loss_std']
df_total = df[df['test_smape_std'] > 0][cols]
df_total = df_total.reset_index(drop=True)
algorithms = ['ets', 'lightgbm', 'seasonalnaive', 'theta']

# Add other methods
for algorithm in algorithms:
    df_algorithm = pd.read_csv(f'experiments/{dataset}/{algorithm}/{algorithm}_test.csv', index_col=0)
    df_total = df_total.append({'algorithm':algorithm}, ignore_index=True)
    df_total.loc[len(df_total)-1, 'test_smape_mean'] = df_algorithm[df_algorithm.Quantile == 0.5]['sMAPE'].values
    df_total.loc[len(df_total)-1, 'test_nrmse_mean']= df_algorithm[df_algorithm.Quantile == 0.5]['NRMSE'].values
    df_total.loc[len(df_total)-1, 'test_q50_mean'] = df_algorithm[df_algorithm.Quantile == 0.5]['QuantileLoss'].values
    df_total.loc[len(df_total)-1, 'test_mean_quantile_loss_mean'] = df_algorithm['QuantileLoss'].mean()

df_total.to_csv(f'experiments/{dataset}/allresults_allmethods.csv')