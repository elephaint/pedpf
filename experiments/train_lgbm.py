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
import time
import torch
import pandas as pd
import lightgbm as lgb
import torch.utils.data as torchdata
import numpy as np
import optuna
import joblib
from lib.utils import fix_seed, calc_metrics
from data.datasets import timeseries_dataset
#%% Initialize parameters for datasets
datasets = ['uci_electricity','uci_traffic','kaggle_favorita', 'kaggle_webtraffic', 'kaggle_m5']
dim_inputseqlens = [168, 168, 90, 90, 90]
dim_outputseqlens = [24, 24, 28, 30, 28]
dim_maxseqlens = [500, 500, 150, 150, 119]
#%% Initiate experiment
algorithm = 'lightgbm'
dataset_id = 0
seed = 0
fix_seed(seed)
num_samples_train = 1500000 if datasets[dataset_id] == 'kaggle_m5' else 500000
num_samples_validate = 30000 if datasets[dataset_id] == 'kaggle_m5' else 10000
num_samples_test = 10000
#%% Load data
dataset_name = datasets[dataset_id]
experiment_dir = 'experiments/'+dataset_name
filename = f"{experiment_dir}/{algorithm}/{algorithm}"
dim_inputseqlen = dim_inputseqlens[dataset_id] # Input sequence length
dim_outputseqlen = dim_outputseqlens[dataset_id]  # Output prediction length
dim_maxseqlen = dim_maxseqlens[dataset_id]
# Import data
dset = timeseries_dataset(dataset_name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen)
training_set = dset.load('train')
validation_set = dset.load('validate')
test_set = dset.load('test')
# Initialize sample sets
id_samples_train = torch.randperm(len(training_set))[:num_samples_train]
id_samples_validate = torch.randperm(len(validation_set))[:num_samples_validate]
fix_seed(seed)
id_samples_test = torch.randperm(len(test_set))[:num_samples_test]
#%% Make datasets appropriate for LightGBM
# Get data
def get_Xy(data, id_samples):
    data_subset = torchdata.Subset(data, id_samples)
    dataloader = torch.utils.data.DataLoader(data_subset, batch_size=len(data))
    X, y = iter(dataloader).next()
    return X, y.squeeze().T

# Build datasets for lgb
def create_data(X, y, t, dim_inputseqlen, dim_outputseqlen):
    X_cov = X[:, t:t + dim_inputseqlen + dim_outputseqlen + 1, :-1].reshape(X.shape[0], -1)
    X_lag = X[:, t:t + dim_inputseqlen + 1, -1]
    if t > 0:
        X_lag[:, -t:] = y[:t].T
        
    X_lag = X_lag.reshape(X.shape[0], -1)
    X_out = torch.cat((X_cov, X_lag), dim=1)
    # assert torch.allclose(y[t], X[:, dim_inputseqlen + 1 + t:dim_inputseqlen + 1 + 1 + t, -1].squeeze())
    
    return (X_out.contiguous().numpy(), y[t].contiguous().numpy())

# Get data from torchdataset
X_train, y_train = get_Xy(training_set, id_samples_train)
X_val, y_val = get_Xy(validation_set, id_samples_validate)
X_test, y_test = get_Xy(test_set, id_samples_test)
# Create numpy datasets in correct tabular format
train_data = create_data(X_train, y_train, 0, dim_inputseqlen, dim_outputseqlen)
valid_data = create_data(X_val, y_val, 0, dim_inputseqlen, dim_outputseqlen)
test_data = create_data(X_test, y_test, 0, dim_inputseqlen, dim_outputseqlen)
#%% Load best hyperparameters from HPO optimization if it exists
params = {  'n_estimators':2000,
            'early_stopping_rounds':100,
            'max_bin':1024,
            'objective':'quantile',
            'seed':seed,
            'force_col_wise':True}
try:
    study = joblib.load(f"{filename}_hpo.pkl")
    for key, value in study.best_trial.params.items():
        params[key] = value
except FileNotFoundError:
        params['reg_alpha'] = 1e-8
        params['reg_lambda'] = 2
        params['num_leaves'] = 100
        params['feature_fraction'] = 0.90
        params['bagging_fraction'] = 0.40
        params['min_data_in_leaf'] = 100

# LGB datasets
train_set_lgb = lgb.Dataset(train_data[0], label=train_data[1], categorical_feature=np.arange(training_set.d_emb).tolist(), free_raw_data=False)
valid_set_lgb = lgb.Dataset(valid_data[0], label=valid_data[1], categorical_feature=np.arange(validation_set.d_emb).tolist(), free_raw_data=False)

#%% Execute experiment
quantiles = np.arange(1, 10) / 10
# quantiles = np.array([0.5])
total_time = 0
# Create arrays to be filled
yhat_tot_validate = torch.zeros((len(quantiles), dim_outputseqlen, num_samples_validate), dtype=torch.float32)
yhat_tot_test = torch.zeros((len(quantiles), dim_outputseqlen, num_samples_test), dtype=torch.float32)
data_max = 1.0 if dataset_name== 'uci_traffic' else 1e12
data_min = 0.0

# Loop over all quantiles
for q, quantile in enumerate(quantiles):
    params['alpha'] = quantile
    # Train to retrieve best iteration
    print('Validating...')
    start = time.perf_counter()    
    model = lgb.train(params, train_set_lgb, valid_sets=[train_set_lgb, valid_set_lgb])
    end = time.perf_counter()
    validation_time = end - start
    total_time += validation_time
    print(f'Fold time: {validation_time:.2f}s')
    #% Predictions - recursive predictions
    print('Prediction...')
    for t in range(dim_outputseqlen):
        print(f'Predicting quantile {quantile} - timestep {t + 1}')
        # Create current input data
        X_val_current, _ = create_data(X_val, yhat_tot_validate[q], t, dim_inputseqlen, dim_outputseqlen)
        X_test_current, _ = create_data(X_test, yhat_tot_test[q], t, dim_inputseqlen, dim_outputseqlen)
        # Predict current timestep
        yhat_val_current = model.predict(X_val_current).clip(data_min, data_max)
        yhat_test_current = model.predict(X_test_current).clip(data_min, data_max)
        # Fill bookkeeping variable
        yhat_tot_validate[q, t] = torch.from_numpy(yhat_val_current)
        yhat_tot_test[q, t] = torch.from_numpy(yhat_test_current)
   
# Compute statistics for all quantiles and entire prediction horizon
df_validate = calc_metrics(yhat_tot_validate.numpy(), y_val.numpy(), quantiles)
df_test = calc_metrics(yhat_tot_test.numpy(), y_test.numpy(), quantiles)
#%% Save
df_validate.to_csv(filename + '_validate.csv')
df_test.to_csv(filename + '_test.csv')
timings = pd.DataFrame([total_time])
timings.columns = [algorithm]
timings.to_csv(f"{experiment_dir}/{algorithm}/timings.csv")
#%% Optimizer hyperparameters
# We optimize the hyperparameters on the quantile 0.5 level on the validation 
# set, for 100 trials of Bayesian HPO
class Objective(object):
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid
        
    def __call__(self, trial):
        n_estimators = 2000
        early_stopping_rounds = 100
        params = {
            'verbosity': 1,
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 1000),
            'objective':'quantile',
            'alpha':0.5,
            'seed':2,
            'force_col_wise':True,
            'feature_pre_filter': False
        }
               
       
        model = lgb.train(params, self.train, 
                          valid_sets=[self.train, self.valid], 
                          num_boost_round = n_estimators,
                          early_stopping_rounds=early_stopping_rounds)
        
        score = model.best_score['valid_1']['quantile']
        
        return score

# LGB datasets
train_set_lgb = lgb.Dataset(train_data[0], label=train_data[1], categorical_feature=np.arange(training_set.d_emb).tolist(), free_raw_data=False)
valid_set_lgb = lgb.Dataset(valid_data[0], label=valid_data[1], categorical_feature=np.arange(validation_set.d_emb).tolist(), free_raw_data=False)

# Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
objective = Objective(train_set_lgb, valid_set_lgb)
study.optimize(objective, n_trials=100)
print(study.best_trial)
joblib.dump(study, f"{filename}_hpo.pkl")