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
import datetime
#%% Remove some unwanted stuff
with open('data/uci_traffic/PEMS_train', 'r') as infile, \
     open('data/uci_traffic/PEMS_train_adj', 'w') as outfile:
    data = infile.read()
    data = data.replace("[", "")
    data = data.replace("]", "")
    data = data.replace(";", " ")
    outfile.write(data)
    
with open('data/uci_traffic/PEMS_test', 'r') as infile, \
     open('data/uci_traffic/PEMS_test_adj', 'w') as outfile:
    data = infile.read()
    data = data.replace("[", "")
    data = data.replace("]", "")
    data = data.replace(";", " ")
    outfile.write(data)
    
with open('data/uci_traffic/randperm', 'r') as infile, \
     open('data/uci_traffic/randperm_adj', 'w') as outfile:
    data = infile.read()
    data = data.replace("[", "")
    data = data.replace("]", "")
    outfile.write(data)

#%% Load original data, concatenate, so as to construct correct time order
df1 = np.loadtxt('data/uci_traffic/PEMS_train_adj', delimiter=' ')
df2 = np.loadtxt('data/uci_traffic/PEMS_test_adj', delimiter=' ')
df = np.concatenate((df1, df2), axis=0)
randperm_old = np.loadtxt('data/uci_traffic/randperm_adj.txt', dtype='int')
randperm_new = np.arange(0, 440)
randperm = np.stack((randperm_old, randperm_new), axis=1)
time_order = randperm[randperm[:, 0].argsort(), 1]
# Correct time order
df_correct = df[time_order].astype('float32')
# Reshape to [num_dates, num_series, num_hours, num_samples_per_hour]
df_correct = df_correct.reshape(-1, 963, 24, 6)
# Use only hourly data by averaging over samples per hour
df_correct = df_correct.mean(axis=-1)
# Use only dates up to 1-10-2008 (because these can be reconciled with actual dates & missing information - see Excel file)
data = df_correct[:267]
data = data.transpose(0, 2, 1)
#%% Create pd dataframe. We use hourly data up to 1-10-2008
index = pd.date_range('2008-01-01', periods=6576, freq='h')
index = pd.MultiIndex.from_arrays([index.date, index.time], names=['date','time'])
dates = pd.to_datetime(index.get_level_values(0).unique(), format='%Y-%m-%d').date
index_dates = pd.read_csv('data/uci_traffic/index_dates.csv', delimiter=';', index_col=[0])
index_dates = pd.to_datetime(index_dates['Date'], format='%d/%m/%Y')
index_dates = pd.Index(index_dates)[:267]
index_dates = index_dates.date
columns = ['carlane_'+str(i+1) for i in range(963)]
df_new = pd.DataFrame(index=index, columns=columns)
df_new['missing'] = 0
j = 0
for date in dates:
    if date in index_dates:
        df_new.loc[(date, slice(None)), 'carlane_1':'carlane_963'] = data[j]
        df_new.loc[(date, slice(None)), 'missing'] = 0
        j += 1
    else:
        df_new.loc[(date, slice(None)), 'carlane_1':'carlane_963'] = 0
        df_new.loc[(date, slice(None)), 'missing'] = 1

df_new.to_csv('data/uci_traffic/dataset.csv')