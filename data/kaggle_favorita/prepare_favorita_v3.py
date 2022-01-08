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

Basically label encode of everything; concatenate to item level and save one big data file
easy covariates
"""
import numpy as np
import pandas as pd
#%%
dim_inputseqlen = 90
dim_outputseqlen = 30
dim_maxseqlen = 150
#%%
import py7zr
# datasets = ['train', 'holiday_events','items', 'oil', 'transactions']
datasets = ['holidays_events']
dataset = datasets[0]
with py7zr.SevenZipFile(f'data/kaggle_favorita/{dataset}.csv.7z', mode='r') as z:
    z.extractall('data/kaggle_favorita/')


#%% Create subset
df_train = pd.read_csv('data/kaggle_favorita/train_adj.csv', parse_dates=['date'], index_col=[0], dtype={'store_nbr':'int32','item_nbr_new':'int32','unit_sales':'float32','onpromotion':'int8'})
# Subset
df_subset = df_train 
df_subset = df_train[df_train['date'] < '2014-04-01']
#%% Data subset
min_dates = df_subset.groupby(['store_nbr','item_nbr_new'])['date'].min().copy()
min_dates.name = 'date_first_sale'
df = df_subset.copy()
df.set_index(['date','store_nbr','item_nbr_new'], inplace=True)
df = df['unit_sales'].copy()
df = df.unstack(0).fillna(0).copy()
df = df.stack()
df = df.reset_index()
df.rename(columns={0:'unit_sales'}, inplace=True)
df = pd.merge(df, min_dates, left_on=['store_nbr','item_nbr_new'], right_on=['store_nbr','item_nbr_new'], how='left')
df = pd.merge(df, df_subset[['date','store_nbr','item_nbr_new','onpromotion']], left_on=['date','store_nbr','item_nbr_new'], right_on=['date','store_nbr','item_nbr_new'], how='left')
df = df.fillna(0)
df['days_since_first_sale'] = (df['date'] - df['date_first_sale']).dt.days
df = df[df['days_since_first_sale'] >= -dim_inputseqlen]
df = df.reset_index(drop=True)
df['unit_sales'] = df['unit_sales'].clip(lower=0)
df['on_sale'] = (df['days_since_first_sale'] >= 0) * 1
#%% Merge other
df_holiday = pd.read_csv('data/kaggle_favorita/holiday_adj.csv', parse_dates=['date'])
# Holidays has duplicates, we'll drop the duplicates for now and only use indicator for holiday
df_holiday = df_holiday.drop_duplicates(subset='date')
df_holiday['holiday'] = 1
df_holiday = df_holiday.drop(columns=['holiday_type','holiday_locale','holiday_locale_name','holiday_description','holiday_transferred'])
df_items = pd.read_csv('data/kaggle_favorita/items_adj.csv', dtype='int16')
#df_stores = pd.read_csv('data/kaggle_favorita/stores_adj.csv', dtype='int8')
df_oil = pd.read_csv('data/kaggle_favorita/oil_adj.csv', parse_dates=['date'])
df_transactions = pd.read_csv('data/kaggle_favorita/transactions.csv', parse_dates=['date'])
#%%
df = pd.merge(df, df_holiday, left_on='date', right_on='date', how='left').fillna(0)
df = df.astype({'holiday':'int16'})
#df = pd.merge(df, df_items, left_on='item_nbr_new', right_on='item_nbr_new', how='left')
#df = df.astype({'item_family':'int16', 'item_class':'int16', 'item_perishable':'int16'})
#df = pd.merge(df, df_stores, left_on='store_nbr', right_on='store_nbr', how='left')
df = pd.merge(df, df_transactions, left_on=['store_nbr','date'], right_on=['store_nbr','date'], how='left')
df = df.astype({'transactions':'float32'})
df['transactions_missing'] = df.transactions.isnull() * 1
df['transactions'] = df['transactions'].fillna(0)
df = pd.merge(df, df_oil, left_on='date', right_on='date', how='left')
df['oil_price_missing'] = df['oil_price_missing'].isnull() * 1 + df['oil_price_missing'].fillna(0)
df['oil_price'] = df['oil_price'].fillna(0)
df = df.astype({'oil_price':'float32','oil_price_missing':'int8'})
#%% Add lags
df['unit_sales_lagged'] = df.groupby(['store_nbr','item_nbr_new'])['unit_sales'].shift(1)
df['unit_sales_missing_lagged'] = df['unit_sales_lagged'].isnull() * 1
df['unit_sales_lagged'] = df['unit_sales_lagged'].fillna(0)
df['oil_price_lagged'] = df.groupby(['date'])['oil_price'].shift(1)
df['oil_price_lagged'] = df['oil_price_lagged'].fillna(0)
df['oil_price_missing_lagged'] = df.groupby(['date'])['oil_price_missing'].shift(1)
df['oil_price_missing_lagged'] = df['oil_price_missing_lagged'].fillna(1)
df['transactions_lagged'] = df.groupby(['date','store_nbr'])['transactions'].shift(1)
df['transactions_lagged'] = df['transactions_lagged'].fillna(0)
df['transactions_missing_lagged'] = df.groupby(['date','store_nbr'])['transactions_missing'].shift(1)
df['transactions_missing_lagged'] = df['transactions_missing_lagged'].fillna(1)
df['DayOfWeek_sin'] = np.sin(df.date.dt.dayofweek * (2 * np.pi / 7))
df['DayOfWeek_cos'] = np.cos(df.date.dt.dayofweek * (2 * np.pi / 7))
df['Month_sin'] = np.sin(df.date.dt.month * (2 * np.pi / 12))
df['Month_cos'] = np.cos(df.date.dt.month * (2 * np.pi / 12))
df_new = df[['date','unit_sales','item_nbr_new','store_nbr','holiday',  
         'onpromotion', 'on_sale', 'DayOfWeek_sin','DayOfWeek_cos', 'Month_sin', 'Month_cos',
         'transactions_missing_lagged', 'oil_price_missing_lagged', 'unit_sales_missing_lagged',
         'transactions_lagged', 'oil_price_lagged', 'unit_sales_lagged']]
#%% Create index
idx = df.groupby(['store_nbr','item_nbr_new'])['unit_sales'].rolling(dim_maxseqlen).count()
idx = idx[idx == dim_maxseqlen]
indices = (idx.index.get_level_values(2) - dim_maxseqlen + 1)
dates = df['date'][indices]
index_array = dates.reset_index()
#%%
df_new = df_new.drop(columns=['date'])
store = pd.HDFStore('data/kaggle_favorita/df_full_2013.h5')
df_new.to_hdf('data/kaggle_favorita/df_full_2013.h5', key='favorita')
index_array.to_hdf('data/kaggle_favorita/df_full_2013.h5', key='index')


