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
import torch
import torch.utils.data as torchdata
import pandas as pd
#%%
# Read data from source
df = pd.read_csv('data/kaggle_webtraffic/train_2.csv', index_col=[0]).fillna(0)
# Get length of series and dates
num_series = len(df)
num_dates = len(df.columns)
# Remove page names, we're ignoring names
idx_series = [str(i) for i in range(num_series)]
df.index = idx_series
#%% Select top10000 series up to 31-12-2016
pageviews = df.loc[:, :'2016-12-31'].sum(axis=1)
indices = pageviews.sort_values(ascending=False)[:10000].index
df = df.loc[indices]
#%%
df = pd.DataFrame(df.stack())
df.columns = ['Page_views']
df.index = df.index.set_levels([df.index.levels[0], pd.to_datetime(df.index.levels[1])])
df.index.names = ['series','date']
#%%
# Stack and create new dataframe
df['Series'] = df.index.get_level_values(0)
# Add time-based covariates
dates = df.index.get_level_values(1)
dates_dow = dates.dayofweek
dates_day = dates.day
dates_week = dates.week
dates_month = dates.month
df['DayOfWeek_sin'] = np.sin(dates_dow * (2 * np.pi / 7))
df['DayOfWeek_cos'] = np.cos(dates_dow * (2 * np.pi / 7))
df['DayOfMonth_sin'] = np.sin(dates_day * (2 * np.pi / 31))
df['DayOfMonth_cos'] = np.cos(dates_day * (2 * np.pi / 31))
df['WeekOfYear_sin'] = np.sin(dates_week * (2 * np.pi / 53))
df['WeekOfYear_cos'] = np.cos(dates_week * (2 * np.pi / 53))
df['MonthOfYear_sin'] = np.sin(dates_month * (2 * np.pi / 12))
df['MonthOfYear_cos'] = np.cos(dates_month * (2 * np.pi / 12))
# Add lagged output variable
df['Page_views_lag'] = df.groupby(level=0)['Page_views'].shift(1)
# Remove first date (contains NaNs for the lagged column)
df = df.loc[pd.IndexSlice[:, '2015-07-02':], :]      
#%% Create subsets 
store = pd.HDFStore('data/kaggle_webtraffic/df_top10000.h5')
df.to_hdf('data/kaggle_webtraffic/df_top10000.h5', key='webtraffic')