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
#%% Training class
class timeseries_dataset():
    def __init__(self, name, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen):
        self.name = name
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        
    def load(self, mode):   
        if self.name == 'uci_electricity':
            output = uci_electricity(self.dim_inputseqlen, self.dim_outputseqlen, self.dim_maxseqlen, mode, self.name)

        if self.name == 'uci_traffic':
            output = uci_traffic(self.dim_inputseqlen, self.dim_outputseqlen, self.dim_maxseqlen, mode, self.name)

        if self.name == 'kaggle_webtraffic':
            output = kaggle_webtraffic(self.dim_inputseqlen, self.dim_outputseqlen, self.dim_maxseqlen, mode, self.name)

        if self.name == 'kaggle_favorita':
            output = kaggle_favorita(self.dim_inputseqlen, self.dim_outputseqlen, self.dim_maxseqlen, mode, self.name)

        if self.name == 'kaggle_m5':
            output = kaggle_m5(self.dim_inputseqlen, self.dim_outputseqlen, self.dim_maxseqlen, mode, self.name)

        return output

#%% UCI - Electricity
# Source: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
class uci_electricity(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen, mode, name):
        """ 
        Load UCI Electricity dataset in format [samples, seqlen, features]
        """        
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode
        self.p_train = 0.8
        self.p_validate = 0.1
        self.name = name
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx, 0], self.index[idx, 1]]
        y = self.Y[self.index[idx, 0], self.index[idx, 1]]
        
        return x, y  

    def get_data(self):
        # Read data from source
        df = pd.read_csv('data/uci_electricity/LD2011_2014.txt', sep = ';', parse_dates=[0], infer_datetime_format=True, dtype='float32', decimal=',', index_col=[0])
        # Subtract 15 minutes from index to make index the starting time (instead of ending time)
        df.index = df.index + pd.Timedelta(minutes=-15)
        # Aggregate to hourly level if desired
        df = df.groupby([df.index.date, df.index.hour]).sum()
        df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])
        # Create index for allowable entries (not only zeros)
        self.num_series = len(df.columns)
        self.num_dates = len(df)
        arr_online = np.zeros((self.num_dates, self.num_series))
        index = np.empty((0, 2), dtype='int')
        for i in range(len(df.columns)):
            idx = np.flatnonzero(df.iloc[:, i])
            arr_online[idx, i] = 1
            idx = np.arange(idx[0] - self.window, np.minimum(idx[-1] + 1, len(df) - self.dim_maxseqlen))
            idx = idx[idx >= 0]
            arr = np.array([np.repeat(i, len(idx)), idx]).T
            index = np.append(index, arr, axis = 0)
        # Stack (and recreate Dataframe because stupid Pandas creates a Series otherwise)
        df = pd.DataFrame(df.stack())
        # Add time-based covariates
        df['Series'] = df.index.get_level_values(2)
        df['Series'] = df['Series'].str.replace(r'MT_','').astype('int') - 1
        # Add scaled time-based features
        df['Month_sin'] = np.sin(df.index.get_level_values(0).month * (2 * np.pi / 12))
        df['Month_cos'] = np.cos(df.index.get_level_values(0).month * (2 * np.pi / 12))    
        df['DayOfWeek_sin'] = np.sin(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['DayOfWeek_cos'] = np.cos(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['HourOfDay_sin'] = np.sin(df.index.get_level_values(1) * (2 * np.pi / 24))
        df['HourOfDay_cos'] = np.cos(df.index.get_level_values(1) * (2 * np.pi / 24))
        df['Online'] = arr_online.reshape(-1, 1)        
        # Rename target column
        df.rename(columns={0:'E_consumption'}, inplace=True)
        # Add lagged output variable
        df['E_consumption_lag'] = df.groupby(level=2)['E_consumption'].shift(1)
        # Remove first date (contains NaNs for the lagged column)
        df = df.iloc[370:]        
        # Sort by series
        df.index.names = ['date','hour','series']
        df.sort_index(level=['series','date','hour'], inplace=True)
        # Create feature matrix X and output vector Y
        df_Y = df[['E_consumption']]
        df.drop(['E_consumption'], axis=1, inplace=True)
        df_X = df
        # Convert dataframe to numpy and reshape to [series x dates x features] format   
        X = df_X.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_X.columns))
        Y = df_Y.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_Y.columns))
        # Input and output dimensions
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        # Convert to torch
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # Create subsequences by unfolding along date dimension with a sliding window
        Xt, Yt = X.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2), Y.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2)
        # Create train, validate and test sets
        num_dates_train = int(self.p_train * Xt.shape[1])
        num_dates_validate = int(self.p_validate * Xt.shape[1])

        index = torch.from_numpy(index)       
        # Get datasets
        if self.mode == 'train':
            self.index = index[index[:, 1] < num_dates_train]
        elif self.mode == 'validate':
            self.index = index[(index[:, 1] >= num_dates_train) & (index[:, 1] < num_dates_train + num_dates_validate)]
        elif self.mode == 'test':
            self.index = index[(index[:, 1] >= num_dates_train + num_dates_validate + self.dim_outputseqlen - 1)]
        
        # Useful for use in algorithms - dimension of lags and dimension of covariates (minus dim of time series ID)
        self.d_lag = self.dim_output
        self.d_emb = 1
        self.d_cov = self.dim_input - self.dim_output - 1
        
        return Xt, Yt[:, :, self.dim_inputseqlen:self.window, :]
#%% UCI - Traffic
# Source: http://archive.ics.uci.edu/ml/datasets/PEMS-SF
class uci_traffic(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen, mode, name):
        """ 
        Load UCI Traffic dataset in format [samples, seqlen, features]
        """        
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode
        self.p_train = 0.8
        self.p_validate = 0.1
        self.name = name
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx, 0], self.index[idx, 1]]
        y = self.Y[self.index[idx, 0], self.index[idx, 1]]
        
        return x, y  

    def get_data(self):
        # Read data from source
        df = pd.read_csv('data/uci_traffic/dataset.csv', index_col=0, infer_datetime_format=True, parse_dates=[[0, 1]])
        # Extract missing column
        missing = df['missing'].copy()
        df = df.drop(columns=['missing'])
        # Create index for allowable entries
        self.num_series = len(df.columns)
        self.num_dates = len(df)
        index_col1 = np.repeat(np.arange(0, self.num_series), self.num_dates - self.dim_maxseqlen)
        index_col2 = np.tile(np.arange(0, self.num_dates - self.dim_maxseqlen), self.num_series)
        index = np.stack((index_col1, index_col2), axis=1)
        # Stack (and recreate Dataframe because stupid Pandas creates a Series otherwise)
        df = pd.DataFrame(df.stack())
        # Add series indicator as integer
        df['Series'] = df.index.get_level_values(1)
        df['Series'] = df['Series'].str.replace(r'carlane_','').astype('int') - 1
        # Add time-based covariates
        df['DayOfWeek_sin'] = np.sin(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['DayOfWeek_cos'] = np.cos(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['HourOfDay_sin'] = np.sin(df.index.get_level_values(0).hour * (2 * np.pi / 24))
        df['HourOfDay_cos'] = np.cos(df.index.get_level_values(0).hour * (2 * np.pi / 24))
        df['Available'] = 1      
        df.loc[(missing[missing == 1].index, slice(None)), 'Available'] = 0
        # Rename target column
        df.rename(columns={0:'Occupancy_rate'}, inplace=True)
        # Add lagged output variable
        df['Occupancy_rate_lag'] = df.groupby(level=1)['Occupancy_rate'].shift(1)
        # Remove first date (contains NaNs for the lagged column)
        df = df.iloc[self.num_series:]        
        # Sort by series
        df.index.names = ['date_time','series']
        df = df.sort_index(level=['series','date_time'])
        # Create feature matrix X and output vector Y
        df_Y = df[['Occupancy_rate']]
        df.drop(['Occupancy_rate'], axis=1, inplace=True)
        df_X = df
        # Convert dataframe to numpy and reshape to [series x dates x features] format   
        X = df_X.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_X.columns))
        Y = df_Y.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_Y.columns))
        # Input and output dimensions
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        # Convert to torch
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # Create subsequences by unfolding along date dimension with a sliding window
        Xt, Yt = X.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2), Y.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2)
        # Create train, validate and test sets
        num_dates_train = int(self.p_train * Xt.shape[1])
        num_dates_validate = int(self.p_validate * Xt.shape[1])

        index = torch.from_numpy(index)       
        # Get datasets
        if self.mode == 'train':
            self.index = index[index[:, 1] < num_dates_train]
        elif self.mode == 'validate':
            self.index = index[(index[:, 1] >= num_dates_train) & (index[:, 1] < num_dates_train + num_dates_validate)]
        elif self.mode == 'test':
            self.index = index[(index[:, 1] >= num_dates_train + num_dates_validate + self.dim_outputseqlen - 1)]
        
        # Useful for use in algorithms - dimension of lags and dimension of covariates (minus dim of time series ID)
        self.d_lag = self.dim_output
        self.d_emb = 1
        self.d_cov = self.dim_input - self.dim_output - 1
        
        return Xt, Yt[:, :, self.dim_inputseqlen:self.window, :]
#%% Kaggle - Webtraffic
# Source: https://www.kaggle.com/c/web-traffic-time-series-forecasting/data
class kaggle_webtraffic(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen, mode, name):
        """ 
        Load Kaggle Web Traffic dataset in format [samples, seqlen, features]
        """        
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode
        self.train_maxdate = '2016-12-31'
        self.validate_maxdate = '2017-03-31'
        self.name = name
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx, 0], self.index[idx, 1]]
        y = self.Y[self.index[idx, 0], self.index[idx, 1]]
        
        return x, y  

    def get_data(self):
        # Read data from source
        df = pd.read_csv('data/kaggle_webtraffic/train_2.csv', index_col=[0]).fillna(0).transpose()
        # Set index
        df.index = pd.to_datetime(df.index)        
        # Training, validation, test
        num_dates_train = len(df.loc[:self.train_maxdate]) - self.dim_maxseqlen
        num_dates_validate = len(df.loc[:self.validate_maxdate]) - self.dim_maxseqlen
        # Use only top-10000
        pageviews = df.loc[:'2016-12-31'].sum(axis=0)
        indices = pageviews.sort_values(ascending=False)[:10000].index
        df = df.loc[:, indices]
        # Rename pages to simple index - we ignore information in the page name
        columns = [str(i) for i in range(len(df.columns))]
        df.columns = columns
        # Series
        self.num_series = len(df.columns)
        self.num_dates = len(df) - 1
        # Create index for allowable entries
        index_col1 = np.repeat(np.arange(0, self.num_series), self.num_dates - self.dim_maxseqlen + 1)
        index_col2 = np.tile(np.arange(0, self.num_dates - self.dim_maxseqlen + 1), self.num_series)
        index = np.stack((index_col1, index_col2), axis=1)
        # Reset index
        
        # Stack (and recreate Dataframe because stupid Pandas creates a Series otherwise)
        df = pd.DataFrame(df.stack())
        # Add series indicator as integer
        df['Series'] = df.index.get_level_values(1)
        df['Series'] = df['Series'].astype('int')
        # Add time-based covariates
        df['DayOfWeek_sin'] = np.sin(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['DayOfWeek_cos'] = np.cos(df.index.get_level_values(0).dayofweek * (2 * np.pi / 7))
        df['DayOfMonth_sin'] = np.sin(df.index.get_level_values(0).day * (2 * np.pi / 31))
        df['DayOfMonth_cos'] = np.cos(df.index.get_level_values(0).day * (2 * np.pi / 31))
        df['WeekOfYear_sin'] = np.sin(df.index.get_level_values(0).week * (2 * np.pi / 53))
        df['WeekOfYear_cos'] = np.cos(df.index.get_level_values(0).week * (2 * np.pi / 53))
        df['MonthOfYear_sin'] = np.sin(df.index.get_level_values(0).month * (2 * np.pi / 12))
        df['MonthOfYear_cos'] = np.cos(df.index.get_level_values(0).month * (2 * np.pi / 12))
        # Rename target column
        df.rename(columns={0:'Page_views'}, inplace=True)
        # Add lagged output variable
        df['Page_views_lag'] = df.groupby(level=1)['Page_views'].shift(1)
        # Remove first date (contains NaNs for the lagged column)
        df = df.iloc[self.num_series:]        
        # Sort by series
        df.index.names = ['date_time','series']
        df = df.sort_index(level=['series','date_time'])
        # Create feature matrix X and output vector Y
        df_Y = df[['Page_views']]
        df.drop(['Page_views'], axis=1, inplace=True)
        df_X = df
        # Convert dataframe to numpy and reshape to [series x dates x features] format   
        X = df_X.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_X.columns))
        Y = df_Y.to_numpy(dtype='float32').reshape(self.num_series, -1, len(df_Y.columns))
        # Input and output dimensions
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        # Convert to torch
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # Create subsequences by unfolding along date dimension with a sliding window
        Xt, Yt = X.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2), Y.unfold(-2, self.dim_maxseqlen, 1).permute(0, 1, 3, 2)

        index = torch.from_numpy(index)       
        # Get datasets
        if self.mode == 'train':
            self.index = index[index[:, 1] < num_dates_train]
        elif self.mode == 'validate':
            self.index = index[(index[:, 1] >= num_dates_train) & (index[:, 1] < num_dates_validate)]
        elif self.mode == 'test':
            self.index = index[(index[:, 1] >= num_dates_validate)]
        
        # Useful for use in algorithms - dimension of lags and dimension of covariates (minus dim of time series ID)
        self.d_lag = self.dim_output
        self.d_emb = 1
        self.d_cov = self.dim_input - self.dim_output - 1
        
        return Xt, Yt[:, :, self.dim_inputseqlen:self.window, :]
#%% Kaggle - Favorita
# Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
class kaggle_favorita(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen, mode, name):
        """ 
        Load Favorita dataset in format [samples, seqlen, features]
        """        
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode
        self.train_maxdate = '2013-09-02'
        self.validate_maxdate = '2013-10-03'
        self.name = name
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx]:self.index[idx] + self.dim_maxseqlen]
        y = self.Y[self.index[idx] + self.dim_inputseqlen:self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen]
        
        return x, y  

    def get_data(self):
        # Read data from source
        df = pd.read_hdf('data/kaggle_favorita/df_full_2013.h5', key='favorita')
        index = pd.read_hdf('data/kaggle_favorita/df_full_2013.h5', key='index')        
        df_Y = df[['unit_sales']]
        df_X = df.drop(columns=['unit_sales', 'transactions_missing_lagged', 'oil_price_missing_lagged', 'unit_sales_missing_lagged', 'transactions_lagged', 'oil_price_lagged'])
        # Convert dataframe to numpy and reshape to [series x dates x features] format   
        X = df_X.to_numpy(dtype='float32')
        Y = df_Y.to_numpy(dtype='float32')
        # Input and output dimensions
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        # Convert to torch
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # Get datasets
        if self.mode == 'train':
            idx = index[index['date'] <= self.train_maxdate]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'validate':
            idx = index[(index['date'] <= self.validate_maxdate) & (index['date'] > self.train_maxdate)]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        elif self.mode == 'test':
            idx = index[index['date'] > self.validate_maxdate]['index'].to_numpy()
            self.index = torch.from_numpy(idx)
        # Useful for use in algorithms - dimension of lags and dimension of covariates (minus dim of time series ID)
        self.d_lag = 1 # number of lags in input
        self.d_emb = 2 # number of embedding categoricals in input
        self.d_cov = self.dim_input - self.d_lag - self.d_emb  # number of covariates input
        
        return X, Y
#%% Kaggle - M5
# Source: https://www.kaggle.com/c/m5-forecasting-accuracy
class kaggle_m5(torchdata.Dataset):
    def __init__(self, dim_inputseqlen, dim_outputseqlen, dim_maxseqlen, mode, name):
        """ 
        Load m5 dataset in format [samples, seqlen, features]
        """        
        self.dim_inputseqlen = dim_inputseqlen
        self.dim_outputseqlen = dim_outputseqlen
        self.window = dim_inputseqlen + dim_outputseqlen
        self.dim_maxseqlen = dim_maxseqlen
        self.mode = mode
        self.train_mindate = '2014-01-01'
        self.train_maxdate = '2015-12-31'
        self.validate_maxdate = '2016-01-31'
        self.name = name
        self.X, self.Y = self.get_data()
        
    def __len__(self):
        return len(self.index)
        
    def __getitem__(self, idx):        
        x = self.X[self.index[idx]:self.index[idx] + self.dim_maxseqlen]
        y = self.Y[(self.index[idx] + self.dim_inputseqlen):(self.index[idx] + self.dim_outputseqlen + self.dim_inputseqlen)]
        
        return x, y  

    def get_data(self):
        # Read data from source
        df = pd.read_feather('data/kaggle_m5/m5_dataset_products.feather')
        # Build index
        idx = df.groupby(['id_enc'])['sales'].rolling(self.dim_maxseqlen).count()
        idx = idx[idx == self.dim_maxseqlen]
        indices = (idx.index.get_level_values(1) - self.dim_maxseqlen + 1)
        dates = df[['id_enc','date']].loc[indices]
        df_index = dates.reset_index()
        # Create X and Y
        df_Y = df[['sales']]
        df_X = df.drop(columns=['id_enc','date','sales'])
        # Convert dataframe to numpy    
        X = df_X.to_numpy(dtype='float32')
        Y = df_Y.to_numpy(dtype='float32')
        # Input and output dimensions
        self.dim_input, self.dim_output = X.shape[-1], Y.shape[-1]
        # Convert to torch
        X, Y = torch.from_numpy(X), torch.from_numpy(Y)
        # Get datasets
        if self.mode == 'train':
            self.df_index = df_index[(df_index['date'] <= self.train_maxdate) & (df_index['date'] >= self.train_mindate)]
        elif self.mode == 'validate':
            self.df_index = df_index[(df_index['date'] <= self.validate_maxdate) & (df_index['date'] > self.train_maxdate)]
        elif self.mode == 'test':
            self.df_index = df_index[df_index['date'] > self.validate_maxdate]
        # Create index and sample weights
        self.index = torch.from_numpy(self.df_index['index'].to_numpy())
        # Useful for use in algorithms - dimension of lags and dimension of covariates
        self.d_lag = 1 # number of lags unknown in the future in input (sales_lag, sales_total_lag). E.g. note that sales_lag_365 is always available to the learner given we only have an max_inputseqlen of 118
        self.d_emb = 7 # number of embedding categoricals in input
        self.d_cov = self.dim_input - self.d_lag - self.d_emb  # number of covariates input

        return X, Y