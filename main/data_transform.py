import pandas as pd
import numpy as np
import os

def do_transform(train_data):
    history = 180
    bins = [180, 90, 30, 7, 3]
    res = []
    for i in range(180, train_data.shape[0]-1):
        train_temp = train_data[['Close', 'Volume']][i-history:i]
        close = train_temp[['Close']]/train_temp[['Close']].values[-1]
        vol = train_temp[['Volume']]/train_temp[['Volume']].values[-1]
        diff_close = [close.values[-j][0]-close.values[-1][0] for j in bins]
        stdev_close = [close[-j:].std().values[0] for j in bins]
        avg_vol = [vol[-j:].mean().values[0] for j in bins]
        close_change_tomorrow = [train_data[['Close']].values[i+1][0]/train_temp[['Close']].values[-1][0] - train_data[['Close']].values[i][0]/train_temp[['Close']].values[-1][0]]

        total_row = diff_close + stdev_close + avg_vol + close_change_tomorrow
        res.append(total_row)
    return pd.DataFrame(res, columns=['diff_close_180', 'diff_close_90', 'diff_close_30', 'diff_close_7', 'diff_close_3', 'std_close_180', 'std_close_90', 'std_close_30', 'std_close_7', 'std_close_3', 'avg_vol_180', 'avg_vol_90', 'avg_vol_30', 'avg_vol_7', 'avg_vol_3', 'target'])


def transform_file(fname: str):
    # Read given file
    # Do transformations
    # Add in timestamp and stock columns
    stock = fname.split('/')[2].split('_')[0]
    train_data = pd.read_csv(fname)
    transformed_data = do_transform(train_data)
    timestamps = np.array(train_data[['timestamp']][180:train_data.shape[0]-1])
    transformed_data.insert(0, 'timestamp', timestamps)
    transformed_data.insert(0, 'stock', stock)
    return transformed_data

def clean_files(fnames: list):
    # Iterate through given file names
    # Get cleaned dataframe for each file
    # Return concatenated dataframes
    df_list = []
    for fname in fnames:
        df_list.append(transform_data(fname))
    df_all = pd.concat(df_list)
    return df_all

def main():
	# We have 60 CSV files
	#    30 tickers, 1 file containing 2012 to 2018, 1 file containing 2018 to 2020
	raw_data_dir = 'data_raw'
	all_train_data = [pd.read_csv(f'{raw_data_dir}/{i}') for i in os.listdir(raw_data_dir) if '2012' in i]
	all_test_data = [pd.read_csv(f'{raw_data_dir}/{i}') for i in os.listdir(raw_data_dir) if '2020' in i]

	


