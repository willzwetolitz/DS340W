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
        close_change_tomorrow = [np.array(train_temp[['Close']].values[i+1][0]/train_temp[['Close']].values[-1][0] - train_temp[['Close']].values[i][0]/train_temp[['Close']].values[-1][0])*100]

        total_row = diff_close + stdev_close + avg_vol + close_change_tomorrow
        res.append(total_row)
    return pd.DataFrame(res, columns=['diff_close_180', 'diff_close_90', 'diff_close_30', 'diff_close_7', 'diff_close_3', 'std_close_180', 'std_close_90', 'std_close_30', 'std_close_7', 'std_close_3', 'avg_vol_180', 'avg_vol_90', 'avg_vol_30', 'avg_vol_7', 'avg_vol_3', 'target'])

def get_lobf(ys: pd.DataFrame):
    xs = np.arange(ys.shape[0])
    m = (xs.mean()*ys.mean() - (xs*ys).mean())/(xs.mean()*xs.mean() - (xs*xs).mean())
    b = ys.mean() - m*xs.mean()
    reg_line = [m*x+b for x in xs]
    y_bar = ys.mean()
    r2 = 1 - sum((ys[i]-reg_line[i])**2 for i in range(len(ys)))/sum((y-y_bar)**2 for y in ys)
    return m, r2 if not np.isnan(r2) else 0

def do_transform_2(train_data):
    history = 180
    bins = [180, 90, 30, 7, 3]
    res = []
    for i in range(history, train_data.shape[0]-1):
        train_temp = train_data[['Close', 'Volume']][i-history:i]
        close = train_temp[['Close']]/train_temp[['Close']].values[-1]
        vol = train_temp[['Volume']]/train_temp[['Volume']].values[-1]
        close_slope_and_r2 = [get_lobf(close.Close.values[-j:]) for j in bins]        
        close_slope = [i[0] for i in close_slope_and_r2]
        close_r2 = [i[1] for i in close_slope_and_r2]
        vol_slope_and_r2 = [get_lobf(vol.Volume.values[-j:]) for j in bins]
        vol_slope = [i[0] for i in vol_slope_and_r2]
        vol_r2 = [i[1] for i in vol_slope_and_r2]
        close_change_tomorrow = [np.array(train_temp[['Close']].values[i+1][0]/train_temp[['Close']].values[-1][0] - train_temp[['Close']].values[i][0]/train_temp[['Close']].values[-1][0])*100]
        total_row = close_slope + close_r2 + vol_slope + vol_r2 + close_change_tomorrow
        res.append(total_row)

    return pd.DataFrame(res, columns = [f'{j}-{i}' for j in ['close-slope', 'close-r2', 'vol-slope', 'vol-r2'] for i in bins] + ['close_tomorrow'])

def do_transform_3(train_data,
    history = 180,
    long = 180,
    short = 3,
    short_target = 1):
    res = []

    for i in range(180, train_data.shape[0]-max(short, short_target)):
        train_temp = train_data[['Close', 'Volume']][i-history:i]
        close = train_temp.Close/train_temp.Close.values[-1]

        close_slope_long = get_lobf(close.values[-long:])[0]
        close_slope_short = get_lobf(close.values[-short:])[0]
        vol_avg_long = train_temp.Volume.mean()
        vol_avg_short = train_temp.Volume.mean()

        # Flag to see if current close slope is greater than 180
        close_slope_diff_indicator = 1 if close_slope_short > close_slope_long*1.5 else -1 if close_slope_long > close_slope_short*1.5 else 0
        close_slope_abs_indicator = 1 if close_slope_short > 0 else -1 if 0 > close_slope_short else 0
        vol_diff_indicator = 1 if vol_avg_short > vol_avg_long else -1 if vol_avg_long > vol_avg_short else 0

        target = train_data.Close[i]/train_data.Close[i-1] - 1
        target_2 = train_data.Close.loc[i:i+short_target].mean()/train_data.Close[i-1] - 1

        row = [close_slope_long, close_slope_short, close_slope_diff_indicator, close_slope_abs_indicator, vol_diff_indicator, target, target_2]
        
        res.append(row)

    return pd.DataFrame(res, columns = ['close_slope_long', 'close_slope_short', 'close_slope_diff_indicator', 'close_slope_abs_indicator', 'vol_diff_indicator', 'target', 'target_2'])


def transform_file(fname: str):
    # Read given file
    # Do transformations
    # Add in timestamp and stock columns
    stock = fname[fname.rindex('/')+1:].split('_')[0]
    train_data = pd.read_csv(fname)
    train_data = train_data[train_data.Volume != 0]
    transformed_data = do_transform(train_data)

    # history = 180
    # long = 180
    # short = 3
    # short_target = 3
    # transformed_data = do_transform_3(train_data, history, long, short, short_target)

    abs_close = np.array(train_data[['Close']][180:train_data.shape[0]])
    abs_target = np.array(train_data[['Close']][181:train_data.shape[0]+1])
    timestamps = np.array(train_data[['Timestamp']][180:train_data.shape[0]])
    transformed_data.insert(0,'True_close', abs_close)
    transformed_data.insert(0,'True_target', abs_target)
    transformed_data.insert(0, 'Timestamp', timestamps)
    transformed_data.insert(0, 'stock', stock)
    return transformed_data

def clean_files(fnames: list):
    # Iterate through given file names
    # Get cleaned dataframe for each file
    # Return concatenated dataframes
    df_list = []
    for i, fname in enumerate(fnames):
        print(f'{i+1}/{len(fnames)}\r', end='')
        df_list.append(transform_file(fname))
    print()
    df_all = pd.concat(df_list)
    return df_all

def save_data(df: pd.DataFrame, fname: str):
    df.to_csv(fname)

def main():
	# We have 60 CSV files
	#    30 tickers, 1 file containing 2012 to 2018, 1 file containing 2018 to 2020

    raw_data_dir = 'data_raw'
    output_dir = 'data_final'

    train_fnames = [f'{raw_data_dir}/{i}' for i in os.listdir(raw_data_dir) if '2012' in i]
    test_fnames = [f'{raw_data_dir}/{i}' for i in os.listdir(raw_data_dir) if '2020' in i]

    print('train_data')
    train_df = clean_files(train_fnames)
    print('test_data')
    test_df = clean_files(test_fnames)

    print('Saving train_data')
    save_data(train_df, f'{output_dir}/train_data.csv')
    print('Saving test_data')
    save_data(test_df, f'{output_dir}/test_data.csv')

