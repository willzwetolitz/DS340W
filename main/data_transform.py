import pandas as pd
import numpy as np
import os

def main():
	# We have 60 CSV files
	#    30 tickers, 1 file containing 2012 to 2018, 1 file containing 2018 to 2020
	raw_data_dir = 'data_raw'
	all_train_data = [pd.read_csv(f'{raw_data_dir}/{i}') for i in os.listdir(raw_data_dir) if '2012' in i]
	all_test_data = [pd.read_csv(f'{raw_data_dir}/{i}') for i in os.listdir(raw_data_dir) if '2020' in i]

	


