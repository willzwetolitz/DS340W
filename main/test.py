# Pull data from yfinance and save it into memory and disk
import pandas as pd
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import yfinance as yf
import os
from sklearn.ensemble import *

sns.set()
import functions
import sys

# Will need to iterate through all tickers and combine csvs into one dataframe instead of just JPM

def run_modified_code_1(data_package, loc):

	train_tv, train_norm, scaler_train, train = data_package['train']
	test_tv, test_norm, scaler_test, test = data_package['test']

	# Initialization and training of models
	print('Initializing Level 1 models . . .', end='\t')
	ada, bagging, et, gb, rf = functions.init_l1()
	models = (ada, bagging, et, gb, rf)
	print('Done')
	print('Making K-Folded Level 1 train predictions . . .', end='\n')
	l2_training_matrix, train_y = functions.make_l2_training_matrix(models, train_tv, train_norm)
	print('Done')
	print('Training Level 2 model . . .', end='\t')
	l2_model = functions.train_l2(l2_training_matrix, train_y)
	print('Done')
	print('Final fitting Level 1 models . . .', end='\t')
	models = functions.fit_l1(models, train_tv, train_norm)
	print('Done')

	# Making predictions on validation data
	print('Making Level 1 validation predictions . . .', end='\t')
	raw_predictions = functions.predict(models, test_tv)
	l2_test_matrix = functions.get_stacked_predictions(test_norm, raw_predictions)
	print('Done')
	print('Making Level 2 validation predictions . . .', end='\t')
	l2_pred = functions.predict([l2_model], l2_test_matrix)[0]
	l2_pred = np.hstack([test_norm[0],l2_pred[:-1]])
	print('Done')
	l2_final_pred = functions.reverse_close(scaler_test, l2_pred)
	date_original = pd.to_datetime(test.loc[:,'Date']).tolist()
	date_original=pd.Series(date_original).dt.strftime(date_format='%Y-%m-%d').tolist()

	# Save and plot predictions
	print('Saving results . . .',end='\t')
	output = functions.make_dataframe([date_original, l2_final_pred, test.loc[:,'Adj Close']], cols=['Date', 'Pred', 'True'])
	output.to_csv('results/' + loc)
	print('Done')

	acc = l2_model.score(l2_test_matrix, test_norm)
	trend_acc = functions.get_trend_accuracy(l2_pred, test_norm)
	print(f'Accuracy Score: {acc}')
	print(f'Trend Accuracy: {trend_acc}')

	saveit = 'mod_1_accuracy.csv'
	if saveit:
		path = 'results/' + saveit
		if not os.path.exists(path):
			with open(path, 'w+') as f:
				f.write('Accuracy,Trend Accuracy\n')
		with open(path, 'a') as f:
			f.write(f'{acc},{trend_acc}\n')

def run_parent_code(data_package, loc):

	train_tv, train_norm, scaler_train, train = data_package['train']
	test_tv, test_norm, scaler_test, test = data_package['test']

	# Initialization and training of models
	print('Initializing Level 1 models . . .', end='\t')
	ada, bagging, et, gb, rf = functions.init_l1()
	models = (ada, bagging, et, gb, rf)
	print('Done')
	print('Fitting Level 1 models . . .', end='\t')
	models = functions.fit_l1(models, train_tv, train_norm)
	print('Done')
	print('Making Level 1 train predictions . . .', end='\t')
	raw_predictions = functions.predict(models, train_tv)
	l2_training_matrix = functions.get_stacked_predictions(train_norm, raw_predictions)
	print('Done')
	print('Training Level 2 model . . .', end='\t')
	l2_model = functions.train_l2(l2_training_matrix[:-1], train_norm[1:])
	print('Done')

	# Making predictions on validation data
	print('Making Level 1 validation predictions . . .', end='\t')
	raw_predictions = functions.predict(models, test_tv)
	l2_test_matrix = functions.get_stacked_predictions(test_norm, raw_predictions)
	print('Done')
	print('Making Level 2 validation predictions . . .', end='\t')
	l2_pred = functions.predict([l2_model], l2_test_matrix)[0]
	l2_pred = np.hstack([test_norm[0],l2_pred[:-1]])
	print('Done')
	l2_final_pred = functions.reverse_close(scaler_test, l2_pred)
	date_original = pd.to_datetime(test.loc[:, 'Date']).tolist()
	date_original=pd.Series(date_original).dt.strftime(date_format='%Y-%m-%d').tolist()

	# Save and plot predictions
	print('Saving results . . .',end='\t')
	output = functions.make_dataframe([date_original, l2_final_pred, test.loc[:,'Adj Close']], cols=['Date', 'Pred', 'True'])
	output.to_csv('results/' + loc)
	print('Done')

	acc = l2_model.score(l2_test_matrix, test_norm)
	trend_acc = functions.get_trend_accuracy(l2_pred[:-1], test_norm[1:])
	print(f'Accuracy Score: {acc}')
	print(f'Trend Accuracy: {trend_acc}')

	saveit = 'parent_accuracy.csv'
	if saveit:
		path = 'results/' + saveit
		if not os.path.exists(path):
			with open(path, 'w+') as f:
				f.write('Accuracy,Trend Accuracy\n')
		with open(path, 'a') as f:
			f.write(f'{acc},{trend_acc}\n')

def get_data():
	# Data preparation and encoding
	print('Loading data . . . ', end='\t')
	train, test = functions.get_ticker_data('JPM', 'data_raw')
	print('Done')
	print('Normalizing data . . . ', end='\t')
	train_norm, scaler_train = functions.get_normalized(train)
	test_norm, scaler_test = functions.get_normalized(test)
	print('Done')
	print('Encoding data . . . ', end='\t')
	train_tv, test_tv = functions.encode(train_norm, test_norm) # tv is short for Thought Vector
	print('Done')

	data = {'train': [train_tv, train_norm, scaler_train, train],
			'test': [test_tv, test_norm, scaler_test, test]}
	return data

def main(args=['data']):
	if 'data' not in args:
		print('Pulling data . . .', end='\t')
		functions.data_to_csv("JPM", "data_raw")
		print('Done')

	data_package = get_data()
	run_modified_code_1(data_package, f'mod_1_output.csv')

	# n = 10
	# for i in range(n):
	# 	print(f'Running parent code {i}/{n}')
	# 	data_package = get_data()
	# 	run_parent_code(data_package, f'parent_paper_output_{i}.csv')

if __name__ == '__main__':
	main()
