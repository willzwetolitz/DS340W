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
import fix_yahoo_finance as yf
import os
from sklearn.ensemble import *

sns.set()
import functions

# Will need to iterate through all tickers and combine csvs into one dataframe instead of just JPM

functions.data_to_csv("JPM", "data_raw")
jpm_train, jpm_test = functions.get_ticker_data("JPM", "data_raw")
jpm_close_normalize_train, jpm_close_normalize_test = functions.normalize_ticker(jpm_train, jpm_test)
jpm_thought_vector_train, jpm_thought_vector_test = functions.encode(jpm_close_normalize_train, jpm_close_normalize_test)
ada, bagging, et, gb, rf = functions.fit_l1(jpm_thought_vector_train, jpm_close_normalize_train)
preds = functions.make_preds_l1((ada, bagging, et, gb, rf), jpm_thought_vector_train)
ada_pred, bagging_pred, et_pred, gb_pred, rf_pred = preds
stack_predict, actuals = functions.get_actual(jpm_close_normalize_train, preds)
l2_model = functions.train_l2(stack_predict, jpm_close_normalize_train)


date_ori = pd.to_datetime(jpm_train.iloc[:, 0]).tolist()
date_ori_test = pd.to_datetime(jpm_test.iloc[:, 0]).tolist()
xgb_pred = l2_model.predict(stack_predict)
xgb_actual = np.hstack([jpm_close_normalize_train[0],xgb_pred[:-1]])
date_original=pd.Series(date_ori).dt.strftime(date_format='%Y-%m-%d').tolist()
