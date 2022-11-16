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
import xgboost as xgb

sns.set()


### Creating features through an encoder
class encoder:
    def __init__(self, input_, dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20):
        input_size = input_.shape[1]
        self.X = tf.compat.v1.placeholder("float", [None, input_.shape[1]])

        weights = {
            'encoder_h1': tf.Variable(tf.compat.v1.random_normal([input_size, hidden_layer])),
            'encoder_h2': tf.Variable(tf.compat.v1.random_normal([hidden_layer, dimension])),
            'decoder_h1': tf.Variable(tf.compat.v1.random_normal([dimension, hidden_layer])),
            'decoder_h2': tf.Variable(tf.compat.v1.random_normal([hidden_layer, input_size])),
        }

        biases = {
            'encoder_b1': tf.Variable(tf.compat.v1.random_normal([hidden_layer])),
            'encoder_b2': tf.Variable(tf.compat.v1.random_normal([dimension])),
            'decoder_b1': tf.Variable(tf.compat.v1.random_normal([hidden_layer])),
            'decoder_b2': tf.Variable(tf.compat.v1.random_normal([input_size])),
        }

        first_layer_encoder = tf.nn.sigmoid(tf.add(tf.matmul(self.X, weights['encoder_h1']), biases['encoder_b1']))
        self.second_layer_encoder = tf.nn.sigmoid(
            tf.add(tf.matmul(first_layer_encoder, weights['encoder_h2']), biases['encoder_b2']))
        first_layer_decoder = tf.nn.sigmoid(
            tf.add(tf.matmul(self.second_layer_encoder, weights['decoder_h1']), biases['decoder_b1']))
        second_layer_decoder = tf.nn.sigmoid(
            tf.add(tf.matmul(first_layer_decoder, weights['decoder_h2']), biases['decoder_b2']))
        self.cost = tf.reduce_mean(tf.pow(self.X - second_layer_decoder, 2))
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(self.cost)
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(epoch):
            last_time = time.time()
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_})
            if (i + 1) % 10 == 0:
                print('epoch:', i + 1, 'loss:', loss, 'time:', time.time() - last_time)

    def encode(self, input_):
        return self.sess.run(self.second_layer_encoder, feed_dict={self.X: input_})


def data_to_csv(ticker, dest):
    # Takes in a ticker as a string, outputs a csv to the given destination path
    yf.pdr_override(tickers=ticker, period = '1d', start='2012-01-01', end='2017-12-31')
    df_full = pdr.get_data_yahoo(ticker, start="2012-01-01", end='2017-12-31').reset_index()

    yf.pdr_override(tickers=ticker, period = '1d', start='2018-01-01', end='2019-12-31')
    df_full_test = pdr.get_data_yahoo(ticker, start="2018-01-01", end='2019-12-31').reset_index()

    df_full.to_csv(os.path.join(dest, f'{ticker}_2012_2018.csv'),index=False)
    df_full_test.to_csv(os.path.join(dest, f'{ticker}_2018_2020.csv'), index=False)

def get_ticker_data(ticker, folder):
    train = pd.read_csv(os.path.join(folder, f'{ticker}_2012_2018.csv'))
    test = pd.read_csv(os.path.join(folder, f'{ticker}_2018_2020.csv'))
    return train, test

def normalize_ticker(train, test):
    minmax_train = MinMaxScaler().fit(train.loc[:,'Adj Close'].values.reshape((-1,1)))
    close_normalize_train = minmax_train.transform(train.loc[:,"Adj Close"].values.reshape((-1,1))).reshape((-1))
    minmax_test = MinMaxScaler().fit(test.loc[:,'Adj Close'].values.reshape((-1,1)))
    close_normalize_test = minmax_test.transform(test.loc[:, "Adj Close"].values.reshape((-1,1))).reshape((-1))
    return close_normalize_train, close_normalize_test



def reverse_close(scaler, array):
    # Function that generates close price using normalized close.
    # Takes in a MinMaxScaler and array of predictions, returns
    return scaler.inverse_transform(array.reshape((-1,1))).reshape((-1))

def encode(close_normalize_train, close_normalize_test):
    # Fit the encoder to our train data and generate a thought vector for the train data
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    Encoder = encoder(close_normalize_train.reshape((-1, 1)), 32, 0.01, 128, 100)
    thought_vector_train = Encoder.encode(close_normalize_train.reshape((-1, 1)))
    thought_vector_test = Encoder.encode(close_normalize_test.reshape((-1, 1)))
    return thought_vector_train, thought_vector_test

def fit_l1(thought_vector_train, close_normalize_train):
    # Takes in a thought_vector (encoded on close_normalize_train) and normalized close for train set.
    # Returns a list of fit models
    ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
    bagging = BaggingRegressor(n_estimators=500)
    et = ExtraTreesRegressor(n_estimators=500)
    gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
    rf = RandomForestRegressor(n_estimators=500)
    ada.fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    bagging.fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    et.fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    gb.fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    rf.fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    return (ada, bagging, et, gb, rf)

def make_preds_l1(models, thought_vector):
    ada, bagging, et, gb, rf = models
    ada_pred = ada.predict(thought_vector)
    bagging_pred = bagging.predict(thought_vector)
    et_pred = et.predict(thought_vector)
    gb_pred = gb.predict(thought_vector)
    rf_pred = rf.predict(thought_vector)
    return (ada_pred, bagging_pred, et_pred, gb_pred, rf_pred)


def get_actual(close_normalize, preds):
    ada_pred, bagging_pred, et_pred, gb_pred, rf_pred = preds
    ada_actual = np.hstack([close_normalize[0], ada_pred[:-1]])
    bagging_actual = np.hstack([close_normalize[0], bagging_pred[:-1]])
    et_actual = np.hstack([close_normalize[0], et_pred[:-1]])
    gb_actual = np.hstack([close_normalize[0], gb_pred[:-1]])
    rf_actual = np.hstack([close_normalize[0], rf_pred[:-1]])
    stack_predict = np.vstack([ada_actual, bagging_actual, et_actual, gb_actual, rf_actual]).T
    actuals = (ada_actual, bagging_actual, et_actual, gb_actual, rf_actual)
    return stack_predict, actuals

def train_l2(stack_predict, close_normalize_train):
    # Takes in train close_normalize and predictions from level one models
    # Returns a trained level 2 XGBoost model

    params_xgd = {
        'max_depth': 7,
        'objective': 'reg:logistic',
        'learning_rate': 0.05,
        'n_estimators': 10000
    }
    train_y = close_normalize_train[1:]
    clf = xgb.XGBRegressor(**params_xgd)
    clf.fit(stack_predict[:-1, :], train_y, eval_set=[(stack_predict[:-1, :], train_y)],
            eval_metric='rmse', early_stopping_rounds=20, verbose=False)
    return clf

