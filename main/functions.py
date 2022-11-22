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
from sklearn.model_selection import KFold
from pandas_datareader import data as pdr
import yfinance as yf
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


# Fit the encoder to our train data and generate a thought vector for the train data
def encode(df_train, df_test):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    Encoder = encoder(df_train.reshape((-1, 1)), 32, 0.01, 128, 100)
    thought_vector_train = Encoder.encode(df_train.reshape((-1, 1)))
    thought_vector_test = Encoder.encode(df_test.reshape((-1, 1)))
    return thought_vector_train, thought_vector_test

# Takes in a ticker as a string, outputs a csv to the given destination path
def data_to_csv(ticker, dest):
    yf.pdr_override()
    df_full = pdr.get_data_yahoo(ticker, start="2012-01-01", end='2017-12-31').reset_index()

    yf.pdr_override()
    df_full_test = pdr.get_data_yahoo(ticker, start="2018-01-01", end='2019-12-31').reset_index()

    df_full.to_csv(os.path.join(dest, f'{ticker}_2012_2018.csv'),index=False)
    df_full_test.to_csv(os.path.join(dest, f'{ticker}_2018_2020.csv'), index=False)

def get_ticker_data(ticker, folder):
    train = pd.read_csv(os.path.join(folder, f'{ticker}_2012_2018.csv'))
    test = pd.read_csv(os.path.join(folder, f'{ticker}_2018_2020.csv'))
    return train, test

def get_normalized(df):
    minmax = MinMaxScaler().fit(df.loc[:,'Adj Close'].values.reshape((-1,1)))
    df_norm = minmax.transform(df.loc[:,'Adj Close'].values.reshape((-1,1))).reshape((-1))
    return df_norm, minmax

# Function that generates close price using normalized close.
# Takes in a MinMaxScaler and array of predictions, returns
def reverse_close(scaler, array):
    return scaler.inverse_transform(array.reshape((-1,1))).reshape((-1))

def init_l1(n=500, l=0.1):
    ada = AdaBoostRegressor(n_estimators=n, learning_rate=l)
    bagging = BaggingRegressor(n_estimators=n)
    et = ExtraTreesRegressor(n_estimators=n)
    gb = GradientBoostingRegressor(n_estimators=n, learning_rate=l)
    rf = RandomForestRegressor(n_estimators=n)
    return ada, bagging, et, gb, rf

# Create prediction matrix using k-folding
# Level 1 models are trained on k-1 folds of the train data then predict on the kth fold
# 
def make_l2_training_matrix(models, thought_vector, close_normalize, k=10):
    trimmed_tv = thought_vector[:-1,:]

    preds = [np.array([]) for i in range(len(models))]
    actual = np.array([])
    next_day = np.array([])

    k_i = 0
    for train_i, test_i in KFold(k, shuffle=True).split(trimmed_tv):
        k_i += 1
        for i in range(len(models)):
            print(f'\tProcessing fold {k_i} . . . \tModel {i+1}/{len(models)}', end='\r')
            models[i].fit(trimmed_tv[train_i], close_normalize[train_i+1])
            preds[i] = np.hstack([preds[i], models[i].predict(trimmed_tv[test_i])])

        actual = np.hstack([actual, close_normalize[test_i]])
        next_day = np.hstack([next_day, close_normalize[test_i+1]])
        print()
    print('\nDone!')

    stacked_pred = np.vstack([pred for pred in preds]).T
    train_Y = next_day.T
    return stacked_pred, train_Y

# Takes in a thought_vector (encoded on close_normalize_train) and normalized close for train set.
# Returns a list of fit models
def fit_l1(models, thought_vector_train, close_normalize_train):
    for i in range(len(models)):
        models[i].fit(thought_vector_train[:-1, :], close_normalize_train[1:])
    return models

# Make a prediction matrix using a list of models and test data
def predict(models, thought_vector):
    preds = []
    for model in models:
        preds.append(model.predict(thought_vector))
    return preds

# Staggers predictions by 1 and stacks predictions
def get_stacked_predictions(close_normalize, preds):
    for i in range(len(preds)):
        preds[i] = np.hstack([close_normalize[0], preds[i][:-1]])
    return np.vstack([actual for actual in preds]).T

def train_l2(stack_predict, train_y):
    # Takes in train close_normalize and predictions from level one models
    # Returns a trained level 2 XGBoost model
    params_xgd = {
        'max_depth': 7,
        'objective': 'reg:logistic',
        'eval_metric':'rmse', 
        'early_stopping_rounds':20,
        'learning_rate': 0.05,
        'n_estimators': 10000
    }
    clf = xgb.XGBRegressor(**params_xgd)
    clf.fit(stack_predict, train_y, eval_set=[(stack_predict, train_y)], verbose=False)
    return clf

# Calculate the percentage of the time that the model predicted an increase or decrease in the closing price correctly
def get_trend_accuracy(pred, actual):
    correct = 0
    last_actual, last_predict = 0, 0
    size = min(len(pred),len(actual))
    for i in range(size):
        if not last_actual and not last_predict:
            last_actual = float(actual[i])
            last_predict = float(pred[1])
            continue
        if last_actual < float(actual[i])*1.001:
            if last_predict < float(pred[i])*1.001:
                correct += 1
        elif float(actual[i])*1.001 < last_actual:
            if float(pred[i])*1.001 < last_predict:
                correct += 1
        last_actual = float(actual[i])
        last_predict = float(pred[i])

    return correct/size

# Create a pandas DataFrame from a list of lists/np.Arrays
# Saves countless syntax problems
def make_dataframe(lsts, cols):
    if len(lsts) != len(cols):
        raise Exception(f'{len(lsts)=} does not equal {len(cols)=}')
    for i in range(len(lsts)):
        if type(lsts[i]) != np.array:
            lsts[i] = np.array(lsts[i])
    return pd.DataFrame(np.vstack([lst.T for lst in lsts]).T, columns = cols)

'''
df_full = pd.read_csv("JPM_normalized.csv")
plt.figure(figsize = (8,5))
print(df_full.head())
x_range = np.arange(df_full.loc[:,'True'].shape[0])
plt.plot(x_range, df_full.loc[:,'True'], label = 'Actual Close')
plt.plot(x_range, df_full.loc[:,'Mod_pred'], label = 'Modified Model Predicted Close')
plt.legend()
plt.xticks(x_range[::150], df_full.loc[:,'Date'][::150])
plt.title('Predicted Adj. Close (Modified stacked learning) vs. Actual Close Price [JPM]')
plt.show()
'''

# Subtract actual today from predicted tomorrow to get change prediction
# Base trader on change prediction (buy 2 if up 5 or more, buy 1 if up 0-5, sell 2 if down 5 or more, sell 2 if down 0-5)
# Calculate profit/loss from prediction



def basic_strategy(df, cash):
    df['diff'] = df['Mod_pred'].sub(df['True'].shift(1))
    true_col = df['True']
    diff_col = df['diff']
    stock = 0
    for i in range(1, len(true_col)):
        if diff_col[i] > 0 and cash > true_col[i]:
            stock += 1
            cash -= true_col[i]
        elif diff_col[i] < 0 and stock > 0:
            stock -= 1
            cash += true_col[i]
    return stock*true_col[len(true_col)-1] + cash

def double_strategy(df, cash):
    df['diff'] = df['Mod_pred'].sub(df['True'].shift(1))
    true_col = df['True']
    diff_col = df['diff']
    stock = 0
    for i in range(1, len(true_col)):
        if diff_col[i] > 0 and cash > true_col[i]:
            if diff_col[i] > 0.5 and cash > 2*true_col[i]:
                stock += 2
                cash -= 2*true_col[i]
            else:
                stock += 1
                cash -= true_col[i]
        elif diff_col[i] < 0 and stock > 0:
            if diff_col[i] < -0.5 and stock > 1:
                stock -= 2
                cash += 2*true_col[i]
            else:
                stock -= 1
                cash += true_col[i]
    return stock*true_col[len(true_col)-1] + cash

def proportional_strategy(df, cash, per, limit):
    df['diff'] = df['Mod_pred'].sub(df['True'].shift(1))
    true_col = df['True']
    diff_col = df['diff']
    stock = 0
    for i in range(1, len(true_col)):
        if diff_col[i] > 0 and cash > true_col[i]:
            # buy if possible
            buy_count = min(diff_col[i]//per + 1, limit)
            if buy_count*true_col[i] > cash:
                afford_count = cash//true_col[i]
                stock += afford_count
                cash -= afford_count*true_col[i]
            else:
                stock += buy_count
                cash -= buy_count*true_col[i]
        elif diff_col[i] < 0 and stock > 0:
            # sell if possible
            sell_count = min(-diff_col[i]//per + 1, limit)
            if sell_count < stock:
                cash += stock * true_col[i]
                stock = 0
            else:
                stock -= sell_count
                cash += sell_count*true_col[i]
    return stock*true_col[len(true_col)-1] + cash