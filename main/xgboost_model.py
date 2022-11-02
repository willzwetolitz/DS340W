import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def generate_xgb_model(params_xgd = {
		'max_depth': 7,
		'objective': 'reg:logistic',
		'learning_rate': 0.05,
		'n_estimators': 10000
	}):
	clf = xgb.XGBRegressor(**params_xgd)
	return clf

def train_xgb_model(train_X, train_Y, clf):
	if type(train_X) != np.ndarray:
		train_X = np.ndarray(train_X)
	if type(train_Y) != np.ndarray:
		train_Y = np.ndarray(train_Y)
	
	clf.fit(train_X,train_Y, eval_metric='rmse', early_stopping_rounds=20, verbose=False)

	return clf

def validate_xgb_model(validate_X, validate_Y, clf):
	predict_Y = clf.predict(validate_X)

	return mean_squared_error(validate_Y, predict_Y)


def main():
	train_X = [1,2,3]
	train_Y = [1,2,3]
	generate_xgb_model()

if __name__ == "__main__":
	main()