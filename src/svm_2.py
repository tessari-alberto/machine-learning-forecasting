import pandas as pd
import numpy as np
import datetime

import sklearn
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
import optunity
import optunity.metrics

import data_loader
import preprocessing
import climate_preprocessing

np.seterr(divide='ignore', invalid='ignore')


# climate change

def add_shifted_column(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	dfs = [train, test, validation]

	# print(train.head())
	# print(train.tail())

	for df in dfs:
		indexes = []
		for index in df.index:
			indexes.append(index)

		for i in range(0, len(indexes) - 1):
			df.loc[indexes[i], 'y'] = df.loc[indexes[i + 1], 'LandAverageTemperature']


# print(train.head())
# print(train.tail())

def _preproces_svm1(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	dfs = [train, test, validation]
	xs = []
	ys = []

	add_shifted_column(train, test, validation)
	for i in range(len(dfs)):
		dfs[i] = dfs[i].dropna()

	for df in dfs:
		x = []
		y = []
		for index in df.index:
			x.append(df.loc[index, 'LandAverageTemperature'])
			y.append(df.loc[index, 'y'])
		x_np = np.array(x)
		xs.append(x_np.reshape(-1, 1))
		ys.append(y)

	return xs, ys


# features: 5 days of prices
def _preprocess_svm2(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	window_length = 5
	dfs = [train, test, validation]
	xs = []
	ys = []

	for df in dfs:
		x = []
		y = []
		window_offset_count = 0  # skip the first window_length rows
		i = 0
		indexes = []
		for index in df.index:
			window_offset_count += 1
			indexes.append(index)
			if window_offset_count > window_length:
				window = []
				for j in range(0, window_length):
					window.append(df.loc[indexes[i - j], 'WTI'])
				x.append(window)
				y.append(df.loc[index, 'y'])
			i += 1
		x_np = np.array(x)
		xs.append(x_np)
		ys.append(y)

	return xs, ys


'''
	input: WTI
	output: y

	RESULTS:
	Confusion matrix: 
	[[132   0]
	 [118   0]]
		Accuracy:
	0.528
'''


def svm1(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	xs, ys = _preproces_svm1(train, test, validation)
	x_train = xs[0]
	y_train = ys[0]

	x_test = xs[1]
	y_test = ys[1]

	x_val = xs[2]
	y_val = ys[2]
	print('Training...')
	clf = svm.SVR()
	clf.fit(x_train, y_train)
	print('Done')
	validate_svm1(x_val, y_val, clf)


def tuning_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
	"""A generic SVM training function, with arguments based on the chosen kernel."""
	if kernel == 'linear':
		model = sklearn.svm.SVC(kernel=kernel, C=C)
	elif kernel == 'poly':
		model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
	elif kernel == 'rbf':
		model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
	else:
		print("Unknown kernel function: %s" % kernel)
		return
	model.fit(x_train, y_train)
	return model


def tuning_svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
	model = tuning_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
	decision_values = model.decision_function(x_test)
	return optunity.metrics.roc_auc(y_test, decision_values)


x_train = None
y_train = None


def svm2_tuning(train, test, validation):
	xs, ys = _preproces_svm1(train, test, validation)
	global x_train
	global y_train
	x_train = xs[0]
	y_train = ys[0]
	x_val = xs[2]
	y_val = ys[2]

	def tuning_svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
		model = tuning_train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
		decision_values = model.decision_function(x_test)
		return optunity.metrics.roc_auc(y_test, decision_values)

	space = {'kernel': {'linear': {'C': [0, 2]},
	                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
	                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
	                    }
	         }

	print('Cross validating')
	cv_decorator = optunity.cross_validated(x=x_train, y=y_train, num_folds=1)
	tuning_svm_tuned_auroc = cv_decorator(tuning_svm_tuned_auroc)
	print('Tuning')


def validate_svm1(x_val, y_val, model):
	predicted_val_y = []
	# print('Testing...')
	for i in range(0, len(x_val)):
		predicted_val_y.append(model.predict([x_val[i]]))

	# print('\tMean absolute error:')
	error = mean_absolute_error(y_val, predicted_val_y)
	return error


def svm1_tuning(train, test, validation):
	xs, ys = _preproces_svm1(train, test, validation)
	x_train = xs[0]
	y_train = ys[0]
	x_val = xs[2]
	y_val = ys[2]

	# score function: twice iterated 10-fold cross-validated accuracy
	@optunity.cross_validated(x=x_train, y=y_train, num_folds=10, num_iter=2)
	def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
		model = svm.SVR(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
		decision_values = model.predict(x_test)
		return optunity.metrics.roc_auc(y_test, decision_values)

	hyperparameters_2 = {
		'C': 12.620431930182058,
		'gamma': 1.0916563302927996e-05
	}
	hyperparameters_3 = {
		'C': 45.879755387396564,
		'gamma': 0.005822864965270731
	}  # 1.86
	hyperparameters_4 = {
		'C': 2.7938314344707378e+26,
		'gamma': 1.0002069540384684
	}  # error: 1.8598672144573731
	hyperparameters_5 = {
		'C': 1.5507537293892773e+57,
		'gamma': 1.0003165687310716
	}  # error: 1.857474572277201

	default_hyperparameters = {
		'C': 1,
		'gamma': 0.1
	}
	hyperparameters = {}

	if not hyperparameters:
		# perform tuning
		# print('performing tuning...')

		hps, _, _ = optunity.maximize(svm_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 2])
		# print('done, printing hyperparameters:\n')
		# print('C:\t', 10 ** hps['logC'], '\ngamma:\t', 10 ** hps['logGamma'])
		hyperparameters['C'] = 10 ** hps['logC']
		hyperparameters['gamma'] = 10 ** hps['logGamma']
	optimal_model = sklearn.svm.SVR(**hyperparameters).fit(x_train, y_train)
	return hyperparameters, validate_svm1(x_val, y_val, optimal_model)


def main():
	train, test, val = climate_preprocessing.land_average_single()
	min_error = None
	while True:
		hyperparameters, result = svm1_tuning(train, test, val)
		if min_error is None or min_error > result:
			min_error = result
			print('new minimum error found, printing hyperparameters:')
			print('C:\t', 10 ** hyperparameters['C'], '\ngamma:\t', 10 ** hyperparameters['gamma'])
			print(f'\terror: {result}')


# base result: 1.8839129590737358

# svm1_tuning(train, test, validation)


if __name__ == '__main__':
	main()
