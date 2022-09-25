import pandas as pd
import numpy as np
import datetime

import sklearn
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import optunity
import optunity.metrics

import data_loader
import preprocessing


def _preproces_svm1(train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
	dfs = [train, test, validation]
	xs = []
	ys = []

	for df in dfs:
		x = []
		y = []
		for index in df.index:
			x.append(df.loc[index, 'WTI'])
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
	clf = svm.SVC()
	clf.fit(x_train, y_train)

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

	space = {'kernel': {'linear': {'C': [0, 2]},
	                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
	                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
	                    }
	         }





def validate_svm1(x_val, y_val, model):
	predicted_val_y = []
	print('Testing...')
	for i in range(0, len(x_val)):
		predicted_val_y.append(model.predict([x_val[i]]))

	print('\tConfusion matrix: ')
	print(confusion_matrix(y_val, predicted_val_y))

	print('\tAccuracy:')
	print(accuracy_score(y_val, predicted_val_y))


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
	hyperparameters = {}

	if not hyperparameters:
		# perform tuning
		print('performing tuning...')

		hps, _, _ = optunity.maximize(svm_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
		print('done, printing hyperparameters:\n')
		print('C:\t', 10 ** hps['logC'], '\ngamma:\t', 10 ** hps['logGamma'])
		hyperparameters['C'] = 10 ** hps['logC']
		hyperparameters['gamma'] = 10 ** hps['logGamma']
	optimal_model = sklearn.svm.SVC(**hyperparameters).fit(x_train, y_train)
	validate_svm1(x_val, y_val, optimal_model)


def main():
	df = data_loader.load_dataframe()
	validation = data_loader.load_validation_set()

	df_original = df.copy()
	validation_original = validation.copy()

	train, test, validation = preprocessing.prepare_for_classification(df, validation)
	svm1_tuning(train, test, validation)


# svm1_tuning(train, test, validation)


if __name__ == '__main__':
	main()
