import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import data_loader
import preprocessing
import utils
from utils import same_line_print as slp
from window_generator_2 import WindowGenerator
import climate_preprocessing

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

TRAIN = 1
TEST = 10
VALIDATION = 100


# baseline model for comparisson (no changes in LandAverageTemperature)
class Baseline(tf.keras.Model):
	def __init__(self, label_index=None):
		super().__init__()
		self.label_index = label_index

	def call(self, inputs):
		if self.label_index is None:
			return inputs
		result = inputs[:, :, self.label_index]
		return result[:, :, tf.newaxis]


class TfNeural:
	MAX_EPOCHS = 100
	train = None
	test = None
	validation = None
	tmp_path = '../data/tmp/'
	flags = {
		'print_preprocessing': False,
		'print_samples_report': True,
		'plot_dataset': False
	}

	def __init__(self, flags: dict):
		for flag in flags.items():
			self.flags[flag[0]] = flag[1]
		self.preprocessing()


		if self.flags['plot_dataset']:
			self.plot_dataset(TRAIN | TEST | VALIDATION)

		self._convert_to_timestamp()
		self.normalize()

	def preprocessing(self):
		slp('Loading dataset... ')
		print('Done')

		slp('Preprocessing dataset... ')
		self.train, self.test, self.validation = climate_preprocessing.land_average_single()

		print('Done')
		if self.flags['print_preprocessing']:
			self.print_datasets()
		if self.flags['print_samples_report']:
			self._print_samples_report()

		self.train.to_pickle(self.tmp_path + 'train')
		self.test.to_pickle(self.tmp_path + 'test')
		self.validation.to_pickle(self.tmp_path + 'validation')

	def _print_samples_report(self):
		print('\n\n\t\tLOADED SAMPLES REPORT')
		print(
			f'Training set: {len(self.train.index)} samples\nTesting set: {len(self.test.index)} samples\n'
			f'Validation set: {len(self.validation.index)} samples\n')

	def print_datasets(self):
		print('\n\t\tDATASETS:')
		print('\tTraining set: ')
		print('\n', self.train.head(), '\n\n', self.train.tail())
		print('\n\n\tTest set: ')
		print('\n', self.test.head(), '\n\n', self.test.tail())
		print('\n\n\tValidation set: ')
		print('\n', self.validation.head(), '\n\n', self.validation.tail())

	def data_analysis(self):
		print('\n\t\tTRAINING SET:\n', self.train.describe().transpose())
		print('\n\n\t\tTESTING SET:\n', self.test.describe().transpose())
		print('\n\n\t\tVALIDATION SET:\n', self.validation.describe().transpose())

	def _convert_to_timestamp(self):
		self.train.index = pd.to_datetime(self.train.index, format='%d-%m-%Y')

		self.test.index = pd.to_datetime(self.test.index, format='%d-%m-%Y')

		self.validation.index = pd.to_datetime(self.validation.index, format='%d-%m-%Y')

	def plot_frequence(self):
		fft = tf.signal.rfft(self.train['LandAverageTemperature'])
		f_per_dataset = np.arange(0, len(fft))

		n_samples_h = len(self.train['LandAverageTemperature'])
		hours_per_year = 24 * 365.2524
		years_per_dataset = n_samples_h / (hours_per_year)

		f_per_year = f_per_dataset / years_per_dataset
		plt.step(f_per_year, np.abs(fft))
		plt.xscale('log')
		plt.ylim(0, 400000)
		plt.xlim([0.1, max(plt.xlim())])
		plt.xticks([1, 30, 365.2524], labels=['1/Year', '1/month', '1/day'])
		_ = plt.xlabel('Frequency (log scale)')
		plt.show()

	def plot_dataset(self, set=111):
		p_train = set % 10 == 1
		p_test = int(set / 10) % 10 == 1
		p_validation = int(set / 100) % 10 == 1

		if p_train:
			self.train.LandAverageTemperature.plot(figsize=(15, 8), title='LandAverageTemperature', fontsize=14, label='train')
		if p_test:
			self.test.LandAverageTemperature.plot(figsize=(15, 8), title='LandAverageTemperature', fontsize=14, label='test')
		if p_validation:
			self.validation.LandAverageTemperature.plot(figsize=(15, 8), title='LandAverageTemperature', fontsize=14, label='validation')
		plt.xlabel("Datetime")
		plt.ylabel("LandAverageTemperature")
		plt.legend(loc='best')
		plt.show()

	def normalize(self, plot=False):
		train_mean = self.train.mean()
		train_std = self.train.std()

		self.train = (self.train - train_mean) / train_std
		self.test = (self.test - train_mean) / train_std
		self.validation = (self.validation - train_mean) / train_std

	def create_windows(self, in_len, out_len, shift, label_columns=None):
		self.window = WindowGenerator(input_width=in_len, label_width=out_len, shift=shift, label_columns=label_columns,
		                              train_df=self.train, test_df=self.test, val_df=self.validation)

	def print_window(self):
		# print(self.window)
		for example_inputs, example_labels in self.window.train.take(1):
			print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
			print(f'Labels shape (batch, time, features): {example_labels.shape}')

	def print_example_window(self):
		example_window = tf.stack([np.array(self.train[:self.window.total_window_size]),
		                           np.array(self.train[100:100 + self.window.total_window_size]),
		                           np.array(self.train[200:200 + self.window.total_window_size])])
		example_inputs, example_labels = self.window.split_window(example_window)
		print('All shapes are: (batch, time, features)')
		print(f'Window shape: {example_window.shape}')
		print(f'Inputs shape: {example_inputs.shape}')
		print(f'Labels shape: {example_labels.shape}')
		# self.window.plot(inputs=example_inputs, labels=example_labels)
		for example_inputs, example_labels in self.window.train.take(1):
			print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
			print(f'Labels shape (batch, time, features): {example_labels.shape}')

	def get_window_column_indices(self):
		return self.window.get_column_indices()

	def get_window(self):
		return self.window

	def compile_and_fit(self, model, window, patience=2):
		early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
		                                                  patience=patience,
		                                                  mode='min')

		model.compile(loss=tf.keras.losses.MeanSquaredError(),
		              optimizer=tf.keras.optimizers.Adam(),
		              metrics=[tf.keras.metrics.MeanAbsoluteError()])

		history = model.fit(window.train, epochs=self.MAX_EPOCHS,
		                    validation_data=window.val,
		                    callbacks=[early_stopping])
		return history


val_performance = {}
performance = {}
# linear, one layer
def first_model(tfn: TfNeural):
	# 'Linear': [0.0022376261185854673, 0.035708628594875336]
	tfn.create_windows(in_len=1, out_len=1, shift=1,
	                   label_columns=['LandAverageTemperature'])  # loss: 0.0072 - mean_absolute_error: 0.0705
	# performance 'Linear': [0.0027509911451488733, 0.04063655808568001]
	# tfn.create_windows(in_len=24, out_len=24, shift=1, label_columns=['LandAverageTemperature'])
	tfn.print_window()
	window = tfn.get_window()
	baseline = Baseline(label_index=tfn.get_window_column_indices()['LandAverageTemperature'])
	baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
	                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
	# window.plot(baseline)
	global val_performance
	global performance
	val_performance['Baseline'] = baseline.evaluate(window.val)
	performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

	linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
	print('Input shape:', window.example[0].shape)
	print('Output shape:', linear(window.example[0]).shape)

	history = tfn.compile_and_fit(linear, window)

	val_performance['linear'] = linear.evaluate(window.val)
	performance['linear'] = linear.evaluate(window.test, verbose=0)

	print(f'\n\nval performance: {val_performance}')
	print(f'\n\nperformance: {performance}')


# linear, more layers
def second_model(tfn: TfNeural):
	# 'Dense': [0.0021276292391121387, 0.03380898758769035]
	# tfn.create_windows(in_len=1, out_len=1, shift=1, label_columns=['LandAverageTemperature']) #loss: 0.0020 - mean_absolute_error: 0.0312
	# 'Dense': [0.0021205213852226734, 0.033576034009456635]
	tfn.create_windows(in_len=24, out_len=24, shift=1, label_columns=['LandAverageTemperature'])
	tfn.print_window()
	window = tfn.get_window()
	baseline = Baseline(label_index=tfn.get_window_column_indices()['LandAverageTemperature'])
	baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
	                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
	# window.plot(baseline)
	global val_performance
	global performance
	val_performance['Baseline'] = baseline.evaluate(window.val)
	performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

	dense = tf.keras.Sequential([
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=64, activation='relu'),
		tf.keras.layers.Dense(units=1)
	])
	print('Input shape:', window.example[0].shape)
	print('Output shape:', dense(window.example[0]).shape)

	history = tfn.compile_and_fit(dense, window)

	val_performance['linear_more_layers'] = dense.evaluate(window.val)
	performance['linear_more_layers'] = dense.evaluate(window.test, verbose=0)
	print(f'\n\nval performance: {val_performance}')
	print(f'\n\nperformance: {performance}')


# multi step dense model
def third_model(tfn: TfNeural):
	CONV_WIDTH = 12  # n giorni di input per 1 output
	# 'multi_step_dense ': [0.002276645740494132,  0.03497900441288948] CONV 3
	# 'multi_step_dense ': [0.0021659997291862965, 0.034185741096735]   CONV 4 <-
	# 'multi_step_dense ': [0.002225790871307254,  0.03457000106573105] CONV 5
	# 'multi_step_dense ': [0.0021946390625089407, 0.03415375202894211] CONV 6
	# 'multi_step_dense ': [0.0022112722508609295, 0.03482969477772713] CONV 7
	# 'multi_step_dense ': [0.0030520951841026545, 0.04249454662203789] CONV 30
	tfn.create_windows(in_len=CONV_WIDTH, out_len=1, shift=1, label_columns=['LandAverageTemperature'])
	tfn.print_window()
	window = tfn.get_window()
	baseline = Baseline(label_index=tfn.get_window_column_indices()['LandAverageTemperature'])
	baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
	                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
	# window.plot(baseline)
	global val_performance
	global performance
	val_performance['Baseline'] = baseline.evaluate(window.val)
	performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

	multi_step_dense = tf.keras.Sequential([
		# Shape: (time, features) => (time*features)
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1),
		# Add back the time dimension.
		# Shape: (outputs) => (1, outputs)
		tf.keras.layers.Reshape([1, -1]),
	])
	print('Input shape:', window.example[0].shape)
	print('Output shape:', multi_step_dense(window.example[0]).shape)

	history = tfn.compile_and_fit(multi_step_dense, window)

	val_performance['multi-step_dense'] = multi_step_dense.evaluate(window.val)
	performance['multi-step_dense '] = multi_step_dense.evaluate(window.test, verbose=0)
	print(f'\n\nval performance: {val_performance}')
	print(f'\n\nperformance: {performance}')

	window.plot(multi_step_dense)


# convolutional model
def fourth_model(tfn: TfNeural):
	CONV_WIDTH = 11
	# 'conv_model ': [0.0026435689069330692, 0.03874440863728523] CONV 4
	# 'conv_model ': [0.002282282803207636,  0.03551756963133812] CONV 7
	# 'conv_model ': [0.0025794184766709805, 0.038141898810863495] CONV 30
	tfn.create_windows(in_len=CONV_WIDTH, out_len=1, shift=1, label_columns=['LandAverageTemperature'])
	tfn.print_window()
	window = tfn.get_window()
	baseline = Baseline(label_index=tfn.get_window_column_indices()['LandAverageTemperature'])
	baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
	                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
	# window.plot(baseline)
	global val_performance
	global performance
	val_performance['Baseline'] = baseline.evaluate(window.val)
	performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

	conv_model = tf.keras.Sequential([
		tf.keras.layers.Conv1D(filters=32,
		                       kernel_size=(CONV_WIDTH,),
		                       activation='relu'),
		tf.keras.layers.Dense(units=32, activation='relu'),
		tf.keras.layers.Dense(units=1),
	])
	print('Input shape:', window.example[0].shape)
	print('Output shape:', conv_model(window.example[0]).shape)

	history = tfn.compile_and_fit(conv_model, window)

	val_performance['CNN'] = conv_model.evaluate(window.val)
	performance['CNN '] = conv_model.evaluate(window.test, verbose=0)
	print(f'\n\nval performance: {val_performance}')
	print(f'\n\nperformance: {performance}')

	LABEL_WIDTH = 24
	INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
	tfn.create_windows(
		in_len=INPUT_WIDTH,
		out_len=LABEL_WIDTH,
		shift=1,
		label_columns=['LandAverageTemperature'])
	wide_conv_window = tfn.get_window()

	wide_conv_window.plot(conv_model)


# RNN
def fifth_model(tfn: TfNeural):
	# 'conv_model ': [0.0022057273890823126, 0.03405054286122322] # 24 24
	tfn.create_windows(in_len=12, out_len=1, shift=1, label_columns=['LandAverageTemperature'])
	tfn.print_window()
	window = tfn.get_window()
	baseline = Baseline(label_index=tfn.get_window_column_indices()['LandAverageTemperature'])
	baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
	                 metrics=[tf.keras.metrics.MeanAbsoluteError()])
	# window.plot(baseline)
	global val_performance
	global performance
	val_performance['Baseline'] = baseline.evaluate(window.val)
	performance['Baseline'] = baseline.evaluate(window.test, verbose=0)

	lstm_model  = tf.keras.Sequential([
		# Shape [batch, time, features] => [batch, time, lstm_units]
		tf.keras.layers.LSTM(32, return_sequences=True),
		# Shape => [batch, time, features]
		tf.keras.layers.Dense(units=1)
	])
	print('Input shape:', window.example[0].shape)
	print('Output shape:', lstm_model (window.example[0]).shape)

	history = tfn.compile_and_fit(lstm_model , window)

	val_performance['RNN'] = lstm_model .evaluate(window.val)
	performance['RNN '] = lstm_model .evaluate(window.test, verbose=0)
	print(f'\n\nval performance: {val_performance}')
	print(f'\n\nperformance: {performance}')

	#window.plot(lstm_model)
	return lstm_model


def test_all_models(tfn: TfNeural):
	global val_performance
	global performance

	first_model(tfn)
	second_model(tfn)
	third_model(tfn)
	fourth_model(tfn)
	lstm_model = fifth_model(tfn)

	x = np.arange(len(performance))
	width = 0.3
	metric_name = 'mean_absolute_error'
	metric_index = lstm_model.metrics_names.index('mean_absolute_error')
	val_mae = [v[metric_index] for v in val_performance.values()]
	test_mae = [v[metric_index] for v in performance.values()]

	plt.ylabel('mean_absolute_error [T (degC), normalized]')
	plt.bar(x - 0.17, val_mae, width, label='Validation')
	plt.bar(x + 0.17, test_mae, width, label='Test')
	plt.xticks(ticks=x, labels=performance.keys(),
	           rotation=45)
	_ = plt.legend()
	plt.show()

	for name, value in performance.items():
		print(f'{name:12s}: {value[1]:0.4f}')

def main():
	flags = {
		'print_preprocessing': True,
		'print_samples_report': True,
		'plot_dataset': False
	}
	tfn = TfNeural(flags=flags)
	# tfn.create_windows(in_len=6, out_len=1, shift=1, label_columns=['LandAverageTemperature'])
	test_all_models(tfn)


# tfn.print_datasets()


if __name__ == '__main__':
	main()
