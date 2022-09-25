import pandas as pd
from datetime import datetime
import data_loader
import numpy as np
import utils
import matplotlib.pyplot as plt

# modeling and forecasting
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from joblib import dump, load

pd.options.mode.chained_assignment = None  # default='warn'

df_original = None
validation_original = None


def _plot(df: pd.DataFrame):
	print(df.describe())
	df.plot(x='date', y=['WTI', 'BRT'])
	plt.show()


def _split_dataset(df: pd.DataFrame):
	train_lower_limit = utils.date_from_year('2000')
	train_upper_limit = utils.date_from_year('2019')

	train_mask = (df['date'] > train_lower_limit) & (df['date'] < train_upper_limit)
	train = df.loc[train_mask]
	test = df[df['date'] >= '2019']

	return train, test


def _weekend_extractor(row):
	return row.dayofweek


def _plot_combinend_dataset(train, test, validation):
	train.WTI.plot(figsize=(15, 8), title='WTI price', fontsize=14, label='train')
	test.WTI.plot(figsize=(15, 8), title='WTI price', fontsize=14, label='test')
	validation.WTI.plot(figsize=(15, 8), title='WTI price', fontsize=14, label='validation')
	plt.xlabel("Datetime")
	plt.ylabel("WTI price")
	plt.legend(loc='best')
	plt.show()


def _mild_preprocessing(df: pd.DataFrame, validation: pd.DataFrame):
	df = df[df['date'] >= '2000']
	df = df.drop(columns=['BRT'])
	df.index = df['date']
	validation.index = validation['date']
	train_ref, test_ref = _split_dataset(df)

	train = pd.DataFrame(train_ref)
	test = pd.DataFrame(test_ref)

	# dropping NaN rows
	train = train.dropna()
	test = test.dropna()
	validation = validation.dropna()

	return train, test, validation


def _full_preprocessing(df: pd.DataFrame, validation: pd.DataFrame):
	df = df[df['date'] >= '2000']

	# adding features: year, month, day
	# main dataset
	for i in (df, df_original, validation):
		i['year'] = i.date.dt.year
		i['month'] = i.date.dt.month
		i['day'] = i.date.dt.day

	# validation dataset

	# adding feature: day of the week
	# main dataset
	df['day_of_week'] = df['date'].apply(_weekend_extractor)
	# validation
	validation['day_of_week'] = validation['date'].apply(_weekend_extractor)

	df = df.drop(columns=['BRT'])

	df.index = df['date']

	# df.groupby('month')['WTI'].mean().plot.bar()
	# plt.show()

	# tmp = df.groupby(['year', 'month'])['WTI'].mean()
	# tmp.plot(title='WTI price Monthwise')
	# plt.show()

	# checking data's behavior after resampling
	# day_of_week = df.resample('day_of_week').mean()
	daily = df.resample('D').mean()
	weekly = df.resample('W').mean()
	monthly = df.resample('M').mean()
	yearly = df.resample('Y').mean()

	# fig, axs = plt.subplots(4, 1)
	# day_of_week.Count.plot(figsize=(15, 8), title='Hourly', fontsize=14, ax=axs[0])
	# daily.WTI.plot(figsize=(15, 8), title='Daily', fontsize=14, ax=axs[0])
	# weekly.WTI.plot(figsize=(15, 8), title='Weekly', fontsize=14, ax=axs[1])
	# monthly.WTI.plot(figsize=(15, 8), title='Monthly', fontsize=14, ax=axs[2])
	# yearly.WTI.plot(figsize=(15, 8), title='Yearly', fontsize=14, ax=axs[3])
	# plt.show()

	train_ref, test_ref = _split_dataset(df)

	train = pd.DataFrame(train_ref)
	test = pd.DataFrame(test_ref)

	# dropping NaN rows
	train = train.dropna()
	test = test.dropna()
	validation = validation.dropna()

	# resampling the datasets (training, testing and validation) to daily

	train['Timestamp'] = pd.to_datetime(train.date, format='%d-%m-%Y')
	train.index = train.Timestamp
	train = train.resample('D').mean()

	test['Timestamp'] = pd.to_datetime(test.date, format='%d-%m-%Y')
	test.index = test.Timestamp
	test = test.resample('D').mean()

	validation['Timestamp'] = pd.to_datetime(validation.date, format='%d-%m-%Y')
	validation.index = validation.Timestamp
	validation = validation.resample('D').mean()

	# _plot_combinend_dataset(train, test, validation)

	return train, test, validation


def main():
	global df_original
	global validation_original

	df = data_loader.load_dataframe()
	validation = data_loader.load_validation_set()

	df_original = df.copy()
	validation_original = validation.copy

	train, test, validation = _mild_preprocessing(df, validation)
	print(train.head())


# print(validation.tail(50))


# _plot(df)

# train, test = _split_dataset(df)


if __name__ == '__main__':
	main()
