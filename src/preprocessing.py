import pandas as pd
import utils
import data_loader
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


def _split_dataset(df: pd.DataFrame, validation: pd.DataFrame):
	train_lower_limit = utils.date_from_year('2000')
	train_upper_limit = utils.date_from_year('2018')
	validation_lower_limit = utils.date_from_year('2019')

	train_mask = (df['date'] > train_lower_limit) & (df['date'] < train_upper_limit)
	test_mask = (df['date'] > train_upper_limit) & (df['date'] < validation_lower_limit)
	train = df.loc[train_mask]
	test = df.loc[test_mask]  # df[df['date'] >= '2019']
	validation = validation[validation['date'] >= '2019']

	return train, test, validation


def _weekend_extractor(row):
	return row.dayofweek


def mild_preprocessing(df: pd.DataFrame, validation: pd.DataFrame):
	df = df[df['date'] >= '2000']
	df = df.drop(columns=['BRT'])
	validation = validation.drop(columns=['BRT'])
	df.index = df['date']
	validation.index = validation['date']
	train_ref, test_ref, validation_ref = _split_dataset(df, validation)

	train = pd.DataFrame(train_ref)
	test = pd.DataFrame(test_ref)
	validation = pd.DataFrame(validation_ref)

	# dropping NaN rows
	train = train.dropna()
	test = test.dropna()
	validation = validation.dropna()

	train = train.drop(columns='date')
	test = test.drop(columns='date')
	validation = validation.drop(columns='date')

	return train, test, validation


def prepare_for_classification(df: pd.DataFrame, validation: pd.DataFrame):
	train, test, validation = full_preprocessing(df, validation)

	dfs = [train, test, validation]

	for dataframe in dfs:
		prev_index = None
		dataframe['y'] = -1
		for index in dataframe.index:
			if prev_index is None:
				prev_index = index
				continue
			curr_wti = dataframe.loc[index, 'WTI']
			prev_wti = dataframe.loc[prev_index, 'WTI']
			if curr_wti > prev_wti:
				dataframe.loc[prev_index, 'y'] = 1
			else:
				dataframe.loc[prev_index, 'y'] = 0

			prev_index = index

	train = train.dropna()
	test = test.dropna()
	validation = validation.dropna()

	train.drop(train.tail(1).index, inplace=True)
	test.drop(test.tail(1).index, inplace=True)
	validation.drop(validation.tail(1).index, inplace=True)

	return train, test, validation


def full_preprocessing(df: pd.DataFrame, validation: pd.DataFrame):
	df = df[df['date'] >= '2000']

	# adding features: year, month, day
	# main dataset
	for i in (df, validation):
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

	train_ref, test_ref, val_ref = _split_dataset(df, validation)

	train = pd.DataFrame(train_ref)
	test = pd.DataFrame(test_ref)
	validation = pd.DataFrame(val_ref)

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


def _test_classification_preprocessing():
	df = data_loader.load_dataframe()
	val = data_loader.load_validation_set()

	train, test, val = prepare_for_classification(df, val)

	print(train.tail())
	print(test.tail())
	print(val.tail())


def plot_dataset():
	df = data_loader.load_dataframe()
	df['Timestamp'] = pd.to_datetime(df.date, format='%d-%m-%Y')
	df.index = df['Timestamp']
	df = df.drop(columns='Timestamp')
	plt.plot(df['WTI'])
	#df = full_preprocessing(df, val)

	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.show()


def main():
	# _test_classification_preprocessing()
	plot_dataset()


if __name__ == '__main__':
	main()
