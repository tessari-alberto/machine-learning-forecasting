import pandas as pd
import utils
import data_loader
import numpy as np
import matplotlib.pyplot as plt


def load_global():
	return data_loader.load_climate_dataframe().dropna()


def split_dataset(df, train_percent, test_percent):
	total_rows = len(df.index)
	train_rows = int(train_percent / 100 * total_rows)
	test_rows = train_rows + int(test_percent / 100 * total_rows)

	# df.iloc[:, :train_rows]
	tmp_df_list = np.split(df, [train_rows, test_rows], axis=0)
	# print(tmp_df_list)
	train = tmp_df_list[0]
	test = tmp_df_list[1]
	val = tmp_df_list[2]

	return train, test, val


# columns:
# LandAverageTemperature  LandAverageTemperatureUncertainty LandMaxTemperature
# LandMaxTemperatureUncertainty LandMinTemperature LandMinTemperatureUncertainty
# LandAndOceanAverageTemperature LandAndOceanAverageTemperatureUncertainty
# first date: 1750-01-01 (only land average temperature)
# first complete data date: 1850-01-01
# last date: 2015-12-01
def std_preprocessing():
	df = load_global()
	print(df.head())


def land_average_single():
	df = load_global()
	df = df.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperature', 'LandMaxTemperatureUncertainty',
	                      'LandMinTemperature', 'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperature',
	                      'LandAndOceanAverageTemperatureUncertainty'])
	train, test, val = split_dataset(df, 70, 20)
	return train, test, val


def land_average_half():
	df = load_global()
	df = df.dropna()
	df = df.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperatureUncertainty', 'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperatureUncertainty'])
	train, test, val = split_dataset(df, 70, 20)
	return train, test, val


def land_average_all():
	df = load_global()
	df = df.dropna()
	# df = df.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperatureUncertainty', 'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperatureUncertainty'])
	train, test, val = split_dataset(df, 70, 20)
	return train, test, val

def plot_dataset():
	df = load_global()
	train, test, val = split_dataset(df, 70, 20)
	plt.plot(val['LandAverageTemperature'])
	plt.xlabel('Date')
	plt.ylabel('°C', rotation=0)
	plt.show()

if __name__ == '__main__':
	plot_dataset()
