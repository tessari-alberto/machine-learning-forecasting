import commons
import utils
import pandas as pd

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


def _first_examination():
	with open(commons.first_dataset_path) as fd:
		counter = 0
		for line in fd.readlines():
			counter += 1
			if counter == 1:
				print(line)


def _replace_line(file_path, pattern, subst):
	# Create temp file
	fh, abs_path = mkstemp()
	with fdopen(fh, 'w') as new_file:
		with open(file_path) as old_file:
			for line in old_file:
				new_file.write(line.replace(pattern, subst))
	# Copy the file permissions from the old file to the new file
	copymode(file_path, abs_path)
	# Remove original file
	remove(file_path)
	# Move new file
	move(abs_path, file_path)


def _translate_months_data1():
	months = {
		'gen': 'Jan',
		'feb': 'Feb',
		'mar': 'Mar',
		'apr': 'Apr',
		'mag': 'May',
		'giu': 'Jun',
		'lug': 'Jul',
		'ago': 'Aug',
		'set': 'Sep',
		'ott': 'Opt',
		'nov': 'Nov',
		'dic': 'Dec',
		'Opt': 'Oct'
	}

	for ita, eng in months.items():
		if ita != eng:
			print(f'Replacing: {ita} -> {eng}')
			_replace_line(commons.first_dataset_path, ita, eng)


def load_dataframe():
	df = pd.read_csv(commons.first_dataset_path, decimal=',')
	df['date'] = pd.to_datetime(df['date'])
	df['WTI'] = df['WTI'].astype(float)
	df['BRT'] = df['BRT'].replace(',', '.', regex=True).astype(float)
	return df

def load_climate_dataframe(dataset_name='global'):
	pd.set_option('display.max_columns', None)
	if dataset_name == 'global':
		df = pd.read_csv(commons.world_climate_dataset_path)

		df['date'] = pd.to_datetime(df['dt'])
		df.index = df['date']
		df = df.drop(columns=['dt', 'date'])
		return df



def load_validation_set():
	df = pd.read_csv(commons.validation_dataset_path, decimal=',')
	df['date'] = pd.to_datetime(df['DATE'])
	df = df.drop(df[df.DCOILWTICO == '.'].index)
	df['WTI'] = df['DCOILWTICO'].astype(float)

	selection_lower_limit = utils.date_from_year('2021')
	selection_upper_limit = utils.date_from_year('2022')

	select_2021_mask = (df['date'] > selection_lower_limit) & (df['date'] < selection_upper_limit)
	df = df.loc[select_2021_mask]

	df = df.drop(columns=['DATE', 'DCOILWTICO'])

	return df


def main():
	load_climate_dataframe()


if __name__ == '__main__':
	main()
