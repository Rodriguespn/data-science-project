mport os
from numpy import log
from pandas import read_csv, Series
from scipy.stats import norm, expon, lognorm
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, subplots, Axes, title
from seaborn import distplot, heatmap

from utils.ds_charts import get_variable_types, choose_grid, bar_chart, multiple_bar_chart, multiple_line_chart, HEIGHT
from utils.util import get_datafile, print_with_label


def _compute_known_distributions(x_values: list) -> dict:
	distributions = dict()
	# Gaussian
	mean, sigma = norm.fit(x_values)
	distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
	# Exponential
	loc, scale = expon.fit(x_values)
	distributions['Exp(%.2f)' % (1 / scale)] = expon.pdf(x_values, loc, scale)
	# LogNorm
	sigma, loc, scale = lognorm.fit(x_values)
	distributions['LogNor(%.1f,%.2f)' % (log(scale), sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
	return distributions


def _histogram_with_distributions(ax: Axes, series: Series, var: str):
	values = series.sort_values().values
	ax.hist(values, 20, density=True)
	distributions = _compute_known_distributions(values)
	multiple_line_chart(values, distributions, ax=ax, title='Best fit for %s' % var, xlabel=var, ylabel='')


def data_dimensionality():
	register_matplotlib_converters()
	filename = 'data/' + get_datafile('filename') + '.csv'
	data = read_csv(filename, na_values='', parse_dates=get_datafile('date_columns'), date_parser=get_datafile('date_parser'))
	print_with_label("data.shape", data.shape, symbol='-')

	# Nr Records and Nr Variables
	print_with_label("Nr Records and Nr Variables")
	figure(figsize=(4, 2))
	values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
	bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')

	directory = 'images/' + get_datafile('filename') + "/dimensinality"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + '/records_variables.png')
	show()

	# Variables types
	print_with_label("Variable types")
	print_with_label("data.dtypes", data.dtypes, symbol='-')

	cat_vars = data.select_dtypes(include='object')
	data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
	print_with_label("data.dtypes", data.dtypes, symbol='-')

	variable_types = get_variable_types(data)
	print_with_label("variable_types", variable_types, symbol='-')
	counts = {}
	for tp in variable_types.keys():
		counts[tp] = len(variable_types[tp])
	figure(figsize=(4, 2))
	bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')

	savefig(directory + '/variable_types.png')
	show()

	# Missing values
	print_with_label("Missing values")
	mv = {}
	for var in data:
		nr = data[var].isna().sum()
		if nr > 0:
			mv[var] = nr
	figure()
	bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
	savefig(directory + '/missing_values.png')
	show()


def data_distribution():
	register_matplotlib_converters()
	filename = 'data/' + get_datafile('filename') + '.csv'
	data = read_csv(filename, na_values='', parse_dates=get_datafile('date_columns'), date_parser=get_datafile('date_parser'))
	summary5 = data.describe()
	print_with_label("summary5", summary5, symbol='-')

	# TODO FIXME LALA nondeterministic(?) error
	# Boxplots
	print_with_label("Boxplots")
	data.boxplot(rot=45)

	directory = 'images/' + get_datafile('filename') + "/distribution"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/global_boxplot.png")
	show()

	# Single boxplot
	print_with_label("Single Boxplots")
	numeric_vars = get_variable_types(data)['Numeric']
	if not numeric_vars:
		raise ValueError('There are no numeric variables.')
	rows, cols = choose_grid(len(numeric_vars))
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	i, j = 0, 0
	for n in range(len(numeric_vars)):
		axs[i, j].set_title('Boxplot for %s' % numeric_vars[n])
		axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
		i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
	savefig(directory + "/single_boxplots.png")
	show()

	# Outliers
	print_with_label("Outliers")
	nr_stdev = 2
	numeric_vars = get_variable_types(data)['Numeric']
	if not numeric_vars:
		raise ValueError('There are no numeric variables.')
	outliers_iqr = []
	outliers_stdev = []
	summary5 = data.describe(include='number')
	for var in numeric_vars:
		iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
		outliers_iqr += [
			data[data[var] > summary5[var]['75%'] + iqr].count()[var] +
			data[data[var] < summary5[var]['25%'] - iqr].count()[var]]
		std = nr_stdev * summary5[var]['std']
		outliers_stdev += [
			data[data[var] > summary5[var]['mean'] + std].count()[var] +
			data[data[var] < summary5[var]['mean'] - std].count()[var]]
	outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
	figure(figsize=(12, HEIGHT))
	multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False) 
	savefig(directory + "/outliers.png")
	show()

	# Histograms
	print_with_label("Histograms")
	numeric_vars = get_variable_types(data)['Numeric']
	if not numeric_vars:
		raise ValueError('There are no numeric variables.')
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	i, j = 0, 0
	for n in range(len(numeric_vars)):
		axs[i, j].set_title('Histogram for %s' % numeric_vars[n])
		axs[i, j].set_xlabel(numeric_vars[n])
		axs[i, j].set_ylabel("nr records")
		axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
		i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
	savefig(directory + "/single_histograms_numeric.png")
	show()

	# Probabilities distribution
	# print_with_label("Probabilities distribution")
	# numeric_vars = get_variable_types(data)['Numeric']
	# if not numeric_vars:
	# 	raise ValueError('There are no numeric variables.')
	# fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	# i, j = 0, 0
	# for n in range(len(numeric_vars)):
	# 	axs[i, j].set_title('Histogram with trend for %s' % numeric_vars[n])
	# 	distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
	# 	i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
	# savefig(directory + "/histograms_trend_numeric.png")
	# show()

	numeric_vars = get_variable_types(data)['Numeric']
	if not numeric_vars:
		raise ValueError('There are no numeric variables.')
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	i, j = 0, 0
	for n in range(len(numeric_vars)):
		_histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
		i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
	savefig(directory + "/histogram_numeric_distribution.png")
	show()

	# Symbolic variables
	print_with_label("Symbolic Variables")
	symbolic_vars = get_variable_types(data)['Symbolic']
	if not symbolic_vars:
		raise ValueError('There are no symbolic variables.')
	rows, cols = choose_grid(len(symbolic_vars))
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	i, j = 0, 0
	for n in range(len(symbolic_vars)):
		counts = data[symbolic_vars[n]].value_counts()
		bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s' % symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
		i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
	savefig(directory + "/histograms_symbolic.png")
	show()


def data_granularity():
	filename = 'data/' + get_datafile('filename') + '.csv'
	data = read_csv(filename, na_values='', parse_dates=get_datafile('date_columns'), date_parser=get_datafile('date_parser'))
	values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

	# Numeric
	print_with_label("Granularity Numeric")
	variables = get_variable_types(data)['Numeric']
	if not variables:
		raise ValueError('There are no numeric variables.')
	rows = len(variables)
	bins = (10, 100, 1000)
	cols = len(bins)
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	for i in range(rows):
		for j in range(cols):
			axs[i, j].set_title('Histogram for %s %d bins' % (variables[i], bins[j]))
			axs[i, j].set_xlabel(variables[i])
			axs[i, j].set_ylabel('Nr records')
			axs[i, j].hist(data[variables[i]].values, bins=bins[j])

	directory = 'images/' + get_datafile('filename') + "/granularity"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/granularity_study_numeric.png")
	show()

	# TODO FIXME LALA taxonomies
	# Symbolic
	# print_with_label("Granularity Symbolic")
	# variables = get_variable_types(data)['Symbolic']
	# if not variables:
	# 	raise ValueError('There are no symbolic variables.')
	# rows = len(variables)
	# bins = (10, 100, 1000)
	# cols = len(bins)
	# fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	# for i in range(rows):
	# 	for j in range(cols):
	# 		axs[i, j].set_title('Histogram for %s %d bins' % (variables[i], bins[j]))
	# 		axs[i, j].set_xlabel(variables[i])
	# 		axs[i, j].set_ylabel('Nr records')
	# 		axs[i, j].hist(data[variables[i]].values, bins=bins[j])
	# savefig(directory + "/granularity_study_symbolic.png")
	# show()

	# Dates
	print_with_label("Granularity Dates")
	variables = get_variable_types(data)['Date']
	if not variables:
		raise ValueError('There are no date variables.')
	rows = len(variables)
	bins = (10, 100, 1000)
	cols = len(bins)
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	for i in range(rows):
		for j in range(cols):
			axs[i, j].set_title('Histogram for %s %d bins' % (variables[i], bins[j]))
			axs[i, j].set_xlabel(variables[i])
			axs[i, j].set_ylabel('Nr records')
			axs[i, j].hist(data[variables[i]].values, bins=bins[j])
	savefig(directory + "/granularity_study_date.png")
	show()


def data_sparsity():
	register_matplotlib_converters()
	filename = 'data/' + get_datafile('filename') + '.csv'
	data = read_csv(filename, na_values='', parse_dates=get_datafile('date_columns'), date_parser=get_datafile('date_parser'))

	# Scatter plots numeric
	print_with_label("Scatter plots numeric")
	numeric_vars = get_variable_types(data)['Numeric']
	if not numeric_vars:
		raise ValueError('There are no numeric variables.')
	rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
	fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	for i in range(len(numeric_vars)):
		var1 = numeric_vars[i]
		for j in range(i + 1, len(numeric_vars)):
			var2 = numeric_vars[j]
			axs[i, j - 1].set_title("%s x %s" % (var1, var2))
			axs[i, j - 1].set_xlabel(var1)
			axs[i, j - 1].set_ylabel(var2)
			print(data[var1])
			axs[i, j - 1].scatter(data[var1], data[var2])

	directory = 'images/' + get_datafile('filename') + "/sparsity"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/sparsity_study_numeric.png")
	show()

	# TODO FIXME LALA error
	# scatter plots symbolic all pairs
	# symbolic_data = data.copy()
	# pd.set_option('display.max_columns', 500)
	# pd.set_option('display.width', 1000)
	# print_with_label("Scatter plots symbolic all pairs")
	# symbolic_vars = get_variable_types(symbolic_data)['Symbolic']
	# print("symbolic_vars\n", symbolic_vars)
	#
	# if not symbolic_vars:
	# 	raise ValueError('There are no symbolic variables.')
	# rows, cols = len(symbolic_vars) - 1, len(symbolic_vars) - 1
	# fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
	# for i in range(len(symbolic_vars)):
	# 	var1 = symbolic_vars[i]
	# 	symbolic_data[var1].fillna('', inplace=True)
	# 	for j in range(i + 1, len(symbolic_vars)):
	# 		var2 = symbolic_vars[j]
	# 		axs[i, j - 1].set_title("%s x %s" % (var1, var2))
	# 		axs[i, j - 1].set_xlabel(var1)
	# 		axs[i, j - 1].set_ylabel(var2)
	# 		print(symbolic_data[var1])
	# 		symbolic_data[var2].fillna('', inplace=True)
	# 		print(symbolic_data[var2])
	# 		axs[i, j - 1].scatter(symbolic_data[var1], symbolic_data[var2])
	# print("Saving image")
	# savefig(directory + "/sparsity_study_symbolic.png")
	# show()

	# Correlation analysis
	print_with_label("Correlation")
	corr_mtx = abs(data.corr())
	print_with_label("corr_mtx", corr_mtx, symbol='-')

	# Correlation plot
	print_with_label("Correlation plot")
	fig = figure(figsize=[12, 12])
	heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
	title('Correlation analysis')
	savefig(directory + "/correlation_analysis.png")
	show()
