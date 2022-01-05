import os
from numpy import ndarray
from pandas import read_csv, DataFrame, concat, unique
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import subplots, savefig, show
from utils.ds_charts import get_variable_types, plot_evaluation_results, multiple_line_chart
from utils.util import get_datafile, set_datafile, print_with_label


def data_scaling(prep_filename):
	register_matplotlib_converters()
	filename = 'data-prep/' + prep_filename + '.csv'
	data = read_csv(filename)

	# Dats scaling
	print_with_label("Data scaling")
	variable_types = get_variable_types(data)
	numeric_vars = variable_types['Numeric']
	symbolic_vars = variable_types['Symbolic']
	boolean_vars = variable_types['Binary']
	df_nr = data[numeric_vars]
	df_sb = data[symbolic_vars]
	df_bool = data[boolean_vars]

	print_with_label("Data scaling z-score")
	transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
	tmp = DataFrame(transf.transform(df_nr), index=data.index, columns=numeric_vars)
	norm_data_zscore = concat([tmp, df_sb, df_bool], axis=1)
	norm_data_zscore.to_csv('data-scaled/' + prep_filename + '_scaled_zscore.csv', index=False)
	print_with_label("norm_data_zscore", norm_data_zscore.describe(), symbol='-')

	print_with_label("Data scaling minmax")
	transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
	tmp = DataFrame(transf.transform(df_nr), index=data.index, columns=numeric_vars)
	norm_data_minmax = concat([tmp, df_sb, df_bool], axis=1)
	norm_data_minmax.to_csv('data-scaled/' + prep_filename + '_scaled_minmax.csv', index=False)
	print_with_label("norm_data_minmax", norm_data_minmax.describe(), symbol='-')

	fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
	axs[0, 0].set_title('Original data')
	data.boxplot(ax=axs[0, 0])
	axs[0, 1].set_title('Z-score normalization')
	norm_data_zscore.boxplot(ax=axs[0, 1])
	axs[0, 2].set_title('MinMax normalization')
	norm_data_minmax.boxplot(ax=axs[0, 2])

	directory = 'images/' + prep_filename
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/scaled_boxplot.png")
	show()


def data_knn(prep_filename, target_class):
	filename = 'data-scaled/' + prep_filename + '.csv'
	data: DataFrame = read_csv(filename)

	data.drop(['City_EN', 'Prov_EN'], 1, inplace=True)
	data.drop(data[data['GbCity'] == 's'].index, inplace=True)

	y: ndarray = data.pop(target_class).values
	x: ndarray = data.values
	labels: np.ndarray = unique(y)
	labels.sort()

	trnX, tstX, trnY, tstY = train_test_split(x, y, train_size=0.7, stratify=y)

	nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
	dist = ['manhattan', 'euclidean', 'chebyshev']
	values = {}
	best = (0, '')
	last_best = 0
	for d in dist:
		yvalues = []
		for n in nvalues:
			knn = KNeighborsClassifier(n_neighbors=n, metric=d)
			knn.fit(trnX, trnY)
			prdY = knn.predict(tstX)
			yvalues.append(accuracy_score(tstY, prdY))
			if yvalues[-1] > last_best:
				best = (n, d)
				last_best = yvalues[-1]
			print(d, n)
		values[d] = yvalues

	multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)

	directory = 'images/' + prep_filename + "/knn_study"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/" + target_class + ".png")
	show()
	print('Best results with %d neighbors and %s' % (best[0], best[1]))

	clf = KNeighborsClassifier(n_neighbors=best[0], metric=best[1])
	clf.fit(trnX, trnY)
	prd_trn = clf.predict(trnX)
	prd_tst = clf.predict(tstX)
	plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
	directory = 'images/' + prep_filename + "/knn_best"
	if not os.path.exists(directory):
		os.makedirs(directory)

	savefig(directory + "/" + target_class + ".png")
	show()
