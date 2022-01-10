import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, argsort
from pandas import DataFrame, read_csv, unique, concat
from matplotlib.pyplot import figure, savefig, show, subplots
from sklearn.neighbors import KNeighborsClassifier
from utils.ds_charts import plot_evaluation_results, multiple_line_chart, get_variable_types, bar_chart, horizontal_bar_chart
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from imblearn.over_sampling import SMOTE


DATA_FOLDER = "data/"

DATA_PREP_FOLDER = DATA_FOLDER + "data-prep/"
DATA_FILE_UNSCALED = DATA_PREP_FOLDER + "dummification.csv"
DATA_FILE_MINMAX = DATA_PREP_FOLDER + "scaled_minmax.csv"
DATA_FILE_ZSCORE = DATA_PREP_FOLDER + "scaled_z_score.csv"

DATA_TRAIN_FOLDER = DATA_FOLDER + "data-train/"
DATA_TRAIN_UNSCALED = DATA_TRAIN_FOLDER + "unscaled.csv"
DATA_TRAIN_MINXMAX = DATA_TRAIN_FOLDER + "scaled_minmax.csv"
DATA_TRAIN_ZSCORE = DATA_TRAIN_FOLDER + "scaled_z_score.csv"
DATA_TRAIN_UNDERSAMPLING = DATA_TRAIN_FOLDER + "balanced_undersampling.csv"
DATA_TRAIN_OVERSAMPLING = DATA_TRAIN_FOLDER + "balanced_oversampling.csv"
DATA_TRAIN_SMOTE = DATA_TRAIN_FOLDER + "balanced_smote.csv"

DATA_TEST_FOLDER = DATA_FOLDER + "data-test/"
DATA_TEST_UNSCALED = DATA_TEST_FOLDER + "unscaled.csv"
DATA_TEST_MINXMAX = DATA_TEST_FOLDER + "scaled_minmax.csv"
DATA_TEST_ZSCORE = DATA_TEST_FOLDER + "scaled_z_score.csv"

IMAGES_FOLDER = "images/"
KNN_FOLDER = IMAGES_FOLDER + "knn/"
NAIVE_BAYES_FOLDER = IMAGES_FOLDER + "naive_bayes/"
DECISION_TREES_FOLDER = IMAGES_FOLDER + "decision_trees/"

TARGET_CLASS = 'ALARM'

if not os.path.exists(DATA_TRAIN_FOLDER):
    os.makedirs(DATA_TRAIN_FOLDER)
if not os.path.exists(DATA_TEST_FOLDER):
    os.makedirs(DATA_TEST_FOLDER)
if not os.path.exists(KNN_FOLDER):
    os.makedirs(KNN_FOLDER)
if not os.path.exists(NAIVE_BAYES_FOLDER):
    os.makedirs(NAIVE_BAYES_FOLDER)
if not os.path.exists(DECISION_TREES_FOLDER):
    os.makedirs(DECISION_TREES_FOLDER)


def write_to_file(file_name, text):
    file = open(file_name, "w")
    print(text)
    file.write(text + "\n")
    file.close()


def split_data(data_file, train_file, test_file):
    print("-------------------------------------------")
    print("Starting spliting data of {} file".format(data_file))

    data: DataFrame = read_csv(data_file)

    symbolic_vars = get_variable_types(data)['Symbolic']
    if symbolic_vars:
        label_encoder = LabelEncoder()
        for var in symbolic_vars:
            data[var] = label_encoder.fit_transform(data[var])

    y: ndarray = data.pop(TARGET_CLASS).values
    x: ndarray = data.values
    labels: np.ndarray = unique(y)
    labels.sort()

    trnX, tstX, trnY, tstY = train_test_split(x, y, train_size=0.7, stratify=y)

    train = concat([DataFrame(trnX, columns=data.columns),
                   DataFrame(trnY, columns=[TARGET_CLASS])], axis=1)
    train.to_csv(train_file, index=False)
    print("Train data file {} saved".format(train_file))

    test = concat([DataFrame(tstX, columns=data.columns),
                  DataFrame(tstY, columns=[TARGET_CLASS])], axis=1)
    test.to_csv(test_file, index=False)
    print("Test data file {} saved".format(train_file))

    print("Finished spliting data of {} file".format(data_file))
    print("-------------------------------------------")


def balance(data_train_file, undersampling_file, oversampling_file, smote_file):
    print("-------------------------------------------")
    print("Starting balancing data of {} file".format(data_train_file))

    data: DataFrame = read_csv(data_train_file)
    target_count = data[TARGET_CLASS].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    # undersampling
    df_positives = data[data[TARGET_CLASS] == positive_class]
    df_negatives = data[data[TARGET_CLASS] == negative_class]
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    df_under = concat([df_positives, df_neg_sample], axis=0)
    df_under.to_csv(undersampling_file, index=False)
    print("Undersampling file {} saved".format(undersampling_file))

    # oversampling
    df_pos_sample = DataFrame(df_positives.sample(
        len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    df_over.to_csv(oversampling_file, index=False)
    print("Oversampling file {} saved".format(oversampling_file))

    # smote

    RANDOM_STATE = 42
    df_copy = data.copy(deep=True)
    symbolic_vars = get_variable_types(data)['Symbolic']
    if symbolic_vars:
        label_encoder = LabelEncoder()
        for var in symbolic_vars:
            df_copy[var] = label_encoder.fit_transform(df_copy[var])
    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = df_copy.pop(TARGET_CLASS).values
    X = df_copy.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(df_copy.columns) + [TARGET_CLASS]
    df_smote.to_csv(smote_file, index=False)
    print("SMOTE file {} saved".format(smote_file))
    print("Finished balancing data of {} file".format(data_train_file))
    print("-------------------------------------------")


def knn_study(data_train_file, data_test_file, files_name, metrics=[], n_neighbors=[]):
    print("-------------------------------------------")
    print("Starting knn study of {} and {} files - {}".format(data_train_file,
          data_test_file, files_name))

    train: DataFrame = read_csv(data_train_file)
    trnY: ndarray = train.pop(TARGET_CLASS).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(data_test_file)
    tstY: ndarray = test.pop(TARGET_CLASS).values
    tstX: ndarray = test.values

    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17,
               19] if not n_neighbors else n_neighbors
    dist = ['manhattan', 'euclidean',
            'chebyshev'] if not metrics else metrics

    values = {}
    best = (0, '')
    last_best = 0
    min_y = 1
    for d in dist:
        yvalues = []
        for n in nvalues:
            start = time.perf_counter()

            knn = KNeighborsClassifier(
                n_neighbors=n, metric=d, algorithm='kd_tree', n_jobs=-1)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (n, d)
                last_best = yvalues[-1]
            if yvalues[-1] < min_y:
                min_y = yvalues[-1]

            print("--- KNN run time params=(d={}, n={}) = {} seconds ---".format(d,
                  n, time.perf_counter() - start))
        values[d] = yvalues

    figure()

    ax = plt.gca()
    # space between the min value and the x-axis is 10% of graph
    min_y = max(0, (10 * min_y - 1) / 9)
    ax.set_ylim([min_y, 1])

    multiple_line_chart(nvalues, values, title='KNN variants',
                        xlabel='n', ylabel='accuracy', percentage=False, ax=ax)
    savefig(KNN_FOLDER + "knn_study_" + files_name + ".png")
    show()
    write_to_file(KNN_FOLDER + "knn_best_" + files_name + ".txt",
                  'Best results with %d neighbors and %s ==> accuracy=%1.2f' % (best[0], best[1], last_best))

    clf = KNeighborsClassifier(
        n_neighbors=best[0], metric=best[1], algorithm='kd_tree', n_jobs=-1)
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)

    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(KNN_FOLDER + "knn_evaluation_" + files_name + ".png")
    show()

    print("Finished knn study of {} and {} files - {}".format(data_train_file,
          data_test_file, files_name))
    print("-------------------------------------------")


def naive_bayes_study(data_train_file, data_test_file, files_name):
    print("-------------------------------------------")
    print("Starting naive bayes study of {} and {} files - {}".format(
        data_train_file, data_test_file, files_name))
    train: DataFrame = read_csv(data_train_file)
    trnY: ndarray = train.pop(TARGET_CLASS).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(data_test_file)
    tstY: ndarray = test.pop(TARGET_CLASS).values
    tstX: ndarray = test.values

    estimators = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }

    xvalues = []
    yvalues = []
    best = ''
    last_best = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(accuracy_score(tstY, prdY))
        if yvalues[-1] > last_best:
            best = clf
            last_best = yvalues[-1]

    figure()
    bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models',
              ylabel='accuracy', percentage=True)
    savefig(NAIVE_BAYES_FOLDER + "naive_bayes_study_" + files_name + ".png")
    show()
    write_to_file(NAIVE_BAYES_FOLDER + "naive_bayes_best_" +
                  files_name + ".txt", 'Best results with %s ==> accuracy=%1.2f' % (best, last_best))

    clf = estimators[best]
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(NAIVE_BAYES_FOLDER + "naive_bayes_evaluation_" + files_name + ".png")
    show()

    print("Finisehd naive bayes study of {} and {} files - {}".format(
        data_train_file, data_test_file, files_name))
    print("-------------------------------------------")


def decision_trees_study(data_train_file, data_test_file, files_name):
    print("-------------------------------------------")
    print("Starting decision tree study of {} and {} files - {}".format(
        data_train_file, data_test_file, files_name))
    train: DataFrame = read_csv(data_train_file)
    trnY: ndarray = train.pop(TARGET_CLASS).values
    trnX: ndarray = train.values
    labels = unique(trnY)
    labels.sort()

    test: DataFrame = read_csv(data_test_file)
    tstY: ndarray = test.pop(TARGET_CLASS).values
    tstX: ndarray = test.values

    min_impurity_decrease = [0.01, 0.005, 0.0025, 0.001, 0.0005]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('', 0, 0.0)
    last_best = 0
    best_model = None

    min_y = 1
    figure()
    fig, axs = subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(
                    max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_model = tree
                if yvalues[-1] < min_y:
                    min_y = yvalues[-1]

            values[d] = yvalues

        # space between the min value and the x-axis is 10% of graph
        min_y = max(0, (10 * min_y - 1) / 9)
        axs[0, k].set_ylim([min_y, 1])

        multiple_line_chart(min_impurity_decrease, values,
                            ax=axs[0, k], title=f'Decision Trees with {f} criteria', xlabel='min_impurity_decrease', ylabel='accuracy', percentage=False)

    savefig(DECISION_TREES_FOLDER +
            "decision_trees_study_" + files_name + ".png")
    show()
    write_to_file(DECISION_TREES_FOLDER + "naive_bayes_best_" + files_name + ".txt",
                  'Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.2f ==> accuracy=%1.2f' % (best[0], best[1], best[2], last_best))

    labels = [str(value) for value in labels]
    plot_tree(best_model, feature_names=train.columns, class_names=labels)
    savefig(DECISION_TREES_FOLDER +
            "decision_trees_best_tree_" + files_name + ".png")
    show()

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)
    plot_evaluation_results(labels, trnY, prd_trn, tstY, prd_tst)
    savefig(DECISION_TREES_FOLDER +
            "decision_trees_evaluation_" + files_name + ".png")
    show()

    variables = train.columns
    importances = best_model.feature_importances_
    indices = argsort(importances)[::-1]
    elems = []
    imp_values = []
    for f in range(len(variables)):
        elems += [variables[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f'{f + 1}. feature {elems[f]} ({importances[indices[f]]})')

    figure()
    horizontal_bar_chart(elems, imp_values, error=None,
                         title='Decision Tree Features importance', xlabel='importance', ylabel='variables')
    savefig(DECISION_TREES_FOLDER +
            "decision_trees_ranking_" + files_name + ".png")
    show()

    print("Finished decision tree study of {} and {} files - {}".format(
        data_train_file, data_test_file, files_name))
    print("-------------------------------------------")


if __name__ == "__main__":
    import time

    start = time.perf_counter()

    split_data(DATA_FILE_UNSCALED, DATA_TRAIN_UNSCALED, DATA_TEST_UNSCALED)
    split_data(DATA_FILE_MINMAX, DATA_TRAIN_MINXMAX, DATA_TEST_MINXMAX)
    split_data(DATA_FILE_ZSCORE, DATA_TRAIN_ZSCORE, DATA_TEST_ZSCORE)

    print("--- Split data run time = {} seconds ---".format(time.perf_counter() - start))

    start = time.perf_counter()

    knn_study(DATA_TRAIN_UNSCALED, DATA_TEST_UNSCALED, "unscaled")
    knn_study(DATA_TRAIN_MINXMAX, DATA_TEST_MINXMAX, "minmax")
    knn_study(DATA_TRAIN_ZSCORE, DATA_TEST_ZSCORE, "z_score")

    print("--- Unbalanced KNN run time = {} seconds ---".format(time.perf_counter() - start))

    # start = time.perf_counter()

    # balance(DATA_TRAIN_UNSCALED, DATA_TRAIN_UNDERSAMPLING,
    #         DATA_TRAIN_OVERSAMPLING, DATA_TRAIN_SMOTE)

    # knn_study(DATA_TRAIN_UNDERSAMPLING, DATA_TEST_UNSCALED, "undersampling")
    # knn_study(DATA_TRAIN_OVERSAMPLING, DATA_TEST_UNSCALED, "oversampling")
    # knn_study(DATA_TRAIN_SMOTE, DATA_TEST_UNSCALED, "smote")

    # print("--- Balanced KNN run time = {} seconds ---".format(time.perf_counter() - start))

    # start = time.perf_counter()

    # naive_bayes_study(DATA_TRAIN_UNSCALED, DATA_TEST_UNSCALED, "unscaled")
    # naive_bayes_study(DATA_TRAIN_UNDERSAMPLING,
    #                   DATA_TEST_UNSCALED, "undersampling")
    # naive_bayes_study(DATA_TRAIN_OVERSAMPLING,
    #                   DATA_TEST_UNSCALED, "oversampling")
    # naive_bayes_study(DATA_TRAIN_SMOTE, DATA_TEST_UNSCALED, "smote")

    # print("--- Naive Bayes time = {} seconds ---".format(time.perf_counter() - start))

    # start = time.perf_counter()

    # decision_trees_study(DATA_TRAIN_UNSCALED, DATA_TEST_UNSCALED, "unscaled")

    # print("--- Decision tree run time = {} seconds ---".format(time.perf_counter() - start))
