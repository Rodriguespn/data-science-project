from data_profiling import data_dimensionality, data_distribution, data_granularity, data_sparsity
from data_scaling_knn import data_scaling, data_knn
from utils.util import DATA_FILES_IDS, set_datafile


def main():
    # for f in data_files:
    # set_datafile(DATA_FILES_IDS[0]['tabular'])
    # data_dimensionality()
    # data_distribution()
    # data_granularity()
    # data_sparsity()
    # data_scaling(DATA_FILES_IDS[1]['tabular'] + "_mv")
    data_knn(DATA_FILES_IDS[1]['tabular'] + "_mv_scaled_zscore", "ALARM_Safe")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
