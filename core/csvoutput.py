# Different functions for result output.

import os.path
import os
import dirman


def grid_search_csv(i, performance_df, algorithm_descriptor):

    dir_path = dirman.OutputMainDirCreator()
    filename = i + "_" + algorithm_descriptor.algorithm_name + "_gridsearch_performance.csv"
    filename = os.path.join(dir_path, filename)

    performance_df.to_csv(filename, index_label="gs_num")


def additional_metrics_csv(ith_dir_path, additional_metrics_df):

    filename = "additional_metrics.csv"

    filename = os.path.join(ith_dir_path, filename)
    additional_metrics_df.to_csv(filename, index=False)


def roc_curve_csv(ith_dir_path, clf_object, mode):

    filename = "ROC_" + mode + "_" + clf_object.get_name_str() + ".csv"

    filename = os.path.join(ith_dir_path, filename)
    clf_object.roc_curve_df.to_csv(filename, index=False)


def prediction_csv(ith_dir_path, clf_object, mode, generalization_sample_name=None):

    if mode != "generalization":
        filename = "prediction_for_" + mode + "_set_" + \
                   clf_object.get_name_str() + ".csv"
    else:
        filename = "prediction_for_" + generalization_sample_name + "_" + \
                   clf_object.get_name_str() + ".csv"
    filename = os.path.join(ith_dir_path, filename)

    clf_object.prediction_df.to_csv(filename, index=False)
