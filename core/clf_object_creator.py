import methods_class
import dirman


def clf_object_creator(algorithm_descriptor, class_weight, validation_flag=False, number_of_run=None):
    """Creates all classifiers defined in configuration file."""

    objectlist = []
    algorithm_specification = algorithm_descriptor.algorithm_options_specification

    if validation_flag is False:

        for fuzzy_option in algorithm_descriptor.fuzzy_options:

            if algorithm_descriptor.algorithm_name == "rf":
                obj = \
                    methods_class._Classifier(algorithm_name=algorithm_descriptor.algorithm_name,
                                              fuzzyoption=fuzzy_option,
                                              class_weight=class_weight)
                objectlist.append(obj)

            elif algorithm_descriptor.algorithm_name == "extrarand":
                obj = \
                    methods_class._Classifier(algorithm_name=algorithm_descriptor.algorithm_name,
                                              fuzzyoption=fuzzy_option,
                                              class_weight=class_weight)
                objectlist.append(obj)

            elif algorithm_descriptor.algorithm_name == "svm":
                for kernel in algorithm_specification["kernel_names"]:
                    for nclasstrat in algorithm_specification["n_class_strategy"]:

                        additional_info_dict={}
                        additional_info_dict['kernel_name'] = kernel
                        additional_info_dict['n_class_strategy'] = nclasstrat
                        additional_info_dict['prob_estimation'] = algorithm_specification["prob_estimation"]

                        obj = \
                            methods_class._Classifier(algorithm_name=algorithm_descriptor.algorithm_name,
                                                      fuzzyoption=fuzzy_option,
                                                      additional_info_dict=additional_info_dict,
                                                      class_weight=class_weight)
                        objectlist.append(obj)

    else:

        best_df = get_df_of_best_results(algorithm_descriptor, number_of_run)

        for index, row in best_df.iterrows():
            char_clf = row['classifier'].split('_')
            additional_info_dict, fuzzy_option = get_additional_info_dict(char_clf, algorithm_descriptor)
            obj = \
                methods_class._Classifier(algorithm_name=algorithm_descriptor.algorithm_name,
                                          fuzzyoption=fuzzy_option,
                                          additional_info_dict=additional_info_dict,
                                          class_weight=class_weight)
            best_params_dict  = obj.get_params_from_df(row)
            obj.exact_parameters(best_params_dict)
            objectlist.append(obj)

    return objectlist


def get_df_of_best_results(algorithm_descriptor, number_of_run):
    """Loads .csv file with training results as pandas data frame
    and returns best parameters and results.
    Used for validation and generalization."""
    training_output_path = dirman.OutputMainDirCreator()
    filename = str(number_of_run) + "_" + algorithm_descriptor.algorithm_name + "_gridsearch_performance.csv"
    gs_df = dirman.LoadDataPandas(training_output_path, filename)
    best_result_df = gs_df.loc[gs_df['gs_num'] == algorithm_descriptor.num_of_deep_gs_loops]

    return best_result_df


def get_additional_info_dict(char_clf, algorithm_descriptor):
    """Additional info for validation used in clf_object_creator()"""
    additional_info_dict = {}
    fuzzy_option = None

    clf_type = char_clf[0]

    if clf_type == "svm":

        kernel_name = char_clf[1]
        fuzzy_option = char_clf[2]
        nclasstrat = char_clf[3]

        additional_info_dict['kernel_name'] = kernel_name
        additional_info_dict['n_class_strategy'] = nclasstrat
        additional_info_dict['prob_estimation'] = \
            algorithm_descriptor.algorithm_options_specification["prob_estimation"]

    elif clf_type == "rf":
        fuzzy_option = char_clf[1]
        additional_info_dict = {}

    elif clf_type == "extrarand":
        fuzzy_option = char_clf[1]
        additional_info_dict = {}

    return additional_info_dict, fuzzy_option
