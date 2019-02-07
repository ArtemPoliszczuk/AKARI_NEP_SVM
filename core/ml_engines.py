import pandas as pd

import listcreator
import dirman
import clf_object_creator as clf_creator
import csvoutput
import gridsearch_sklearn
import validation

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def ml_liftoff(mode, rand_numbers, algorithm_descriptor):

    print("\n",mode, " start.")

    main_output_path = dirman.OutputMainDirCreator(catalog_name=mode + "_output")

    # Load training (and validation) data from csv files.
    training_data_dict = dirman.load_data(algorithm_descriptor, mode="training")
    if mode == "validation":
        validation_data_dict = dirman.load_data(algorithm_descriptor, mode="validation")

    for i, rand_num in enumerate(rand_numbers):
        print("run: ", i)

        if mode == "training":

            ith_dir_path = dirman.ithOutputDirCreator(main_output_path,
                            algorithm_descriptor.algorithm_name + '_'+str(i))

            lstcr_dict = listcreator.ListCreator(algorithm_descriptor, rand_num,
                                      training_dict=training_data_dict)
            clflist = clf_creator.clf_object_creator(algorithm_descriptor,
                                                     lstcr_dict['class_weight_dict'])

        elif mode == "validation":

            ith_dir_path = dirman.ithOutputDirCreator(main_output_path,
                                algorithm_descriptor.algorithm_name + \
                                '_' + str(i) + '_validation_')
            lstcr_dict = listcreator.ListCreator(algorithm_descriptor, rand_num,
                                                 training_dict=training_data_dict,
                                                 validation_dict=validation_data_dict)
            clflist = clf_creator.clf_object_creator(algorithm_descriptor,
                                                     lstcr_dict['class_weight_dict'],
                                                     validation_flag=True,
                                                     number_of_run=i)

        if mode is not "generalization":
            get_results(ith_dir_path, clflist, lstcr_dict,
                        algorithm_descriptor, mode=mode, i=str(i))

        else:
            ith_dir_path = dirman.ithOutputDirCreator(main_output_path,
                                        algorithm_descriptor.algorithm_name + \
                                        '_' + str(i) + '_generalization_')

            gen_names_dict = algorithm_descriptor.generalization_data_names_dict

            for generalization_name in gen_names_dict.keys():
                print("\ngeneralization file: ", generalization_name)

                generalization_sample = dirman.LoadDataPandas("data",
                                                  gen_names_dict[generalization_name],
                                                  return_df=True)

                lstcr_dict = listcreator.ListCreator(algorithm_descriptor, rand_num,
                                                 training_dict=training_data_dict,
                                                 generalization_df=generalization_sample)

                clflist = clf_creator.clf_object_creator(algorithm_descriptor,
                                                     lstcr_dict['class_weight_dict'],
                                                     validation_flag=True,
                                                     number_of_run=i)

                get_results(ith_dir_path, clflist, lstcr_dict,
                            algorithm_descriptor, mode=mode, i=str(i),
                            generalization_sample_name=generalization_name)


def get_results(ith_dir_path, clflist, lstcr_dict, algorithm_descriptor, mode, i=None, generalization_sample_name=None):

    if mode == "training":
        grid_search_info_df = pd.DataFrame()
        additional_metrics_df = pd.DataFrame() # additional metrics calculated on last GS step.

    elif mode == "validation":
        additional_metrics_df = pd.DataFrame()

    while len(clflist)>0:

        print(mode,': ', clflist[-1].get_name_str())

        if mode == "training":
            gridsearch_sklearn.run_gridsearch(clflist[-1], lstcr_dict,
                                              algorithm_descriptor)
            grid_search_info_df = \
                pd.concat([grid_search_info_df, clflist[-1].gridsearch_df],
                          sort=True, join='outer')
            additional_metrics_df = \
                additional_metrics_df.append(clflist[-1].additional_metrics_sr,
                                             ignore_index=True)

            if algorithm_descriptor.roc_curve_flag is True:
                csvoutput.roc_curve_csv(ith_dir_path, clflist[-1], mode)

        elif mode == "validation":

            validation.run_validation(clflist[-1], lstcr_dict, algorithm_descriptor)
            csvoutput.prediction_csv(ith_dir_path, clflist[-1], mode)
            additional_metrics_df = \
                additional_metrics_df.append(clflist[-1].additional_metrics_sr,
                                             ignore_index=True)
            if algorithm_descriptor.roc_curve_flag is True:
                csvoutput.roc_curve_csv(ith_dir_path, clflist[-1], mode)

        elif mode == "generalization":
            validation.run_generalization(clflist[-1], lstcr_dict, algorithm_descriptor)
            csvoutput.prediction_csv(ith_dir_path, clflist[-1], mode,
                                     generalization_sample_name=generalization_sample_name)

        clflist.pop()

    if mode == "training":
        csvoutput.grid_search_csv(i, grid_search_info_df, algorithm_descriptor)
        csvoutput.additional_metrics_csv(ith_dir_path, additional_metrics_df)

    elif mode == "validation":
        csvoutput.additional_metrics_csv(ith_dir_path, additional_metrics_df)

