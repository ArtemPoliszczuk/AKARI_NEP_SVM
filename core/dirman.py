import os
import pandas as pd


def OutputMainDirCreator(catalog_name=None):
    """Creates subdirectory with output data (if doesn't exist yet)
       returns its address."""
    core_dir_path = os.path.dirname(os.path.abspath(__file__))
    split_tuple = os.path.split(core_dir_path)
    main_dir_path = split_tuple[0]

    if catalog_name:
        output_path = os.path.join(main_dir_path, catalog_name)
    else:
        output_path = os.path.join(main_dir_path, 'training_output')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path


def ithOutputDirCreator(main_output_path,additional_name):
    """creates subdirectories for particular runs (for particular
    random numbers used in CV) in the main output catalog and returns
    the address."""
    newdirname = additional_name+"_output"
    ith_newpath = os.path.join(main_output_path, newdirname)

    if not os.path.exists(ith_newpath):
        os.makedirs(ith_newpath)

    return ith_newpath


def LoadDataPandas(dirname, filename, return_df=True):
    """Load csv data. """
    core_dir_path = os.path.dirname(os.path.abspath(__file__))
    split_tuple = os.path.split(core_dir_path)
    main_dir_path = split_tuple[0]

    data_path = os.path.join(main_dir_path, dirname)
    os.chdir(data_path)

    loaded_data = pd.read_csv(filename)

    if return_df is False:
        return loaded_data.values  # returns  numpy array
    else:
        return loaded_data  # returns pandas data frame


def load_data(algorithm_descriptor, mode="training", return_pandas_df=True):

    output_dict = {}

    if mode == "training":
        filenames_dict = algorithm_descriptor.training_data_names_dict
    elif mode == "validation":
        filenames_dict = algorithm_descriptor.validation_data_names_dict
    elif mode == "generalization":
        filenames_dict = algorithm_descriptor.generalization_data_names_dict
    else:
        filenames_dict = None
        print("Wrong mode. (dirman.py)")

    for key in filenames_dict.keys():
        output_dict[key] = LoadDataPandas("data", filenames_dict[key], return_pandas_df)

    return output_dict
