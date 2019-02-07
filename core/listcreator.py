import numpy as np
import pandas as pd
from sklearn import preprocessing

import cross_validation
import weightcreator


def ListCreator(algorithm_descriptor, rand_num, training_dict, validation_dict=None, generalization_df=pd.DataFrame()):
    if algorithm_descriptor.train_sample_ratio < 1.0:
        print("With master test")
        output_dict = ListCreatorWithMasterTest(algorithm_descriptor, training_dict)
    else:
        print("No master test")
        output_dict = DataProcessing(algorithm_descriptor, rand_num, training_dict, validation_dict, generalization_df)
    return output_dict


def ListCreatorWithMasterTest(data_dict, algorithm_descript):
    
    None


def DataProcessing(algorithm_descriptor, rand_num, training_dict, validation_dict=None, generalization_df=None):

    if validation_dict:  # if validation_dict exist:
        validation_flag = True
    else:
        validation_flag = False

    if generalization_df.empty is False:
        generalization_flag = True
    else:
        generalization_flag = False

    output_dict = {}

    # Getting column numbers (which one is feature, uncertainy or
    # additional information about object such as ra,dec, id (other_indices)
    if algorithm_descriptor.if_errors_in_datafile is True:
        other_indices, feature_indices, e_feature_indices = \
            get_feature_indices(algorithm_descriptor, with_errors=True)
    else:
        # not finished.
        other_indices, feature_indices, e_feature_indices = 0, 0, 0
    print('proper order of keys: ', algorithm_descriptor.proper_order_of_labels)
    
    train_data = pd.DataFrame()
    if validation_flag is True:
        test_data = pd.DataFrame()

    class_sizes = []
    
    for i, key in enumerate(algorithm_descriptor.proper_order_of_labels):

        class_sizes.append(len(training_dict[key]))

        training_dict[key]['Y'] = i  # add column with label

        if algorithm_descriptor.distancefuzzy_flag is True:
            training_dict[key]['dist_weights'] = \
                weightcreator.OneClassDistanceWeights(training_dict[key].iloc[:, feature_indices])
        if algorithm_descriptor.errorfuzzy_flag is True:
            training_dict[key]['err_weights'] = \
                weightcreator.OneClassErrorWeights(training_dict[key].iloc[:, e_feature_indices])
        
        if train_data.empty is False:  # if there are already elements in train_data
            train_data = train_data.append(training_dict[key])  # append current df to the train_data
        else:  # train_data is empty yet and this is its first element
            train_data = training_dict[key]  # initialize train_data
        if validation_flag is True:

            validation_dict[key]['Y'] = i

            if test_data.empty is False:
                test_data = test_data.append(validation_dict[key])
            else:
                test_data = validation_dict[key]

    train_data.reset_index(drop=True, inplace=True)  # new indexing for data as a whole.
    """------ scale data ------"""
    scaler = preprocessing.StandardScaler().fit(train_data.iloc[:, feature_indices])
    trainX_rescaled = scaler.transform(train_data.iloc[:, feature_indices])  # rescale data
    trainX_rescaled = np.ascontiguousarray(trainX_rescaled)  # make it C-contiguous
    trainY = train_data['Y']

    if validation_flag is True:
        test_data.reset_index(drop=True, inplace=True)
        testX_rescaled = scaler.transform(test_data.iloc[:, feature_indices])
        testX_rescaled = np.ascontiguousarray(testX_rescaled)
        testY = test_data['Y']

    if generalization_flag is True:
        generalizationX_rescaled = scaler.transform(generalization_df.iloc[:, feature_indices])
        generalizationX_rescaled = np.ascontiguousarray(generalizationX_rescaled)

    # ----- class weights -------
    class_weight_dict = {}

    max_size = float(max(class_sizes))
    for i, key in enumerate(algorithm_descriptor.proper_order_of_labels):
        class_weight = round(max_size/class_sizes[i], 1)
        class_weight_dict[i] = class_weight
    print('class weights: ', class_weight_dict)
    # ---------------------------
    output_dict["class_weight_dict"] = class_weight_dict
    output_dict['X_Train'] = trainX_rescaled
    output_dict['y_Train'] = trainY
    if algorithm_descriptor.distancefuzzy_flag is True:
        output_dict['dist_weights'] = train_data['dist_weights']
    if algorithm_descriptor.errorfuzzy_flag is True:
        output_dict['err_weights'] = train_data['err_weights']

    if (validation_flag is False) and (generalization_flag is False):  # if it's training only
        output_dict['cv_indices'] = \
            cross_validation.stratifiedkfold_indices(trainX_rescaled, trainY, algorithm_descriptor, rand_num)
    elif validation_flag is True:
        output_dict["other_info_df"] = test_data.iloc[:, other_indices]
        output_dict["X_Test"] = testX_rescaled
        output_dict["y_Test"] = testY
        output_dict['test_cv_indices'] = \
            cross_validation.stratifiedkfold_indices(testX_rescaled,
                                                     testY, algorithm_descriptor, rand_num,
                                                     only_test_indices=True)
    elif generalization_flag is True:
        output_dict["other_info_df"] = generalization_df.iloc[:, other_indices]
        output_dict["X_Generalization"] = generalizationX_rescaled

    return output_dict


def get_feature_indices(algorithm_descriptor, with_errors=True):

    other_indices = np.arange(algorithm_descriptor.column_number_of_first_feature, dtype=int)
    print("other indices: ", other_indices)

    if with_errors:  # dataset construction: feature1 e_feature1 feature2 etc.
        # column numbers (which is feature, which is feature error)
        feature_i = algorithm_descriptor.column_number_of_first_feature
        e_feature_i = feature_i + 1
        feature_indices = [feature_i]
        e_feature_indices = [e_feature_i]
        while len(feature_indices) < algorithm_descriptor.number_of_features:
            feature_i += 2
            e_feature_i += 2
            feature_indices.append(feature_i)
            e_feature_indices.append(e_feature_i)
        return other_indices, feature_indices, e_feature_indices
    
    elif with_errors is False:
        # no errors in the data, only features
        feature_i = algorithm_descriptor.column_number_of_first_feature
        feature_indices = [feature_i]
        while len(feature_indices) < algorithm_descriptor.number_of_feature:
            feature_i += 1
            feature_indices.append(feature_i)
        return other_indices, feature_indices
