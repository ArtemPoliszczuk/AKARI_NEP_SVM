"""User's configuration file."""

import description_class
import ml_engines

"""List of random numbers used for creation of cross validation samples.
Results for i-th random number in list will be stored in i-th output directories."""
rand_numbers = [42]
"""------------------ proper order of classes -------------------------------"""
"""Proper order of labels used in classification.
In the case of binary classification and, for example, ROC AUC score it is 
important to control which class is positive (second one) and which is negative 
(first one)."""
proper_order_of_class_labels = ['other', 'qso']
"""------------------ SVM Specification -------------------------------------"""
svm_descriptor = description_class._Algorithm_Descriptor()
"""Define type of ml algorithm you want to use. In this version only svm is available"""
svm_descriptor.initialize_ml_algorithm("svm")
"""Define additional evaluation metrics. Currently avaliable are:
matthews_corrcoef, cohen_kappa, recall, precision, f1 and roc_auc.
To define new metrics modify evaluation_metrics.get_metric_result()."""
svm_descriptor.set_additional_metrics(['matthews_corrcoef', 'cohen_kappa', 'recall', 'precision'])
"""Define if you want ROC to be calculated (True).
Attention: also prob_estimation should be True."""
svm_descriptor.set_roc_curve_flag(True)
"""Define if you want to use fuzzy logic in the classification.
normal - no fuzzy logic.
errorfuzzy - fuzzy memberships based on measurement uncertainties.
distancefuzzy - fuzzy memberships based on distance from the center
                of the class in the input parameter space.
Definition of fuzzy memberships can be found in weightcreator.py"""
svm_descriptor.set_fuzzy_options(["normal", "errorfuzzy", "distancefuzzy"])
"""Set parameter grid for grid search"""
C_values          = [0.1, 1.]
gamma_values      = [0.1, 1.]
coef0_values      = [0.1, 1.]
polynomialdegrees = [2]

#C_values     = [0.001,0.01,0.1,1,10,100,1000]
#gamma_values = [0.001,0.01,0.1,1,10,100,1000]
#coef0_values = [0.001,0.01,0.1,1,10,100,1000]

"""Set rest of the specification:
main_evaluation_metric - metric maximized in the grid search proces
prob_estimation - if Platt's probability should be estimated (True/False)
classification_type - 'binary' or 'multiple' classification.
n_class_strategy - if case is not binary then 'ovo' - one vs one strategy
                   and 'ovr' - one vs rest strategy.
kernel_names - kernel functions for svm classifier ('rbf', 'sigmoid', 'poly')"""
svm_descriptor.set_algorithm_options_specification({'main_evaluation_metric': "roc_auc",
                                                    'prob_estimation': True,
                                                    'classification_type': "binary",
                                                    'n_class_strategy': ["ovo"],
                                                    'kernel_names': ["rbf", "sigmoid"]})

svm_descriptor.set_parameters_for_grid_search({'C': C_values,
                                               'gamma': gamma_values,
                                               'coef0': coef0_values,
                                               'degree': polynomialdegrees})

"""Number of folds for cross validation."""
svm_descriptor.set_num_of_cv(5)
"""Number of steps in the deep grid search"""
svm_descriptor.set_num_of_deep_gs_loops(10)

"""csv data files with training, validation and generalization data."""
training_data_dict = {'other': "other_test.csv",
                      'qso': "qso_test.csv"}
validation_data_dict = {'other': "other_test.csv",
                        'qso': "qso_test.csv"}
generalization_data_dict = {'main_generalization': "other_test.csv",
                            'weird_objects': "qso_test.csv",
                            'other_generalization': "other_test.csv"}

svm_descriptor.set_training_data_names_dict(training_data_dict)
svm_descriptor.set_validation_data_names_dict(validation_data_dict)
svm_descriptor.set_generalization_data_names_dict(generalization_data_dict)

"""number of features in file.
Data files should be stored in 'data' subdirectory, in the same directory where
the 'core' directory is.
Structure of the file:
First columns should contain information about the object 
such as ra,dec, obj. id etc. Nex columns contain features (colors, fluxes, etc.)
in order:
feature1 uncertainty_of_feature1 feature2 uncertainty_of_feature2 etc."""
svm_descriptor.set_number_of_features(3)
svm_descriptor.set_column_number_of_first_feature(2)
svm_descriptor.set_if_errors_in_datafile(True)
svm_descriptor.set_proper_order_of_labels(proper_order_of_class_labels)

#print(svm_descriptor)
#attrs = vars(svm_descriptor)
#print(',\n '.join("%s: %s" % item for item in attrs.items()))

"""------------------------- Training liftoff -------------------------------"""
ml_engines.ml_liftoff(mode="training",
                      rand_numbers=rand_numbers,
                      algorithm_descriptor=svm_descriptor)
"""------------------------- Validation liftoff -----------------------------"""
ml_engines.ml_liftoff(mode="validation",
                      rand_numbers=rand_numbers,
                      algorithm_descriptor=svm_descriptor)
#"""------------------------- Generalization liftoff -------------------------"""
ml_engines.ml_liftoff(mode="generalization",
                      rand_numbers=rand_numbers,
                      algorithm_descriptor=svm_descriptor)
