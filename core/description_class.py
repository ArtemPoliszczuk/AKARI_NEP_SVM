
class _Algorithm_Descriptor(object):
    """Class which contains information about particular type of ML algorithm.
    In current version only SVM algorithm is used."""

    def __init__(self):

        self.algorithm_name = "none"
        self.algorithm_options_specification = {}

        self.additional_metrics = []
        self.roc_curve_flag = True

        self.fuzzy_options = ["normal"]
        self.errorfuzzy_flag = False
        self.distancefuzzy_flag = False

        self.parameters_for_grid_search = {}

        self.train_sample_ratio = 1.0
        self.num_of_cores = -1

        self.num_of_cv = 10
        self.num_of_deep_gs_loops = 5

        self.training_data_names_dict = {}
        self.validation_data_names_dict = {}
        self.generalization_data_names_dict = {}

        self.number_of_features = 3
        self.column_number_of_first_feature = 2
        self.if_errors_in_datafile = True
        self.proper_order_of_labels = []

    def initialize_ml_algorithm(self, algorithm_name_str):

        self.algorithm_name = algorithm_name_str

        if self.algorithm_name == "svm":
            print('SVM algorithm initialized.')
            self.algorithm_options_specification = {'main_evaluation_metric': "accuracy",
                                                    'prob_estimation': True,
                                                    'classification_type': "binary",
                                                    'n_class_strategy': ["ovo"],
                                                    'kernel_names': ["rbf"]}
            self.parameters_for_grid_search = {"C": 1.0,
                                               "gamma": "scale"}

        elif self.algorithm_name == "rf":
            print('Random Forest algorithm initialized.')
            self.parameters_for_grid_search = {"n_estimators": 200,
                                               "max_features": "auto"}

        elif self.algorithm_name == "extrarand":
            print('Extremely Randomized Trees algorithm initialized.')
            self.parameters_for_grid_search = {"n_estimators": 200,
                                               "max_features": "auto"}

        else:
            print("Error. Unknown type of algorithm. (description_class.py")

    def set_additional_metrics(self, additional_metrist_list):
        self.additional_metrics = additional_metrist_list

    def set_roc_curve_flag(self, roc_curve_flag):
        self.roc_curve_flag = roc_curve_flag

    def set_fuzzy_options(self, fuzzy_options):
        self.fuzzy_options = fuzzy_options

        if 'errorfuzzy' not in self.fuzzy_options:
            print('no error fuzzy')
            self.errorfuzzy_flag = False
        elif 'errorfuzzy' in self.fuzzy_options:
            print('error fuzzy flag in On')
            self.errorfuzzy_flag = True

        if 'distancefuzzy' not in self.fuzzy_options:
            print('no distance fuzzy')
            self.distancefuzzy_flag = False
        elif 'distancefuzzy' in self.fuzzy_options:
            print('distance fuzzy flag is On')
            self.distancefuzzy_flag = True

    def set_algorithm_options_specification(self, algorithm_options_specification):
        self.algorithm_options_specification = algorithm_options_specification

    def set_parameters_for_grid_search(self, parameters_for_grid_search_dict):
        self.parameters_for_grid_search = parameters_for_grid_search_dict

    def set_train_sample_ratio(self, train_sample_ratio):
        self.train_sample_ratio = train_sample_ratio

    def set_num_of_cores(self, num_of_cores):
        self.num_of_cores = num_of_cores

    def set_num_of_cv(self, num_of_cv):
        self.num_of_cv = num_of_cv

    def set_num_of_deep_gs_loops(self, num_of_deep_gs_loops):
        self.num_of_deep_gs_loops = num_of_deep_gs_loops

    def set_training_data_names_dict(self, training_data_names_dict):
        self.training_data_names_dict = training_data_names_dict

    def set_validation_data_names_dict(self, validation_data_names_dict):
        self.validation_data_names_dict = validation_data_names_dict

    def set_generalization_data_names_dict(self, generalization_data_names_dict):
        self.generalization_data_names_dict = generalization_data_names_dict

    def set_number_of_features(self, number_of_features):
        self.number_of_features = number_of_features

    def set_column_number_of_first_feature(self, column_number_of_first_feature):
        self.column_number_of_first_feature = column_number_of_first_feature

    def set_if_errors_in_datafile(self, if_errors_in_datafile_flag):
        self.if_errors_in_datafile = if_errors_in_datafile_flag

    def set_proper_order_of_labels(self, proper_order_of_labels_list):
        self.proper_order_of_labels = proper_order_of_labels_list
