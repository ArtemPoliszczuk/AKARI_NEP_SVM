import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm


def set_classifier(algorithm_name, additional_info_dict, class_weight):
    if algorithm_name == "rf":
        clf = ensemble_method(algorithm_name, class_weight)
    elif algorithm_name == "extrarand":
        clf = ensemble_method(algorithm_name, class_weight)
    elif algorithm_name == "svm":
        clf = svm_kernel_method(additional_info_dict, class_weight)

    return clf


def ensemble_method(algorithm_name, class_weight):

    if algorithm_name == "rf":
        clf = RandomForestClassifier(criterion='gini',
                                     class_weight=class_weight,
                                     n_jobs=-1,
                                     random_state=42)
    elif algorithm_name == "extrarand":
        clf = ExtraTreesClassifier(criterion='gini',
                                   class_weight=class_weight,
                                   n_jobs=-1,
                                   random_state=42)

    else:
        clf = None
        print("Error: wrong name (ensemble_methods_class.py)")

    return clf


def svm_kernel_method(additional_info_dict, class_weight):
    if "kernel_name" in additional_info_dict.keys():
        kernel_name = additional_info_dict["kernel_name"]
    else:
        print("no kernel name for svm. rbf was used.")
        kernel_name = "rbf"

    if "n_class_strategy" in additional_info_dict.keys():
        n_class_strategy = additional_info_dict["n_class_strategy"]
    else:
        print("no nclass strategy for svm. ovo was used.")
        n_class_strategy = "ovo"

    if "prob_estimation" in additional_info_dict.keys():
        prob_estimation = additional_info_dict["prob_estimation"]
    else:
        print("no probability flag for svm. False was used.")
        prob_estimation = False

    svm_svc_clf = svm.SVC(gamma='scale', kernel=kernel_name, probability=prob_estimation,
                          class_weight=class_weight, decision_function_shape=n_class_strategy,
                          cache_size=2000, random_state=42)

    return svm_svc_clf


def set_additional_info(algorithm_name, additional_info_dict):
    additional_info = {}

    if algorithm_name == "rf":
        pass
    elif algorithm_name == "extrarand":
        pass
    elif algorithm_name == "svm":
        additional_info['kernel_name'] = additional_info_dict['kernel_name']
        additional_info['n_class_strategy'] = additional_info_dict['n_class_strategy']

    return additional_info


def set_parameter_names(algorithm_name, additional_info_dict):
    parameter_name_list = []

    if algorithm_name == "rf":
        parameter_name_list = ["n_estimators", "max_features"]

    elif algorithm_name == "extrarand":
        parameter_name_list = ["n_estimators", "max_features"]

    elif algorithm_name == "svm":
        if additional_info_dict['kernel_name'] == "rbf":
            parameter_name_list = ["C", "gamma"]
        elif additional_info_dict['kernel_name'] == "sigmoid":
            parameter_name_list = ["C", "gamma", "coef0"]
        elif additional_info_dict['kernel_name'] == "poly":
            parameter_name_list = ["C", "gamma", "coef0", "degree"]

    return parameter_name_list


class _Classifier:

    def __init__(self, algorithm_name, fuzzyoption, additional_info_dict=None, class_weight=None):

        self.algorithm_name = algorithm_name
        self.additional_info_dict = set_additional_info(algorithm_name, additional_info_dict)
        self.clf = set_classifier(algorithm_name, additional_info_dict, class_weight)
        self.fuzzy_option = fuzzyoption
        self.parameters_for_tuning_list = set_parameter_names(algorithm_name, additional_info_dict)

        self.gridsearch_df = None
        self.prediction_df = None
        self.roc_curve_df = None
        self.additional_metrics_sr = None

    def exact_parameters(self, param_dict):

        if self.algorithm_name == "rf":
            self.clf.set_params(n_estimators=int(param_dict['n_estimators']),
                                max_features=int(param_dict['max_features']))

        elif self.algorithm_name == "extrarand":
            self.clf.set_params(n_estimators=int(param_dict['n_estimators']),
                                max_features=int(param_dict['max_features']))

        elif self.algorithm_name == "svm":

            if self.additional_info_dict['kernel_name'] == "rbf":
                self.clf.set_params(C=param_dict['C'],
                                    gamma=param_dict['gamma'])

            elif self.additional_info_dict['kernel_name'] == "sigmoid":
                self.clf.set_params(C=param_dict['C'],
                                    gamma=param_dict['gamma'],
                                    coef0=param_dict['coef0'])

            elif self.additional_info_dict['kernel_name'] == 'poly':
                self.clf.set_params(C=param_dict['C'],
                                    gamma=param_dict['gamma'],
                                    coef0=param_dict['coef0'],
                                    degree=param_dict['degree'])

    def set_GSdf(self, gridsearch_df):
        self.gridsearch_df = gridsearch_df

    def set_additional_metrics_sr(self, additional_metrics_sr):
        self.additional_metrics_sr = additional_metrics_sr

    def set_roc_curve_df(self, roc_curve_df):
        self.roc_curve_df = roc_curve_df

    def set_prediction_df(self, prediction_df):
        self.prediction_df = prediction_df

    def modelfit(self, X, Y, sample_weights=None):
        """used for roc_curve() in evaluation_metrics"""
        self.clf.fit(X, Y, sample_weights)

    def predict(self, x_rescaled_data):
        pred_y = self.clf.predict(x_rescaled_data)
        return pred_y

    def predict_proba(self, x_rescaled_data, proper_order_of_labels):
        probabilities = self.clf.predict_proba(x_rescaled_data)
        probabilities_dict = {}
        for i, label_name in enumerate(proper_order_of_labels):
            probabilities_dict["prob_" + label_name] = probabilities[:, i]

        return probabilities_dict

    def decision_function(self, x_rescaled_data):
        if self.algorithm_name == "svm":
            distance = self.clf.decision_function(x_rescaled_data)
        else:
            print('decision_function() can be used only for svm!')
            distance = None
        return distance

    def fuzzy_membership(self, lstcr_dict):
        if self.fuzzy_option == 'normal':
            s_weights = None
        elif self.fuzzy_option == 'errorfuzzy':
            s_weights = lstcr_dict['err_weights']
        elif self.fuzzy_option == 'distancefuzzy':
            s_weights = lstcr_dict['dist_weights']
        else:
            s_weights = None
            print("Error: unknown fuzzy membership. (svm_class.py)")

        return s_weights

    def get_parameter_grid(self, algorithm_descriptor):

        whole_parameter_grid = algorithm_descriptor.parameters_for_grid_search
        output_parameter_grid = {parameter: whole_parameter_grid[parameter]
                                 for parameter in self.parameters_for_tuning_list}

        return output_parameter_grid

    def get_param_grid_for_deep_gridsearch(self, result_df):

        output_parameter_grid = {}
        parameters_that_should_be_left_the_same = ["poly", "max_features"]

        for parameter in self.parameters_for_tuning_list:
            best_value = result_df.iloc[-1][parameter]
            if parameter not in parameters_that_should_be_left_the_same:

                output_parameter_grid[parameter] = [best_value - best_value / 2,
                                                    best_value,
                                                    best_value + best_value / 2]
            else:
                output_parameter_grid[parameter] = best_value

        return output_parameter_grid

    def get_params_from_df(self, row):

        param_dict = {}
        for parameter in self.parameters_for_tuning_list:
            param_dict[parameter] = row[parameter]
        # print('param dict:', param_dict)
        return param_dict

    def get_name_str(self):

        if self.algorithm_name == "rf":
            name_str = self.algorithm_name + "_" + \
                       self.fuzzy_option

        elif self.algorithm_name == "extrarand":
            name_str = self.algorithm_name + "_" + \
                       self.fuzzy_option

        elif self.algorithm_name == "svm":
            name_str = self.algorithm_name + "_" + \
                       self.additional_info_dict['kernel_name'] + "_" + \
                       self.fuzzy_option + "_" + \
                       self.additional_info_dict['n_class_strategy']

        return name_str

    def fit_model_to_data(self, lstcr_dict):

        if self.fuzzy_option == 'normal':
            self.modelfit(lstcr_dict['X_Train'], lstcr_dict['y_Train'])
        elif self.fuzzy_option == 'errorfuzzy':
            self.modelfit(lstcr_dict['X_Train'], lstcr_dict['y_Train'], lstcr_dict['err_weights'])
        elif self.fuzzy_option == 'distancefuzzy':
            self.modelfit(lstcr_dict['X_Train'], lstcr_dict['y_Train'], lstcr_dict['dist_weights'])


    def make_predictions(self, lstcr_dict, algorithm_descriptor, mode):

        prediction_dict = {}
        if mode == "validation":
            prediction_dict['y_true'] = lstcr_dict['y_Test']
            data_name = "X_Test"
        elif mode == "generalization":
            data_name = "X_Generalization"

        prediction_dict['y_predicted'] = self.predict(lstcr_dict[data_name])

        if self.algorithm_name == "svm":

            prediction_dict['distance'] = self.decision_function(lstcr_dict[data_name])
            if algorithm_descriptor.algorithm_options_specification['prob_estimation'] is True:
                probability_dict = self.predict_proba(lstcr_dict[data_name],
                                                      algorithm_descriptor.proper_order_of_labels)
                prediction_dict.update(probability_dict)

        elif self.algorithm_name == "rf":
            probability_dict = self.predict_proba(lstcr_dict[data_name],
                                                  algorithm_descriptor.proper_order_of_labels)
            prediction_dict.update(probability_dict)

        elif self.algorithm_name == "extrarand":
            probability_dict = self.predict_proba(lstcr_dict[data_name],
                                                  algorithm_descriptor.proper_order_of_labels)
            prediction_dict.update(probability_dict)

        prediction_df = pd.DataFrame(data=prediction_dict)
        prediction_df = pd.concat([lstcr_dict["other_info_df"], prediction_df], axis=1)
        self.set_prediction_df(prediction_df)
