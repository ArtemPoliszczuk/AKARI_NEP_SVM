import warnings
import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from sklearn.model_selection import cross_validate
from scipy import interp

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def eval_metrics(lstcr_dict, clf_object, algorithm_descriptor, validation_flag=False):

    #print(skmetrics.SCORERS.keys())
    metrics = algorithm_descriptor.additional_metrics

    if (validation_flag is True) and (algorithm_descriptor.algorithm_name == "svm"):
        # Main metric maximized in the grid search for SVM is not present in the
        # additional metrics list. For validation we add it to this list.
        gs_metric = algorithm_descriptor.algorithm_options_specification['main_evaluation_metric']
        metrics.append(gs_metric)

    # Getting metric results from cross validation.
    cv_result = get_cv_result(clf_object, lstcr_dict,
                              algorithm_descriptor, metrics,
                              validation_flag)

    # Creating pandas series for metric values storage.
    metrics_sr = pd.Series()
    for metric in metrics:
        metrics_sr.loc[metric] = np.mean(cv_result['test_' + metric])
        metrics_sr.loc[metric + "_std"] = np.std(cv_result['test_' + metric])

    metrics_sr.loc['classifier'] = clf_object.get_name_str()

    clf_object.set_additional_metrics_sr(metrics_sr)

    # ROC curve creation
    if algorithm_descriptor.roc_curve_flag is True:
        roc_curve_df = roc_curve(clf_object, lstcr_dict, algorithm_descriptor, validation_flag)
        clf_object.set_roc_curve_df(roc_curve_df)


def get_cv_result(clf_object, lstcr_dict, algorithm_descriptor, metrics, validation_flag):

    if validation_flag is False:

        num_of_cores = algorithm_descriptor.num_of_cores
        s_weights = clf_object.fuzzy_membership(lstcr_dict)
        X_data = lstcr_dict['X_Train']
        Y_data = lstcr_dict['y_Train']
        cv_indices = lstcr_dict['cv_indices']

        metric_scorers = get_scorers(metrics)

        cv_result_dict = cross_validate(estimator=clf_object.clf,
                                        X=X_data, y=Y_data,
                                        scoring=metric_scorers,
                                        cv=cv_indices,
                                        fit_params={'sample_weight': s_weights},
                                        n_jobs=num_of_cores)

        return cv_result_dict

    else:

        if "roc_auc" in metrics:

            positive_prob_name = "prob_" + algorithm_descriptor.proper_order_of_labels[1]
            y_info_df = clf_object.prediction_df[['y_true', 'y_predicted', positive_prob_name]]
            y_info_df = y_info_df.rename(columns={positive_prob_name: "positive_prob"}, )

        else:
            y_info_df = clf_object.prediction_df[['y_true', 'y_predicted']]

        results_on_subsets_dict = {"test_" + metric_name: [] for metric_name in metrics}

        for indices in lstcr_dict['test_cv_indices']:

            for metric_name in metrics:

                metric_result = get_metric_result(metric_name, y_info_df.loc[indices])
                results_on_subsets_dict["test_" + metric_name].append(metric_result)

        return results_on_subsets_dict


def get_metric_result(metric_name, y_info_df):

    if metric_name == "roc_auc":
        metric_result = skmetrics.roc_auc_score(y_info_df['y_true'], y_info_df['positive_prob'])

    elif metric_name == "matthews_corrcoef":
        metric_result = skmetrics.matthews_corrcoef(y_info_df['y_true'], y_info_df['y_predicted'])

    elif metric_name == "cohen_kappa":
        metric_result = skmetrics.cohen_kappa_score(y_info_df['y_true'], y_info_df['y_predicted'])

    elif metric_name == "recall":
        metric_result = skmetrics.recall_score(y_info_df['y_true'], y_info_df['y_predicted'])

    elif metric_name == "precision":
        metric_result = skmetrics.precision_score(y_info_df['y_true'], y_info_df['y_predicted'])

    elif metric_name == "f1":
        metric_result = skmetrics.f1_score(y_info_df['y_true'], y_info_df['y_predicted'])

    elif metric_name == "accuracy":
        metric_result = skmetrics.accuracy_score(y_info_df['y_true'], y_info_df['y_predicted'])

    else:
        print("Error: unknown metric (evaluation_metrics.py -> get_metri_result())")
        metric_result = None

    return metric_result

def get_confusion_matrix():
    None

def get_scorers(metrics):

    # If metrics is a single string (is a single metric)
    if isinstance(metrics, str) is True:
        metric_scorers = scorer(metrics)

    else:
        metric_scorers = {}
        for metric in metrics:
            metric_scorer = scorer(metric)
            metric_scorers[metric] = metric_scorer

    return metric_scorers


def scorer(metric):

    if metric in skmetrics.SCORERS.keys():
        metric_scorer = skmetrics.get_scorer(metric)
    elif metric == "cohen_kappa":
        metric_scorer = skmetrics.make_scorer(skmetrics.cohen_kappa_score)
    elif metric == "matthews_corrcoef":
        metric_scorer = skmetrics.make_scorer(skmetrics.matthews_corrcoef)
    else:
        metric_scorer = None
        print("Error: unknown metric.(evaluation_metrics.py -> eval_metrics()). ")

    return metric_scorer


def roc_curve(clf_object, lstcr_dict, algorithm_descriptor, validation_flag):

    s_weights = clf_object.fuzzy_membership(lstcr_dict)
    proper_order_of_labels = algorithm_descriptor.proper_order_of_labels

    fprs, tprs, treshs = [], [], []
    mean_fpr = np.linspace(0, 1, 100)

    if validation_flag is False:
        print("training ROC.")
        for cv_ind in lstcr_dict['cv_indices']:

            test_ind = cv_ind[1]
            train_ind = cv_ind[0]

            Ytst = lstcr_dict['y_Train'][test_ind]
            Xtst = lstcr_dict['X_Train'][test_ind]

            Ytr = lstcr_dict['y_Train'][train_ind]
            Xtr = lstcr_dict['X_Train'][train_ind]

            if s_weights is not None:
                tr_s_weights = s_weights[train_ind]
            else:
                tr_s_weights = None

            clf_object.modelfit(Xtr, Ytr, tr_s_weights)
            probabilities_dict = clf_object.predict_proba(Xtst, proper_order_of_labels)
            positive_prob = probabilities_dict["prob_" + proper_order_of_labels[1]]

            fpr, tpr, tresholds = skmetrics.roc_curve(Ytst, positive_prob)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

    elif validation_flag is True:

        print("validation ROC.")
        Ytr = lstcr_dict['y_Train']
        Xtr = lstcr_dict['X_Train']
        clf_object.modelfit(Xtr, Ytr, s_weights)

        for test_ind in lstcr_dict['test_cv_indices']:

            Ytst = lstcr_dict['y_Test'][test_ind]
            Xtst = lstcr_dict['X_Test'][test_ind]

            probabilities_dict = clf_object.predict_proba(Xtst, proper_order_of_labels)
            positive_prob = probabilities_dict["prob_" + proper_order_of_labels[1]]

            fpr, tpr, tresholds = skmetrics.roc_curve(Ytst, positive_prob)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    roc_df = pd.DataFrame({'fpr': mean_fpr, 'tpr': mean_tpr})
    return roc_df
