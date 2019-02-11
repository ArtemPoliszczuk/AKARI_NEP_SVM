# Grid search procedure (and deep grid search)

import pandas as pd
import time
from sklearn.model_selection import GridSearchCV

import evaluation_metrics


def run_gridsearch(clf_object, lstcr_dict, algorithm_descriptor):

    start_time = time.time()

    param_grid = clf_object.get_parameter_grid(algorithm_descriptor)
    # Get results from the shallow grid search:
    grid_search = get_grid_search_output(clf_object, algorithm_descriptor, lstcr_dict, param_grid)

    shallow_gs_result_df = grid_search_results_df(clf_object, grid_search)

    if algorithm_descriptor.num_of_deep_gs_loops>0:
        # Deep grid search:
        result_df, best_parameters = deep_grid_search(clf_object, algorithm_descriptor,
                                                  lstcr_dict, shallow_gs_result_df)
        clf_object.exact_parameters(best_parameters)
        clf_object.set_GSdf(result_df)
    else:
        best_parameters = grid_search.best_params_
        clf_object.exact_parameters(best_parameters)
        clf_object.set_GSdf(shallow_gs_result_df)

    # Other metrics on the best parameters for the main metric:
    evaluation_metrics.eval_metrics(lstcr_dict=lstcr_dict,
                                    clf_object=clf_object,
                                    algorithm_descriptor=algorithm_descriptor)

    print(clf_object.get_name_str() + ": ---%.2f seconds---" % (time.time() - start_time))


def deep_grid_search(clf_object, algorithm_descriptor, lstcr_dict, result_df):

    num_of_deep_gs_loops = algorithm_descriptor.num_of_deep_gs_loops

    best_parameters = None

    for step in range(num_of_deep_gs_loops):
        print('deep_gs', str(step))

        param_grid = clf_object.get_param_grid_for_deep_gridsearch(result_df, step+2)
        grid_search = get_grid_search_output(clf_object, algorithm_descriptor, lstcr_dict, param_grid)

        result_df = grid_search_results_df(clf_object, grid_search,
                                           previous_result_df=result_df)

        if step == num_of_deep_gs_loops - 1:
            best_parameters = grid_search.best_params_

    return result_df, best_parameters


def grid_search_results_df(clf_object, grid_search, previous_result_df = pd.DataFrame()):

    step_sr = pd.Series(data=grid_search.best_params_)
    b_i = grid_search.best_index_
    step_sr.loc['main_ev_metric'] = grid_search.cv_results_['mean_test_score'][b_i]
    step_sr.loc['main_ev_metric_std'] = grid_search.cv_results_['std_test_score'][b_i]
    step_sr.loc['classifier'] = clf_object.get_name_str()
    result_df = previous_result_df.append(step_sr, ignore_index=True)[step_sr.index.tolist()]

    return result_df


def get_grid_search_output(clf_object, algorithm_descriptor, lstcr_dict, param_grid):

    num_of_cores = algorithm_descriptor.num_of_cores

    s_weights = clf_object.fuzzy_membership(lstcr_dict)

    if "main_evaluation_metric" in algorithm_descriptor.algorithm_options_specification.keys():

        main_ev_metric = algorithm_descriptor.algorithm_options_specification['main_evaluation_metric']
        metric_scorers = evaluation_metrics.get_scorers(main_ev_metric)

        grid_search = GridSearchCV(clf_object.clf, param_grid=param_grid,
                                   scoring=metric_scorers, cv=lstcr_dict['cv_indices'],
                                   fit_params={'sample_weight': s_weights}, n_jobs=num_of_cores)

        if isinstance(metric_scorers, dict) is False:
            grid_search.fit(lstcr_dict['X_Train'], lstcr_dict['y_Train'])
        else:  # multimetric GS
            pass

    else:

        grid_search = GridSearchCV(clf_object.clf, param_grid=param_grid,
                                   cv=lstcr_dict['cv_indices'],
                                   fit_params={'sample_weight': s_weights},
                                   n_jobs=num_of_cores)
        grid_search.fit(lstcr_dict['X_Train'], lstcr_dict['y_Train'])

    return grid_search

