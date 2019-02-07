import evaluation_metrics


def run_validation(clf_object, lstcr_dict, algorithm_descriptor):

    clf_object.fit_model_to_data(lstcr_dict)
    clf_object.make_predictions(lstcr_dict, algorithm_descriptor, mode="validation")

    evaluation_metrics.eval_metrics(lstcr_dict=lstcr_dict,
                                    clf_object=clf_object,
                                    algorithm_descriptor=algorithm_descriptor,
                                    validation_flag=True)


def run_generalization(clf_object, lstcr_dict, algorithm_descriptor):

    clf_object.fit_model_to_data(lstcr_dict)
    clf_object.make_predictions(lstcr_dict, algorithm_descriptor, mode="generalization")
