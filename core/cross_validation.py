# Different methods of fold generation. Documentation is available on the scikit page.
# Currently stratifiedkfold_indices() function is used.

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def kfold_generator(trainX, trainY, algorithm_descriptor):

    kfold = KFold(n_splits=algorithm_descriptor.num_of_cv, shuffle=True)
    kf_split = kfold.split(X=trainX, y=trainY)
    
    return kf_split


def stratifiedkfold_indices(Xdata, Ydata, algorithm_descriptor, rand_num, only_test_indices=False):

    indices = []
    strat_kfold = StratifiedKFold(n_splits=algorithm_descriptor.num_of_cv, shuffle=True, random_state=rand_num)

    if only_test_indices is False:
        for train_index, test_index in strat_kfold.split(X=Xdata, y=Ydata):
            indices.append((train_index, test_index))

    elif only_test_indices is True:
        for train_index, test_index in strat_kfold.split(X=Xdata, y=Ydata):
            indices.append(test_index)

    else:
        print('Error. Cross_validation.py')

    return indices
