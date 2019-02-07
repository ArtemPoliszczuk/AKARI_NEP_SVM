# Creates fuzzy memberships. Distance weights are based on
# distance from the center of the class according to Lin & Wang 2002.
# Measurement errors are based on measurement uncertainties.

import numpy as np

def OneClassDistanceWeights(oneclass_df):
    
    mean_feature_values = oneclass_df.mean(axis=0)
    dif_from_mean = mean_feature_values - oneclass_df
    distances = np.linalg.norm(dif_from_mean.values, axis=1)

    r_max = np.amax(distances)
    delta = np.power(10, float(-6))

    dif = np.divide(distances,r_max+delta)
    ones = np.ones(len(oneclass_df))
    arr_weight = np.subtract(ones,dif)

    #print(distances)
    #print(arr_weight)
    #print(np.amax(arr_weight))
    #print(np.amin(arr_weight))

    return arr_weight

def OneClassErrorWeights(err_df):

    sum_of_err = err_df.abs().sum(axis=1)
    max_err = sum_of_err.max()
    delta = np.power(10, float(-6))
    dif = sum_of_err/(max_err+delta)
    arr_weight = 1 - dif

    #print(arr_weight)
    #print(arr_weight.max())
    #print(arr_weight.min())

    return arr_weight
