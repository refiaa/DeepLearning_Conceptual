import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def normalized_root_mean_squared_error(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    return nrmse

def percent_bias(y_true, y_pred):
    pbias = 100.0 * np.sum(y_pred - y_true) / np.sum(y_true)
    return pbias
