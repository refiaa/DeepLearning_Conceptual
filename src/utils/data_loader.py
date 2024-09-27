import os
import numpy as np
import pandas as pd
import joblib

def load_time_series_data(config):
    data_dir = config['data']['output_dir']
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    feature_scaler = joblib.load(os.path.join(data_dir, 'feature_scaler.gz'))
    target_scaler = joblib.load(os.path.join(data_dir, 'target_scaler.gz'))
    return X_train, y_train, X_val, y_val, feature_scaler, target_scaler

def load_sarima_data(config):
    data_file = config['data']['output_file']
    data = pd.read_csv(data_file, parse_dates=['Datetime'], index_col='Datetime')
    return data
