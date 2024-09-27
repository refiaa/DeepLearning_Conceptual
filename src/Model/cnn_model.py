import os
import yaml
import numpy as np
import joblib
import optuna
import sys
import json
from typing import List, Dict, Any
from scipy.stats import pearsonr  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
FUCKING COMPILER ERROR WONT SHUT UP 

THAT IS WHY THERE IS FUCKIN `# type: ignore`IN EVERY tensorflow.keras
"""

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore

from utils.data_loader import load_time_series_data
from utils.metrics import mean_absolute_error, root_mean_squared_error, r_squared
from utils.math_component import normalized_root_mean_squared_error, percent_bias

class CNNModelOpt:
    def __init__(self, config_path: str, opt_config_path: str, log_dir: str = './config-log', log_file: str = 'cnn_trial.log'):
        self.config_path = config_path
        self.opt_config_path = opt_config_path
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(self.log_dir, self.log_file)
        self.config = None
        self.model = None
        self.best_params = None
        self.trial_log = []
        self.load_config()
        self.load_trial_log()

    def load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"Base config file '{self.config_path}' loaded.")

        if os.path.exists(self.opt_config_path):
            with open(self.opt_config_path, 'r', encoding='utf-8') as f:
                opt_config = yaml.safe_load(f)
            print(f"Optimized config file '{self.opt_config_path}' found and loaded.")

            self.update_config(self.config, opt_config)
            print("Configuration updated with optimized settings.")

            self.best_params = self.extract_best_params(opt_config)
            print(f"Extracted best parameters from optimized config: {self.best_params}")
        else:
            print(f"Optimized config file '{self.opt_config_path}' does not exist. Using default settings.")

    def extract_best_params(self, opt_config: Dict[str, Any]) -> Dict[str, Any]:
        best_params = {}
        try:
            conv_layers = opt_config['model']['conv_layers']
            best_params['conv_filters1'] = conv_layers[0]['filters']
            best_params['conv_filters2'] = conv_layers[1]['filters']
            best_params['kernel_size'] = conv_layers[0]['kernel_size']

            pooling = opt_config['model']['pooling']
            best_params['pooling_size'] = pooling['pool_size']

            dense_layers = opt_config['model']['dense_layers']
            best_params['dense_units'] = dense_layers[0]['units']

            best_params['learning_rate'] = opt_config['training']['learning_rate']

            if 'dropout_rate' in opt_config['model']:
                best_params['dropout_rate'] = opt_config['model']['dropout_rate']
            else:
                best_params['dropout_rate'] = 0.3

        except KeyError as e:
            print(f"KeyError while extracting best parameters: {e}")
        return best_params

    def update_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]):
        for key, value in override_config.items():
            if isinstance(value, dict) and key in base_config:
                self.update_config(base_config[key], value)
            else:
                base_config[key] = value

    def load_trial_log(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        trial = json.loads(line.strip())
                        self.trial_log.append(trial)
                    except json.JSONDecodeError:
                        print(f"Invalid log entry skipped: {line.strip()}")
            print(f"Loaded {len(self.trial_log)} past trials from '{self.log_path}'.")
        else:
            print(f"No existing trial log found at '{self.log_path}'. Starting fresh.")

    def save_trial_log(self, trial: Dict[str, Any]):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trial) + '\n')
        self.trial_log.append(trial)
        print(f"Trial parameters and result saved to '{self.log_path}'.")

    def is_duplicate(self, params: Dict[str, Any]) -> bool:
        for trial in self.trial_log:
            if self.compare_params(trial['parameters'], params):
                print(f"Duplicate trial found: {params}")
                return True
        return False

    def compare_params(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        for key in params1:
            if key not in params2:
                return False
            val1 = params1[key]
            val2 = params2[key]
            if isinstance(val1, float) and isinstance(val2, float):
                if not np.isclose(val1, val2, atol=1e-8):
                    return False
            else:
                if val1 != val2:
                    return False
        return True

    def load_data(self):
        self.X_train, self.y_train, self.X_val, self.y_val, self.feature_scaler, self.target_scaler = load_time_series_data(self.config)
        print("Data loaded.")

    def create_model(self, params: Dict[str, Any]) -> Sequential:
        conv_filters1 = params['conv_filters1']
        conv_filters2 = params['conv_filters2']
        conv_filters3 = params['conv_filters3']
        kernel_size = params['kernel_size']
        pooling_size = params['pooling_size']
        dropout_rate = params['dropout_rate']
        dense_units = params['dense_units']
        learning_rate = params['learning_rate']

        sequence_length = self.config['model']['sequence_length']
        feature_dim = self.X_train.shape[2]

        model = Sequential()
        
        model.add(Conv1D(filters=conv_filters1, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(sequence_length, feature_dim)))
        model.add(BatchNormalization())

        model.add(Conv1D(filters=conv_filters2, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())

        model.add(Conv1D(filters=conv_filters3, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())

        model.add(MaxPooling1D(pool_size=pooling_size))

        model.add(Dropout(dropout_rate))

        model.add(Flatten())

        model.add(Dense(units=dense_units, activation='relu'))

        model.add(Dropout(dropout_rate))

        model.add(Dense(units=1, activation='linear'))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model


    def objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            'conv_filters1': trial.suggest_categorical('conv_filters1', [32, 64, 96]),
            'conv_filters2': trial.suggest_categorical('conv_filters2', [32, 64, 96]),
            'conv_filters3': trial.suggest_categorical('conv_filters3', [16, 32, 64]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
            'pooling_size': trial.suggest_categorical('pooling_size', [2, 3]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.25, 0.4),
            'dense_units': trial.suggest_categorical('dense_units', [64, 128, 256]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-4, log=True)
        }

        if self.is_duplicate(params):
            return float('inf')

        model = self.create_model(params)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=150,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )

        val_loss = history.history['val_loss'][-1]

        trial_record = {
            'parameters': params,
            'value': val_loss
        }
        self.save_trial_log(trial_record)

        return val_loss

    def optimize_hyperparameters(self, n_trials: int = 50, timeout: int = 4800):
        study = optuna.create_study(direction='minimize')
    
        if self.best_params:
            if not self.is_duplicate(self.best_params):
                study.enqueue_trial(self.best_params)
                print(f"Enqueued initial trial with best parameters: {self.best_params}")
            else:
                print("Best parameters from optimized config already tried.")

        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        self.best_params = study.best_params
        print(f"Best hyperparameters: {self.best_params}")

        opt_config = {
            'model': {
                'sequence_length': self.config['model']['sequence_length'],
                'conv_layers': [
                    {'filters': self.best_params['conv_filters1'], 'kernel_size': self.best_params['kernel_size'], 'activation': 'relu'},
                    {'filters': self.best_params['conv_filters2'], 'kernel_size': self.best_params['kernel_size'], 'activation': 'relu'}
                ],
                'pooling': {
                    'type': self.config['model']['pooling']['type'],
                    'pool_size': self.best_params['pooling_size']
                },
                'dense_layers': [
                    {'units': self.best_params['dense_units'], 'activation': 'relu'}
                ],
                'output_units': self.config['model']['output_units'],
                'output_activation': self.config['model']['output_activation']
            },
            'training': {
                'loss': self.config['training']['loss'],
                'optimizer': self.config['training']['optimizer'],
                'learning_rate': self.best_params['learning_rate'],
                'epochs': self.config['training']['epochs'],
                'batch_size': self.config['training']['batch_size'],
                'validation_split': self.config['training']['validation_split'],
                'callbacks': self.config['training']['callbacks']
            },
            'data': self.config['data']
        }

        if 'dropout_rate' in self.best_params:
            opt_config['model']['dropout_rate'] = self.best_params['dropout_rate']

        with open(self.opt_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(opt_config, f)
        print(f"Optimized config saved to '{self.opt_config_path}'.")

    def build_best_model(self):
        conv_filters1 = self.best_params['conv_filters1']
        conv_filters2 = self.best_params['conv_filters2']
        conv_filters3 = self.best_params['conv_filters3']

        kernel_size = self.best_params['kernel_size']
        pooling_size = self.best_params['pooling_size']
        dropout_rate = self.best_params.get('dropout_rate', 0.3)
        dense_units = self.best_params['dense_units']
        learning_rate = self.best_params['learning_rate']

        sequence_length = self.config['model']['sequence_length']
        feature_dim = self.X_train.shape[2]

        self.model = Sequential()
        
        # Layer 1: First Convolutional Layer
        self.model.add(Conv1D(filters=conv_filters1, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(sequence_length, feature_dim)))
        
        # Layer 2: First Batch Normalization Layer
        self.model.add(BatchNormalization())

        # Layer 3: Second Convolutional Layer
        self.model.add(Conv1D(filters=conv_filters2, kernel_size=kernel_size, activation='relu', padding='same'))
        
        # Layer 4: Second Batch Normalization Layer
        self.model.add(BatchNormalization())

        # Layer 5: Third Convolutional Layer
        self.model.add(Conv1D(filters=conv_filters3, kernel_size=kernel_size, activation='relu', padding='same'))
        
        # Layer 6: Third Batch Normalization Layer
        self.model.add(BatchNormalization())

        # Layer 7: Max Pooling Layer
        self.model.add(MaxPooling1D(pool_size=pooling_size))

        # Layer 8: First Dropout Layer
        self.model.add(Dropout(dropout_rate))

        # Layer 9: Flatten Layer
        self.model.add(Flatten())

        # Layer 10: First Dense Layer
        self.model.add(Dense(units=dense_units, activation='relu', kernel_regularizer=l2(self.best_params.get('l2_reg', 1e-6))))

        # Layer 11: Second Dropout Layer
        self.model.add(Dropout(dropout_rate))

        # Layer 12: Output Dense Layer
        self.model.add(Dense(units=1, activation='linear'))

        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        print("CNN model built with optimized hyperparameters.")

    def train_best_model(self):
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=150,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
        print("Best model training completed.")

    def evaluate(self):
        y_pred = self.model.predict(self.X_val)
        y_true = self.target_scaler.inverse_transform(self.y_val)
        y_pred = self.target_scaler.inverse_transform(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        pbias = percent_bias(y_true, y_pred)
        corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        r2 = r_squared(y_true, y_pred)

        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation NRMSE: {nrmse:.4f}")
        print(f"Validation PBIAS: {pbias:.4f}%")
        print(f"Validation Pearson's CORR: {corr:.4f}")
        print(f"Validation R^2: {r2:.4f}")

        feature_index_C = self.config['data']['features'].index('C')
        sequences_C = self.X_val[:, :, feature_index_C]

        values_C = sequences_C[:, -1]

        values_C = self.feature_scaler.inverse_transform(self.X_val[:, -1, :])[:, feature_index_C]

        import pandas as pd
        result_df = pd.DataFrame({
            'A': y_true.flatten(),
            'B': y_pred.flatten(),
            'C': values_C.flatten()
        })

        os.makedirs('DIRECTORY_PATH', exist_ok=True)
        result_df.to_csv('DIRECTORY_PATH', index=False)
        print("save to'DIRECTORY_PATH'")

    def save_model(self):
        model_dir = './models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'cnn_model_opt.h5')
        self.model.save(model_path)
        print(f"Optimized model saved to '{model_path}'.")

    def run_optimization(self):
        self.load_data()
        self.optimize_hyperparameters()
        self.build_best_model()
        self.train_best_model()
        self.evaluate()
        self.save_model()

if __name__ == '__main__':
    CONFIG_PATH = './config/cnn_config.yaml'
    OPT_CONFIG_PATH = './config-optimization/cnn_config_opt.yaml'
    cnn_model_opt = CNNModelOpt(CONFIG_PATH, OPT_CONFIG_PATH)
    cnn_model_opt.run_optimization()
