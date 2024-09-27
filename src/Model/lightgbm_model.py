import os
import yaml
import pandas as pd
import optuna
import sys
import json
import numpy as np
import time

from typing import Dict, Any
from scipy.stats import pearsonr  
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from utils.math_component import normalized_root_mean_squared_error, percent_bias

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LightGBMModelOpt:
    def __init__(self, config_path: str, opt_config_path: str, log_dir: str = './config-log', log_file: str = 'lightgbm_trial.log'):
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
            model_params = opt_config['model']['params']
            best_params = model_params.copy()

            if 'training' in opt_config and 'learning_rate' in opt_config['training']:
                best_params['learning_rate'] = opt_config['training']['learning_rate']
        except KeyError as e:
            print(f"KeyError while extracting best parameters: {e}")
            default_params = {
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'max_depth': -1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0
            }
            best_params.update(default_params)
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
        data_dir = self.config['data']['output_dir']
        train_file = os.path.join(data_dir, 'train_data.csv')
        val_file = os.path.join(data_dir, 'val_data.csv')
        self.train_data = pd.read_csv(train_file, parse_dates=['Datetime'], index_col='Datetime')
        self.val_data = pd.read_csv(val_file, parse_dates=['Datetime'], index_col='Datetime')
        self.features = self.config['data']['features']
        self.target = self.config['data']['target']
        self.X_train = self.train_data[self.features]
        self.y_train = self.train_data[self.target]
        self.X_val = self.val_data[self.features]
        self.y_val = self.val_data[self.target]
        print("Data loaded.")

    def create_model(self, params: Dict[str, Any]) -> LGBMRegressor:
        params = params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42
        })
        model = LGBMRegressor(**params)
        return model

    def objective(self, trial: optuna.trial.Trial) -> float:
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 120, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.02, log=False),
            'n_estimators': trial.suggest_int('n_estimators', 1200, 1600),
            'max_depth': trial.suggest_int('max_depth', 7, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 35, 45),
            'subsample': trial.suggest_float('subsample', 0.85, 0.90),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.60, 0.65),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.2, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 0.3, log=True)
        }
        
        if self.is_duplicate(params):
            return float('inf')

        model = self.create_model(params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='rmse',
            callbacks=[early_stopping(stopping_rounds=200)]
        )
        preds = model.predict(self.X_val)
        rmse = root_mean_squared_error(self.y_val, preds)

        trial_record = {
            'parameters': params,
            'value': rmse
        }
        self.save_trial_log(trial_record)

        return rmse

    def optimize_hyperparameters(self, n_trials: int = 200, timeout: int = 3600):
        study = optuna.create_study(direction='minimize')

        if self.best_params:
            if not self.is_duplicate(self.best_params):
                study.enqueue_trial(self.best_params)
                print(f"Enqueued initial trial with best parameters: {self.best_params}")
            else:
                print("Best parameters from optimized config already tried.")

        start_time = time.time()
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        end_time = time.time()

        self.best_params = study.best_params
        print(f"Best parameters: {self.best_params}")

        opt_config = {
            'model': {
                'type': 'LGBMRegressor',
                'params': self.best_params
            },
            'data': self.config['data'],
            'training': self.config['training']
        }

        if 'learning_rate' in self.best_params:
            opt_config['training']['learning_rate'] = self.best_params['learning_rate']

        with open(self.opt_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(opt_config, f)
        print(f"Optimized config saved to '{self.opt_config_path}'.")

    def train_best_model(self):
        params = self.best_params.copy()
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': 42
        })

        model = LGBMRegressor(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric='rmse',
            callbacks=[early_stopping(stopping_rounds=200)]
        )
        self.model = model
        print("Best model training completed.")

    def evaluate(self):
        preds = self.model.predict(self.X_val)
        mae = mean_absolute_error(self.y_val, preds)
        rmse = root_mean_squared_error(self.y_val, preds)
        nrmse = normalized_root_mean_squared_error(self.y_val, preds)
        pbias = percent_bias(self.y_val, preds)
        r2 = r2_score(self.y_val, preds)

        corr, _ = pearsonr(self.y_val, preds)

        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation NRMSE: {nrmse:.4f}")
        print(f"Validation PBIAS: {pbias:.4f}%")
        print(f"Validation Pearson's CORR: {corr:.4f}")
        print(f"Validation R^2: {r2:.4f}")

    def save_model(self):
        model_dir = './models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'lightgbm_model_opt.txt')
        self.model.booster_.save_model(model_path)
        print(f"Model saved to '{model_path}'")

    def save_validation_results(self):
        validation_results = self.val_data.copy()
        validation_results['B'] = self.model.predict(self.X_val)

        output_columns = ['A', 'B', self.target]

        existing_columns = [col for col in output_columns if col in validation_results.columns]
        validation_results = validation_results[existing_columns]

        output_dir = './validation_results'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'validation_comparison_LightGBM.csv')

        validation_results.to_csv(output_file, index=True, encoding='utf-8')
        print(f"Validation results saved to '{output_file}'")

    def run_optimization(self):
        self.load_data()
        self.optimize_hyperparameters()
        self.train_best_model()
        self.evaluate()
        self.save_model()
        self.save_validation_results()

if __name__ == '__main__':
    CONFIG_PATH = './config/lightgbm_config.yaml'
    OPT_CONFIG_PATH = './config-optimization/lightgbm_config_opt.yaml'
    lightgbm_model_opt = LightGBMModelOpt(CONFIG_PATH, OPT_CONFIG_PATH)
    lightgbm_model_opt.run_optimization()