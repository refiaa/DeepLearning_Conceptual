import os
import pandas as pd
import joblib
import warnings
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List

warnings.filterwarnings('ignore')

""""
THIS SCRIPT DO RANDOM FOREST REGRESSOR BASED RESIDUAL MODELING (BIAS CORRECTION)

"""

# MATH FUNCTION HERE

def nash_sutcliffe_efficiency(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

def percent_bias(y_true, y_pred):
    return 100 * np.sum(y_pred - y_true) / np.sum(y_true)

# MATH FUCNTION END

class BiasCorrection:
    def __init__(self, file_paths: List[str], output_dir: str):
        self.file_paths = file_paths
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.models = {}
        self.metrics = {}

    def load_data(self):
        self.data_frames = {}
        for file_path in self.file_paths:
            model_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            self.data_frames[model_name] = df
            print(f"Loaded data from {file_path}")

    def train_correction_models(self):
        for model_name, df in self.data_frames.items():
            df['Residual'] = df['Variation_Placeholder'] - df['Variation_Placeholder_1']
            
            X = df[['Variation_Placeholder_1', 'Variation_Placeholder_2']]
            y = df['Residual']
            
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[model_name] = model
            print(f"Trained residual correction model for {model_name}")

    # K-Fold and GridSearchCV with Random Forest 
    # UnCommnet when wanna use it

    # def train_correction_models(self):
    #     for model_name, df in self.data_frames.items():
    #         df['Residual'] = df['Discharge'] - df['Estimated_Discharge']
            
    #         X = df[['Estimated_Discharge', 'Rainfall']]
    #         y = df['Residual']
            
    #         param_grid = {
    #             'n_estimators': [100, 200],
    #             'max_depth': [None, 10, 20],
    #             'min_samples_split': [2, 5],
    #             'min_samples_leaf': [1, 2],
    #         }
            
    #         rf = RandomForestRegressor(random_state=42)
            
    #         grid_search = GridSearchCV(
    #             estimator=rf,
    #             param_grid=param_grid,
    #             cv=5,
    #             scoring='neg_mean_squared_error',
    #             n_jobs=-1
    #         )
            
    #         grid_search.fit(X, y)
            
    #         best_model = grid_search.best_estimator_
    #         self.models[model_name] = best_model
    #         print(f"Trained residual correction model for {model_name} using GridSearchCV")

    def apply_correction(self):
        for model_name, df in self.data_frames.items():
            model = self.models[model_name]
            X = df[['Variation_Placeholder_1', 'Variation_Placeholder_2']]
            df['Predicted_Residual'] = model.predict(X)
            df['Corrected_Placeholder'] = df['Variation_Placeholder_1'] + df['Predicted_Residual']
            self.data_frames[model_name] = df
            print(f"Applied residual correction to {model_name}")

    def evaluate_models(self):
        for model_name, df in self.data_frames.items():
            y_true = df['Variation_Placeholder']
            y_pred_before = df['Variation_Placeholder_1']
            y_pred_after = df['Corrected_Placeholder']

            mae_before = mean_absolute_error(y_true, y_pred_before)
            rmse_before = mean_squared_error(y_true, y_pred_before, squared=False)
            r2_before = r2_score(y_true, y_pred_before)
            pbias_before = percent_bias(y_true, y_pred_before)
            nse_before = nash_sutcliffe_efficiency(y_true, y_pred_before)
            
            corr_before = np.corrcoef(y_true, y_pred_before)[0, 1]
            nrmse_before = rmse_before / (y_true.max() - y_true.min())

            mae_after = mean_absolute_error(y_true, y_pred_after)
            rmse_after = mean_squared_error(y_true, y_pred_after, squared=False)
            r2_after = r2_score(y_true, y_pred_after)
            pbias_after = percent_bias(y_true, y_pred_after)
            nse_after = nash_sutcliffe_efficiency(y_true, y_pred_after)
            
            corr_after = np.corrcoef(y_true, y_pred_after)[0, 1]
            nrmse_after = rmse_after / (y_true.max() - y_true.min())

            metrics = {
                'MAE_before': mae_before,
                'RMSE_before': rmse_before,
                'R2_before': r2_before,
                'PBIAS_before': pbias_before,
                'NSE_before': nse_before,
                'Corr_before': corr_before,
                'NRMSE_before': nrmse_before,
                'MAE_after': mae_after,
                'RMSE_after': rmse_after,
                'R2_after': r2_after,
                'PBIAS_after': pbias_after,
                'NSE_after': nse_after,
                'Corr_after': corr_after,
                'NRMSE_after': nrmse_after
            }
            self.metrics[model_name] = metrics

            print(f"Model: {model_name}")
            print(f"--- Before Correction ---")
            print(f"MAE   : {mae_before:.4f}")
            print(f"RMSE  : {rmse_before:.4f}")
            print(f"R2    : {r2_before:.4f}")
            print(f"PBIAS : {pbias_before:.2f}%")
            print(f"NSE   : {nse_before:.4f}")
            print(f"Corr  : {corr_before:.4f}")
            print(f"NRMSE : {nrmse_before:.4f}")
            print(f"--- After Correction ---")
            print(f"MAE   : {mae_after:.4f}")
            print(f"RMSE  : {rmse_after:.4f}")
            print(f"R2    : {r2_after:.4f}")
            print(f"PBIAS : {pbias_after:.2f}%")
            print(f"NSE   : {nse_after:.4f}")
            print(f"Corr  : {corr_after:.4f}")
            print(f"NRMSE : {nrmse_after:.4f}")
            print("Evaluation completed.\n")

    def save_results(self):
        for model_name, df in self.data_frames.items():
            output_file = os.path.join(self.output_dir, f"{model_name}_BIAS.csv")
            df.to_csv(output_file)
            print(f"Saved corrected results for {model_name} to {output_file}")

        metrics_df = pd.DataFrame(self.metrics).transpose()
        metrics_file = os.path.join(self.output_dir, "bias_correction_metrics.csv")
        metrics_df.to_csv(metrics_file)
        print(f"Saved bias correction metrics to {metrics_file}")

    def run(self):
        self.load_data()
        self.train_correction_models()
        self.apply_correction()
        self.evaluate_models()
        self.save_results()

if __name__ == "__main__":
    file_paths = [
        'CSV_FILE_PATH_HERE',
    ]
    output_dir = 'OUTPUT_PATH_HERE'

    bias_correction = BiasCorrection(file_paths, output_dir)
    bias_correction.run()
