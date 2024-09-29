import os
import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.preprocessing import MinMaxScaler

class CNNPreprocessor:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.data = None

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        data_config = self.config['data']
        model_config = self.config['model']
        self.input_file = data_config['input_file']
        self.output_dir = data_config['output_dir']
        self.features = data_config['features']
        self.target = data_config['target']
        self.sequence_length = model_config['sequence_length']
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        self.data = pd.read_csv(self.input_file, parse_dates=['Datetime'])
        self.data.sort_values('Datetime', inplace=True)
        self.data.set_index('Datetime', inplace=True)
        print("Data loaded successfully.")

    def handle_missing_values(self):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.data[numeric_columns].interpolate(method='time')
        self.data.fillna(method='bfill', inplace=True)
        print("Missing values handled.")

    def scale_data(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        feature_data = self.data[self.features]
        target_data = self.data[[self.target]]

        self.feature_scaler.fit(feature_data)
        self.target_scaler.fit(target_data)

        scaled_features = self.feature_scaler.transform(feature_data)
        scaled_target = self.target_scaler.transform(target_data)

        self.data_scaled = np.hstack((scaled_features, scaled_target))

        joblib.dump(self.feature_scaler, os.path.join(self.output_dir, 'feature_scaler.gz'))
        joblib.dump(self.target_scaler, os.path.join(self.output_dir, 'target_scaler.gz'))
        print("Data scaling completed and scalers saved.")

    def create_sequences(self):
        sequences = []
        targets = []
        total_samples = len(self.data_scaled)
        feature_dim = len(self.features)
        for i in range(total_samples - self.sequence_length):
            seq_x = self.data_scaled[i:i+self.sequence_length, :feature_dim]
            seq_y = self.data_scaled[i+self.sequence_length, feature_dim]
            sequences.append(seq_x)
            targets.append(seq_y)
        self.X = np.array(sequences)
        self.y = np.array(targets).reshape(-1, 1)
        print("Sequences created for CNN.")

    def train_val_split(self):
        val_ratio = self.config['training']['validation_split']
        total_samples = len(self.X)
        val_start = int(total_samples * (1 - val_ratio))
        self.X_train = self.X[:val_start]
        self.y_train = self.y[:val_start]
        self.X_val = self.X[val_start:]
        self.y_val = self.y[val_start:]
        print("Data split into training and validation sets.")

    def save_data(self):
        np.save(os.path.join(self.output_dir, 'X_train.npy'), self.X_train)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), self.y_train)
        np.save(os.path.join(self.output_dir, 'X_val.npy'), self.X_val)
        np.save(os.path.join(self.output_dir, 'y_val.npy'), self.y_val)
        print(f"Preprocessed data saved to {self.output_dir}")

    def preprocess(self):
        self.load_data()
        self.handle_missing_values()
        self.scale_data()
        self.create_sequences()
        self.train_val_split()
        self.save_data()

if __name__ == '__main__':
    CONFIG_PATH = './config/cnn_config.yaml'
    preprocessor = CNNPreprocessor(CONFIG_PATH)
    preprocessor.preprocess()
