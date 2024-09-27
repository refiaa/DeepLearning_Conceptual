import os
import pandas as pd
import yaml

class LightGBMPreprocessor:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.data = None

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        data_config = self.config['data']
        self.input_file = data_config['input_file']
        self.output_dir = data_config['output_dir']
        self.features = data_config['features']
        self.target = data_config['target']
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        self.data = pd.read_csv(self.input_file, parse_dates=['Datetime'])
        self.data.sort_values('Datetime', inplace=True)
        self.data.set_index('Datetime', inplace=True)
        print("Data loaded successfully.")

    def handle_missing_values(self):
        self.data[self.features + [self.target]] = self.data[self.features + [self.target]].interpolate(method='time').fillna(method='bfill')
        print("Missing values handled.")

    def split_data(self):
        train_end = '2015-09-30 23:00:00'
        val_start = '2015-10-01 00:00:00'
        self.train_data = self.data.loc[:train_end]
        self.val_data = self.data.loc[val_start:]
        print("Data split into training and validation sets.")

    def save_data(self):
        self.train_data.to_csv(os.path.join(self.output_dir, 'train_data.csv'))
        self.val_data.to_csv(os.path.join(self.output_dir, 'val_data.csv'))
        print(f"Preprocessed data saved to {self.output_dir}")

    def preprocess(self):
        self.load_data()
        self.handle_missing_values()
        self.split_data()
        self.save_data()

if __name__ == '__main__':
    CONFIG_PATH = './config/lightgbm_config.yaml'
    preprocessor = LightGBMPreprocessor(CONFIG_PATH)
    preprocessor.preprocess()
