import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, time_steps=50):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_steps = time_steps

    def load_data(self, filepath):
        column_names = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2",
                        "operational_setting_3"] + [f"sensor_measurement_{i}" for i in range(1, 22)]
        data = pd.read_csv(filepath, sep=" ", header=None, names=column_names, index_col=False)
        data = data.dropna(axis=1, how='all')
        return data

    def preprocess(self, data):
        data = data.groupby('unit_number').apply(self._add_remaining_useful_life)
        features = data.drop(['unit_number', 'time_in_cycles', 'RUL'], axis=1)
        labels = data['RUL']
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, labels.values

    def _add_remaining_useful_life(self, df):
        max_cycle = df['time_in_cycles'].max()
        df['RUL'] = max_cycle - df['time_in_cycles']
        return df

    def create_sequences(self, data, labels):
        X, y = [], []
        for i in range(len(data) - self.time_steps):
            X.append(data[i:(i + self.time_steps)])
            y.append(labels[i + self.time_steps])
        return np.array(X), np.array(y)

    def save_scaler(self, filepath):
        import joblib
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath):
        import joblib
        self.scaler = joblib.load(filepath)
