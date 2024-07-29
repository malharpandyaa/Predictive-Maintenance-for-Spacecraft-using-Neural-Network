from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

class PredictiveMaintenanceModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(50, return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
