from data_preprocessing import DataPreprocessor
from model import PredictiveMaintenanceModel
from train import train_model
from evaluate import evaluate_model
from predict import predict_failure

# Load and preprocess data
preprocessor = DataPreprocessor(time_steps=50)
data = preprocessor.load_data('train_FD001.txt')
scaled_features, labels = preprocessor.preprocess(data)
X, y = preprocessor.create_sequences(scaled_features, labels)

# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build and train the model
input_shape = (X_train.shape[1], X_train.shape[2])
model_instance = PredictiveMaintenanceModel(input_shape)
history = train_model(model_instance.model, X_train, y_train, X_val, y_val)

# Evaluate the model
predictions = evaluate_model(model_instance.model, X_test, y_test)

# Save the model and scaler
model_instance.save_model('predictive_maintenance_model.h5')
preprocessor.save_scaler('scaler.pkl')

# Example prediction
new_data = preprocessor.load_data('test_FD001.txt')
new_scaled_features, _ = preprocessor.preprocess(new_data)
new_X, _ = preprocessor.create_sequences(new_scaled_features, np.zeros(len(new_scaled_features)))
predictions = predict_failure(model_instance.model, preprocessor.scaler, new_X)
print(predictions)
