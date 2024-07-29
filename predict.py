def predict_failure(model, scaler, input_data, time_steps=50):
    scaled_data = scaler.transform(input_data)
    sequences = create_sequences(scaled_data, np.zeros(len(scaled_data)), time_steps)
    predictions = model.predict(sequences)
    return predictions

def create_sequences(data, labels, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(labels[i + time_steps])
    return np.array(X), np.array(y)
