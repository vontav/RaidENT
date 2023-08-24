import numpy as np
import os
import data_processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from tensorflow.keras.callbacks import EarlyStopping
import time

# Define sequence length
sequence_length = 10

# Fetch real-time Bitcoin price
real_time_price = data_processing.fetch_real_time_price()
if real_time_price is None:
    print("Real-time price fetch failed. Exiting.")
    exit()

print("Real-time Bitcoin price:", real_time_price)

# Process the normalized real-time price
data_file_path = os.path.join(os.path.dirname(__file__), 'bitcoin_data.csv')
normalized_data, scaler = data_processing.load_historical_data(data_file_path)

# Create sequences and targets from normalized data
sequences, targets = data_processing.create_sequences_and_targets(normalized_data, sequence_length)

# Split data into training and validation sets
train_sequences, val_sequences, train_targets, val_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

# Reshape sequences to match the input shape of LSTM
train_sequences = np.reshape(train_sequences, (train_sequences.shape[0], sequence_length, 1))
val_sequences = np.reshape(val_sequences, (val_sequences.shape[0], sequence_length, 1))

# Define the LSTM model
def build_lstm_model(n_units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Define hyperparameter search space
param_space = {
    'n_units': Integer(10, 100),
    'dropout_rate': Real(0.1, 0.5, prior='log-uniform')
}

# Create KerasRegressor
lstm_regressor = KerasRegressor(build_fn=build_lstm_model, epochs=100, batch_size=32, verbose=0)

# Perform Bayesian hyperparameter optimization
bayes_cv = BayesSearchCV(lstm_regressor, param_space, n_iter=10, cv=TimeSeriesSplit(n_splits=3),
                         scoring=make_scorer(mean_squared_error, greater_is_better=False))
bayes_cv.fit(train_sequences, train_targets)

# Print the best parameters and score
print("Best Parameters:", bayes_cv.best_params_)
print("Best Negative MSE:", bayes_cv.best_score_)

# Save the best model
best_model = bayes_cv.best_estimator_.model
best_model.save('best_bayesian_model.h5')




