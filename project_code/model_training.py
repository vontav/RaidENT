import numpy as np
import os
import data_processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# Define sequence length
sequence_length = 10

# Fetch real-time Bitcoin price
def fetch_real_time_price():
    real_time_price = data_processing.fetch_real_time_price()
    if real_time_price is None:
        print("Real-time price fetch failed.")
    else:
        print("Real-time Bitcoin price:", real_time_price)
    return real_time_price

# Load historical data and scaler
data_file_path = os.path.join(os.path.dirname(__file__), 'bitcoin_data.csv')
normalized_data, scaler = data_processing.load_historical_data(data_file_path)

# Create rolling window for evaluation
rolling_window = normalized_data[-sequence_length:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()

# Define early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint_path = 'C:/Users/sgrif/Desktop/RaidENT/models/best_model_checkpoint.h5'
model_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

# Continuous loop for live data updates and training
while True:
    real_time_price = fetch_real_time_price()
    if real_time_price is not None:
        print("Updating data and retraining model...")
        data_processing.append_real_time_data(data_file_path, real_time_price)
        normalized_data, scaler = data_processing.load_historical_data(data_file_path)
        rolling_window = normalized_data[-sequence_length:]
        
        # Create sequences and targets from normalized data
        sequences, targets = data_processing.create_sequences_and_targets(normalized_data, sequence_length)
        
        # Split data into training and validation sets
        train_sequences, val_sequences, train_targets, val_targets = train_test_split(
            sequences, targets, test_size=0.2, random_state=42)
        
        # Reshape sequences to match the input shape of LSTM
        train_sequences = np.reshape(train_sequences, (train_sequences.shape[0], sequence_length, 1))
        val_sequences = np.reshape(val_sequences, (val_sequences.shape[0], sequence_length, 1))
        
        # Train the model with early stopping and model checkpoint
        history = model.fit(train_sequences, train_targets, validation_data=(val_sequences, val_targets),
                            epochs=100, batch_size=32, callbacks=[early_stopping, model_checkpoint])
    print("Waiting for 5 minutes...")
    time.sleep(300)  # Wait for 5 minutes











