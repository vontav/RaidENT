import numpy as np
import os
import data_processing
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

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

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Print the model summary
model.summary()

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
best_val_rmse = float('inf')
num_iterations = 100  # Change this to the desired number of training iterations
for iteration in range(num_iterations):
    print("Iteration", iteration + 1)
    
    history = model.fit(train_sequences, train_targets, validation_data=(val_sequences, val_targets),
                        epochs=100, batch_size=32, callbacks=[early_stopping])
    
    # Calculate RMSE on validation data
    val_predictions = model.predict(val_sequences)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
    print("Validation RMSE:", val_rmse)

    # Check if the new model is better and save if necessary
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        model.save('C:/Users/sgrif/Desktop/RaidENT/models/Rsme_model.h5')
        print("Best model saved with validation RMSE:", best_val_rmse)

    # Wait for 5 minutes before repeating the process
    time.sleep(300)
