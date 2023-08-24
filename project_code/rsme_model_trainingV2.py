import numpy as np
import os
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
import data_processing  # Import your custom data_processing module

# Define sequence length and other parameters
sequence_length = 10
batch_size = 32
max_epochs = 100
patience = 10  # Increase patience for early stopping

# Load or create the model
model_path = 'C:/Users/sgrif/Desktop/RaidENT/models/Rsme_modelV2.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, save_weights_only=False)

# Training loop
iteration = 1
while True:
    print("Starting Iteration", iteration)
    
    # Load historical data using your custom data_processing module
    normalized_data, scaler = data_processing.load_historical_data('bitcoin_data.csv')
    print("Loaded historical data.")
    
    # Create sequences and targets using your custom data_processing module
    sequences, targets = data_processing.create_sequences_and_targets(normalized_data, sequence_length)
    
    # Split data into training and validation sets
    train_sequences, val_sequences, train_targets, val_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)
    
    # Reshape sequences to match the input shape of LSTM
    train_sequences = np.reshape(train_sequences, (train_sequences.shape[0], sequence_length, 1))
    val_sequences = np.reshape(val_sequences, (val_sequences.shape[0], sequence_length, 1))
    
    print("Training the model...")
    history = model.fit(train_sequences, train_targets, validation_data=(val_sequences, val_targets),
                        epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint])
    print("Model training completed.")
    
    # Wait for a few minutes before making predictions
    print("Waiting for 5 minutes before the next iteration...")
    time.sleep(300)
    
    # Increment iteration counter
    iteration += 1
