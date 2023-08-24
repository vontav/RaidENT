import data_processing
from tensorflow.keras.models import load_model
import numpy as np

# Define sequence length
sequence_length = 10

# Fetch real-time Bitcoin price
real_time_price = data_processing.fetch_real_time_price()
if real_time_price is None:
    print("Real-time price fetch failed. Exiting.")
    exit()

print("Real-time Bitcoin price:", real_time_price)

# Load the trained rolling window model
model = load_model('C:/Users/sgrif/Desktop/RaidENT/models/Rsme_model.h5')

# Load historical data and scaler
normalized_historical_data, scaler = data_processing.load_historical_data('bitcoin_data.csv')

# Create sequences and targets from normalized historical data
sequences, targets = data_processing.create_sequences_and_targets(normalized_historical_data, sequence_length)

# Make predictions using the trained model
predictions = model.predict(sequences)

# Denormalize predictions
denormalized_predictions = data_processing.denormalize_data(predictions, scaler)

# Print denormalized predictions
print("Denormalized predictions:", denormalized_predictions)
print("Real-time Bitcoin price:", real_time_price)










