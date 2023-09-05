import sys
import numpy as np
from tensorflow.keras.models import load_model
import data_processing

class BitcoinPredictionApp:
    def __init__(self):
        self.model = load_model('C:/Users/sgrif/Desktop/RaidENT/models/Prime_model.h5')
        self.normalized_historical_data, self.scaler = data_processing.load_historical_data('bitcoin_data.csv')
        self.sequence_length = 30

    def start_prediction(self):
        real_time_price = data_processing.fetch_real_time_price()
        if real_time_price is None:
            print("Real-time price fetch failed. Exiting.")
            return

        sequences, _ = data_processing.create_sequences_and_targets(self.normalized_historical_data, self.sequence_length)
        input_sequence = sequences[-1:]

        lstm_input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], self.sequence_length, 1))
        conv_input_sequence = np.reshape(input_sequence, (input_sequence.shape[0], self.sequence_length, 1))

        predictions = self.model.predict([lstm_input_sequence, conv_input_sequence])
        denormalized_predictions = data_processing.denormalize_data(predictions, self.scaler)

        predicted_price = int(denormalized_predictions[0][0])
        print(f"Predicted Bitcoin price: {predicted_price}")

if __name__ == "__main__":
    app = BitcoinPredictionApp()
    app.start_prediction()
















