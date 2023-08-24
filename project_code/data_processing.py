import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import csv
import pandas as pd

# Function to normalize data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return normalized_data, scaler

# Function to denormalize data
def denormalize_data(normalized_data, scaler):
    denormalized_data = scaler.inverse_transform(normalized_data)
    return denormalized_data

# Function to create sequences and targets
def create_sequences_and_targets(normalized_data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(normalized_data) - sequence_length):
        sequence = normalized_data[i:i+sequence_length]
        target = normalized_data[i+sequence_length]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Function to load historical data
def load_historical_data(filename):
    df = pd.read_csv(filename, header=None, names=["Bitcoin Price"])
    data = df["Bitcoin Price"].values

    # Normalize the data and get the scaler
    normalized_data, scaler = normalize_data(data)
    return normalized_data, scaler

# Function to append real-time data
def append_real_time_data(filename, new_data):
    with open(filename, 'a') as file:
        file.write(str(new_data) + '\n')

# Function to fetch real-time price
def fetch_real_time_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        bitcoin_price = data.get("bitcoin", {}).get("usd")
        return bitcoin_price
    else:
        print("Failed to fetch real-time Bitcoin price.")
        return None


