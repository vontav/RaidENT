import os
import requests
import time
import csv
import threading

def fetch_btc_price():
    url = "https://www.mexc.com/open/api/v2/market/ticker"
    params = {
        "symbol": "btc_usdt"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            btc_price = data["data"][0].get("last")
            return btc_price
        else:
            print("Invalid data format in API response.")
            return None
    else:
        print("Failed to fetch Bitcoin price from MEXC.")
        return None

def append_btc_price_to_csv(price, filename):
    with open(filename, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([price])

def create_csv_file(filename):
    with open(filename, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Do not write any header

def fetch_and_append_live_price(filename):
    while True:
        btc_price = fetch_btc_price()
        if btc_price is not None:
            append_btc_price_to_csv(btc_price, filename)
        time.sleep(60)  # Fetch every 1 minute

if __name__ == "__main__":
    csv_5min_filename = 'bitcoin_data_5min.csv'
    csv_15min_filename = 'bitcoin_data_15min.csv'
    csv_1hour_filename = 'bitcoin_data_1hour.csv'
    csv_live_filename = 'bitcoin_data_live.csv'
    max_live_data_points = 1000  # Adjust the number of live data points to keep

    # Create CSV files if they don't exist
    if not os.path.exists(csv_5min_filename):
        create_csv_file(csv_5min_filename)
    if not os.path.exists(csv_15min_filename):
        create_csv_file(csv_15min_filename)
    if not os.path.exists(csv_1hour_filename):
        create_csv_file(csv_1hour_filename)
    if not os.path.exists(csv_live_filename):
        create_csv_file(csv_live_filename)

    live_price_thread = threading.Thread(target=fetch_and_append_live_price, args=(csv_live_filename,))
    live_price_thread.start()

    while True:
        btc_price = fetch_btc_price()
        if btc_price is not None:
            print("Latest BTC Price:", btc_price)
            append_btc_price_to_csv(btc_price, csv_5min_filename)
            
            if time.localtime().tm_min % 5 == 0:  # Collect every 5 minutes
                append_btc_price_to_csv(btc_price, csv_5min_filename)
                
            if time.localtime().tm_min % 15 == 0:  # Collect every 15 minutes
                append_btc_price_to_csv(btc_price, csv_15min_filename)
                
            if time.localtime().tm_min == 0:  # Collect at the start of each hour
                append_btc_price_to_csv(btc_price, csv_1hour_filename) 
            
        time.sleep(60)  # Wait for 1 minute before the next request
