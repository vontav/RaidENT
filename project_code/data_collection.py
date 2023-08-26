import requests
import time
import csv

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

if __name__ == "__main__":
    csv_filename = 'bitcoin_data.csv'

    while True:
        btc_price = fetch_btc_price()
        if btc_price is not None:
            print("Latest BTC Price:", btc_price)
            append_btc_price_to_csv(btc_price, csv_filename)
        time.sleep(60)  # Wait for 1 minute before the next request

