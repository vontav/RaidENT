import requests
import time

def get_btc_price():
    url = "https://contract.mexc.com/api/v1/contract/depth/BTC_USDT"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        data = response.json()
        bids = data.get("bids")
        
        if bids:
            btc_price = bids[0][0]  # The highest bid price represents the current price
            return btc_price
        else:
            print("No bid data available.")
            print("Response JSON:", data)
            return None
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None
    except ValueError as e:
        print("Failed to parse response JSON:", e)
        return None

def main():
    while True:
        btc_price = get_btc_price()
        if btc_price is not None:
            print("BTC Price:", btc_price)
        
        time.sleep(10)  # Fetch the price every 10 seconds

if __name__ == "__main__":
    main()





