# project_code/data_collection.py
import os
import data_processing
import time
import csv

data_file_path = 'bitcoin_data.csv'  # CSV file in the same directory as the script

data_collection_interval = 300  # 5 minutes

# Create the CSV file if it doesn't exist
if not os.path.exists(data_file_path):
    with open(data_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Bitcoin Price'])

while True:
    real_time_price = data_processing.fetch_real_time_price()
    if real_time_price is not None:
        with open(data_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([real_time_price])
        print("Real-time Bitcoin price:", real_time_price)
    time.sleep(data_collection_interval)

