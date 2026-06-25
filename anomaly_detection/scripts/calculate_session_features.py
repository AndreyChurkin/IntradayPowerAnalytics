"""
Calculates features for intraday trading sessions in a given intraday continuous market dataset.
The features can then be used for session classification and anomaly detection.

The entire length of each trading session is used for calculating features.
Feature engineering based on truncated sessions or sliding windows is considered in other scripts.

Andrey Churkin
https://andreychurkin.ru/

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from tqdm import tqdm
import time

# Track start time
start_time = time.time()



# Load the intraday market CSV file:
# ID_market_data = pd.read_csv("C://Users//achurkin//Documents//IDC_EPEX_DK1_BestBidAsk_no_duplicates.csv")
ID_market_data = pd.read_csv("C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//models//IDC_EPEX_DK1_BestBidAsk_clean_v1.csv")


# Skip the first useless column:
if "Unnamed: 0" in ID_market_data.columns:
    ID_market_data = ID_market_data.drop(columns=["Unnamed: 0"])


print(f"\nIntraday market data is read:")
print(ID_market_data.head())
# print(ID_market_data.info())
# print(f"\nlen(ID_market_data) = {len(ID_market_data)}")


# # Activate to count duplicates and remove them:
# count_duplicates = ID_market_data.duplicated().sum()
# ID_market_data = ID_market_data.drop_duplicates()
# print(f"\nNumber of duplicating records found in the data set (and deleted): {count_duplicates}")


# Find the delivery start times:
unique_delivery_start = ID_market_data["delivery_start"].unique()
unique_delivery_start = np.sort(unique_delivery_start) # <-- It is super important to sort the delivery dates to avoid bugs in the future!
print(f"\nNumber of unique delivery start times: {len(unique_delivery_start)}")
# print(unique_delivery_start)

# Create grouped data for faster iterations & analysis:
ID_session_groups = ID_market_data.groupby("delivery_start")



# Load the day-ahead market CSV file:
DA_market_data = pd.read_csv("C://Users//achurkin//Documents//MEGA//Imperial College London//Pierre Pinson//ViPES2X//Roar Nicolaisen//DK1 Day-ahead Prices_202301010000-202401010000.csv")
print("\nDay-ahead market data is read:")
# print(DA_market_data.head())
# print(DA_market_data.info())

# Converting DA time data to the intraday delivery format:
DA_convert_1 = DA_market_data['MTU (CET/CEST)'].str.split(' - ').str[0] # get the first string
DA_convert_2 = pd.to_datetime(DA_convert_1, format='%d.%m.%Y %H:%M') # convert to datetime
DA_market_data['MTU (CET/CEST)'] = DA_convert_2 # replace in the original DA data
DA_market_data = DA_market_data.sort_values(by='MTU (CET/CEST)', ascending=True) # sort

# print("DA_market_data:")
# print(DA_market_data)


# Set the number of trading sessions to prepare features for (they will be the input for Isolation Forest later):
# classify_N_sessions = 500
classify_N_sessions = len(unique_delivery_start)


# Defining the features to characterise trading sessions
feature_columns = ["hour_number", "session_length_hours", "total_number_of_orders",
                   "DA_price",
                   "Ask_mean_price", "Ask_price_std",
                   "Bid_mean_price", "Bid_price_std",
                   ]



# Calculate features for each session:
rows = []
for i in tqdm(range(classify_N_sessions), desc="Calculating features for the sessions"):
    session_delivery_start = unique_delivery_start[i]
    session_hour_number = pd.to_datetime(session_delivery_start).hour

    session_ID_data = ID_session_groups.get_group(session_delivery_start)
    session_ID_all_ts = session_ID_data["ts"]
    session_ID_all_ts_datetime = pd.to_datetime(session_ID_all_ts)

    timestamp_earliest = session_ID_all_ts_datetime.min()
    timestamp_latest = session_ID_all_ts_datetime.max()
    time_diff = timestamp_latest - timestamp_earliest
    session_length_hours = time_diff.total_seconds() / 3600  # Convert seconds to hours
    total_number_of_orders = len(session_ID_data)

    session_DA_position = DA_market_data[
        DA_market_data['MTU (CET/CEST)'] == pd.to_datetime(session_delivery_start)
    ].index[0]
    session_DA_price = DA_market_data.loc[session_DA_position, 'Day-ahead Price [EUR/MWh]']

    session_ask_prices = session_ID_data["ask_price"]
    session_bid_prices = session_ID_data["bid_price"]

    Ask_mean_price = session_ask_prices.mean()
    Ask_price_std = session_ask_prices.std()

    Bid_mean_price = session_bid_prices.mean()
    Bid_price_std = session_bid_prices.std()

    rows.append([session_hour_number, session_length_hours, total_number_of_orders,
                 session_DA_price,
                 Ask_mean_price, Ask_price_std,
                 Bid_mean_price, Bid_price_std
                 ])

df_features = pd.DataFrame(rows, columns=feature_columns)

print(f"\nData features have been engineered for {classify_N_sessions} trading sessions")

print(df_features)


df_features.to_csv("../results/session_features_v1.csv", index=False)
print(f"\nSaved successfully as 'trading_sessions_features.csv'")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️  Total time taken: {elapsed_time:.2f} seconds")