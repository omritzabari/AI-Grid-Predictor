import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import hourly


def build_dataset():
    print("1. Loading 10-years of Energy Data...")
    try:
        # Load the CSV file containing historical electricity data
        df_energy = pd.read_csv('PJME_hourly.csv')

        # Convert the 'Datetime' column to pandas datetime objects
        df_energy['Datetime'] = pd.to_datetime(df_energy['Datetime'])

        # Rename the target column for better readability
        df_energy.rename(columns={'PJME_MW': 'consumption'}, inplace=True)

        # Set the Datetime column as the index of the dataframe
        df_energy.set_index('Datetime', inplace=True)

        # Sort the data chronologically to ensure time-series integrity
        df_energy.sort_index(inplace=True)
    except FileNotFoundError:
        # Handle the error if the CSV file is missing from the directory
        print("ERROR: Please download 'PJME_hourly.csv' and place it in the folder.")
        exit()

    print("2. Fetching Historical Weather Data (Year by Year chunking)...")
    # This bypasses the buggy 'Point' search and directly downloads the correct data.
    station_id = '72408'

    all_weather_data = []

    # We fetch data one year at a time to completely avoid the 3-year API limit
    for year in range(2008, 2018):
        print(f"   -> Downloading weather data for {year}...")
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59)

        # Fetch the data for the specific station and year
        weather_data = hourly(station_id, start, end)
        df_year = weather_data.fetch()

        # Verify that the server actually returned data before appending
        if df_year is not None and not df_year.empty:
            all_weather_data.append(df_year)
        else:
            print(f"      [!] Warning: No data returned for the year {year}")

    # Safety check: Prevent crashing if ALL years failed to download
    if not all_weather_data:
        print("\nERROR: Could not download any weather data. Please check your internet connection or firewall.")
        exit()

    print("   -> Combining all years into a single dataset...")
    # Concatenate all the yearly dataframes into one massive dataframe
    df_weather = pd.concat(all_weather_data)

    # Select only the relevant weather columns: temperature, humidity, wind speed
    df_weather = df_weather[['temp', 'rhum', 'wspd']]

    # Rename the columns to match our project's naming conventions
    df_weather.rename(columns={'temp': 'temperature_c', 'rhum': 'humidity_percent', 'wspd': 'wind_speed'}, inplace=True)

    print("3. Merging Energy and Weather Data...")
    # Merge the two datasets based on their datetime indices (inner join)
    df_merged = df_energy.join(df_weather, how='inner')

    # Drop any rows containing missing values after the merge
    df_merged.dropna(inplace=True)

    print("4. Applying Advanced Feature Engineering (Time & Lags)...")
    # Reset the index so Datetime becomes a regular column again
    df_merged.reset_index(inplace=True)

    # Extract basic time-based features from the Datetime object
    df_merged['hour'] = df_merged['Datetime'].dt.hour
    df_merged['day_of_week'] = df_merged['Datetime'].dt.dayofweek
    df_merged['month'] = df_merged['Datetime'].dt.month
    df_merged['day_of_year'] = df_merged['Datetime'].dt.dayofyear

    # Create a binary feature indicating if the day is a weekend (Saturday=5, Sunday=6)
    df_merged['is_weekend'] = df_merged['day_of_week'].isin([5, 6]).astype(int)

    # Define conditions for each season
    conditions = [
        df_merged['month'].isin([12, 1, 2]),
        df_merged['month'].isin([3, 4, 5]),
        df_merged['month'].isin([6, 7, 8]),
        df_merged['month'].isin([9, 10, 11])
    ]

    # Apply the conditions to create a numerical season feature (1 to 4)
    df_merged['season'] = np.select(conditions, [1, 2, 3, 4], default=0)

    # --- ADVANCED LAG FEATURES ---
    print("   -> Calculating historical lags (T-24h and T-168h)...")

    # Create a feature for the power consumption exactly 24 hours ago
    df_merged['lag_24h_consumption'] = df_merged['consumption'].shift(24)

    # Create a feature for the power consumption exactly 168 hours (1 week) ago
    df_merged['lag_168h_consumption'] = df_merged['consumption'].shift(168)

    # Drop the initial rows that lack historical lag data (the first 168 hours)
    df_merged.dropna(inplace=True)

    print("5. Saving the dataset to the SQLite Database...")
    conn = sqlite3.connect('energy_db.sqlite')

    # Export the final dataframe into an SQL table named 'advanced_energy_data'
    df_merged.to_sql('advanced_energy_data', conn, if_exists='replace', index=False)

    # FIXED: Added IF NOT EXISTS to prevent crashes if the index is already there
    conn.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON advanced_energy_data(Datetime);")

    # Close the database connection safely
    conn.close()

    print("-" * 60)
    print(f"SUCCESS! Database created with {len(df_merged)} hourly records (approx 10 years).")
    print(f"Number of features available for ML: {len(df_merged.columns) - 2}")
    print("-" * 60)


if __name__ == "__main__":
    build_dataset()