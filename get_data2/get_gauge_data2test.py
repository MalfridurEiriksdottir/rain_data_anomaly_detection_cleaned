# --- START OF MODIFIED get_gauge_data2.py ---

import os
import ssl
from urllib.request import urlopen
from datetime import datetime
import json
import pandas as pd
# import matplotlib.pyplot as plt # Keep commented out unless needed
# import matplotlib.dates as mdates
from pyproj import Transformer
import re
import numpy as np
import logging
from tqdm import tqdm
import pytz # Import pytz

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

# --- Sensor Metadata Reading (Keep as is) ---
try:
    df_sensors = pd.read_excel('SensorOversigt.xlsx')
    logger.info('Read the SensorOversigt.xlsx file')
except Exception as e: logger.error(f"Error reading SensorOversigt.xlsx: {e}"); exit()
ssl._create_default_https_context = ssl._create_unverified_context
def parse_wkt(wkt):
    if pd.isna(wkt): return np.nan, np.nan
    match = re.match(r"Point\s*\(([-\d\.]+)\s+([-\d\.]+)\)", str(wkt))
    if match:
        try: return float(match.group(1)), float(match.group(2))
        except ValueError: return np.nan, np.nan
    else: return np.nan, np.nan
if 'wkt_geom' not in df_sensors.columns: logger.error("'wkt_geom' not found."); exit()
df_sensors['longitude'], df_sensors['latitude'] = zip(*df_sensors['wkt_geom'].map(parse_wkt))
df_sensors.dropna(subset=['longitude', 'latitude'], inplace=True)
logger.info(f'Extracted coordinates for {len(df_sensors)} sensors.')
transformer_utm_to_wgs = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
transformer_wgs_to_utm = Transformer.from_crs("epsg:4326", "epsg:25832", always_xy=True)
# ---

# --- Gauge File Reading (Keep as is) ---
gauge_folder = "../gauge_data"; gauge_files = []
if not os.path.isdir(gauge_folder): logger.error(f"Gauge folder not found: {gauge_folder}"); exit()
try:
    gauge_files = [f for f in os.listdir(gauge_folder) if f.endswith('.csv') and f.startswith('gaugedata_')]
    logger.info(f'Found {len(gauge_files)} gauge data files.')
except Exception as e: logger.error(f"Error listing gauge files: {e}"); exit()
# ---

# --- Helper Functions ---
def get_channel(df_sensors, file_x, file_y): # Keep as is
    try: lon, lat = transformer_utm_to_wgs.transform(file_x, file_y)
    except Exception as e: logger.error(f"Coord transform failed for ({file_x}, {file_y}): {e}"); return None
    tolerance = 1e-5
    match = df_sensors[(np.isclose(df_sensors['longitude'], lon, atol=tolerance)) & (np.isclose(df_sensors['latitude'], lat, atol=tolerance))]
    if match.empty: logger.warning(f"No sensor match for UTM ({file_x}, {file_y})"); return None
    elif len(match) > 1: logger.warning(f"Multiple matches for UTM ({file_x}, {file_y}). Using first."); return match.iloc[0]['Channel']
    else: return match.iloc[0]['Channel']

# --- MODIFIED Time Series Processing ---
def process_time_series_to_utc(time_series_data):
    """
    Converts raw time series data points (naive string assumed local)
    to a DataFrame with a UTC datetime column.
    """
    if not time_series_data:
        return pd.DataFrame(columns=["datetime", "value"])

    df = pd.DataFrame(time_series_data, columns=["datetime_str", "value"])
    df['datetime'] = pd.to_datetime(df['datetime_str'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['datetime'])
    if df.empty:
        logger.warning("No valid datetime strings found in fetched data.")
        return pd.DataFrame(columns=["datetime", "value"])

    # *** Standard Timezone Handling: Local -> UTC ***
    try:
        df['datetime'] = df['datetime'].dt.tz_localize('Europe/Copenhagen', ambiguous='infer', nonexistent='shift_forward')
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        logger.debug("Timezone localized to Europe/Copenhagen and converted to UTC.")
    except Exception as tz_error:
        logger.error(f"Error during timezone conversion: {tz_error}. Returning naive datetimes.")
        # Fallback: Return naive times if conversion fails, but log the error
        df['datetime'] = pd.to_datetime(df['datetime_str'], format="%Y-%m-%d %H:%M:%S", errors='coerce')


    return df[['datetime', 'value']] # Return UTC datetimes (or naive on error)
# ---

# --- Main Processing Loop ---
DANISH_TZ = pytz.timezone('Europe/Copenhagen') # Define timezone object

for file in tqdm(gauge_files, desc="Processing gauge files (to UTC)"):
    logger.info(f"--- Processing file: {file} ---")
    try:
        # Extract UTM coords (Keep as is)
        filename_noext = file.replace('.csv', ''); parts = filename_noext.split('_')
        if len(parts) < 3: logger.warning(f"Skipping bad format: {file}"); continue
        try: x_filename, y_filename = float(parts[1]), float(parts[2])
        except ValueError: logger.warning(f"Could not parse coords: {file}"); continue

        # Find channel (Keep as is)
        channel_no = get_channel(df_sensors, x_filename, y_filename)
        if channel_no is None: logger.warning(f"No channel for {file}. Skipping."); continue
        logger.info(f"Found Channel {channel_no} for file {file}")

        # Read existing CSV data (Now expecting UTC)
        csv_filepath = os.path.join(gauge_folder, file)
        last_datetime_utc = None
        existing_df = pd.DataFrame(columns=["datetime", "value"])

        if os.path.exists(csv_filepath):
            try:
                # Read CSV, parse dates, ENSURE UTC
                existing_df = pd.read_csv(csv_filepath, parse_dates=['datetime'])
                if not pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                     existing_df['datetime'] = pd.to_datetime(existing_df['datetime'], errors='coerce')
                     existing_df = existing_df.dropna(subset=['datetime'])

                # ** Standardize existing data to UTC **
                if not existing_df.empty:
                    if existing_df['datetime'].dt.tz is None:
                        logger.warning(f"Existing data in {file} is naive. Assuming UTC.")
                        existing_df['datetime'] = existing_df['datetime'].dt.tz_localize('UTC')
                    elif existing_df['datetime'].dt.tz != pytz.UTC:
                        logger.warning(f"Existing data in {file} has timezone {existing_df['datetime'].dt.tz}. Converting to UTC.")
                        existing_df['datetime'] = existing_df['datetime'].dt.tz_convert('UTC')

                    if not existing_df.empty: # Check again after potential drops
                        last_datetime_utc = existing_df['datetime'].max() # Get last UTC time
                        logger.info(f"Existing UTC data found. Last timestamp (UTC): {last_datetime_utc}")

            except Exception as e:
                logger.error(f"Error reading/processing existing UTC file {csv_filepath}: {e}")
                continue

        # Determine fetch range using UTC times
        now_utc = pd.Timestamp.now(tz='UTC')
        if last_datetime_utc is not None:
             fetch_start_utc = last_datetime_utc + pd.Timedelta(microseconds=1)
        else:
             fetch_start_utc = now_utc - pd.Timedelta(days=30) # Default fetch period
             logger.info(f"No existing data, setting fetch start (UTC) to {fetch_start_utc}")
        fetch_end_utc = now_utc

        if fetch_start_utc >= fetch_end_utc:
            logger.info(f"Data up to date for channel {channel_no} (Last UTC: {last_datetime_utc}). Skipping fetch.")
            continue

        # Convert fetch range to Local time for API URL
        fetch_start_local = fetch_start_utc.tz_convert(DANISH_TZ)
        fetch_end_local = fetch_end_utc.tz_convert(DANISH_TZ)
        date_from_str = fetch_start_local.strftime('%Y-%m-%d%%20%H:%M:%S')
        date_to_str = fetch_end_local.strftime('%Y-%m-%d%%20%H:%M:%S')

        service_url = "https://nkaflob.watermanager.dk/"
        request_url = ( f"{service_url}/Services/DataService.ashx?type=graph&channel={channel_no}"
                       f"&DateFrom={date_from_str}&DateTo={date_to_str}&Reduction=no" ) # Use Reduction=no

        logger.info(f"Fetching data for channel {channel_no} (API times: {fetch_start_local} to {fetch_end_local} Local)")
        logger.debug(f"Request URL: {request_url}")

        # Fetch data (Keep as is)
        try:
            with urlopen(request_url) as f: response_text = f.read().decode('utf-8').strip()
        except Exception as e: logger.error(f"URL fetch error for {channel_no}: {e}"); continue
        if not response_text: logger.info(f"No response text for {channel_no}."); continue

        # Parse JSON (Keep as is)
        try: data = json.loads(response_text)
        except json.JSONDecodeError as e: logger.error(f"Invalid JSON for {channel_no}: {e}."); continue

        # Extract data points and process TO UTC
        channel_data = data.get("Channels", []); data_string = "no data"
        if channel_data: data_string = channel_data[0].get("StringifiedData", "no data")
        if data_string == "no data" or not data_string: logger.info(f"No 'StringifiedData' for {channel_no}."); continue

        try:
            raw_points = json.loads(data_string)
            if not isinstance(raw_points, list): logger.error(f"Unexpected StringifiedData format for {channel_no}."); continue
            # *** USE UTC PROCESSING FUNCTION ***
            new_data_df = process_time_series_to_utc(raw_points)
            # *** ------------------------- ***
        except Exception as e: logger.error(f"Error processing time series to UTC for {channel_no}: {e}"); continue

        # Combine with existing UTC data
        if not new_data_df.empty:
            logger.info(f"Fetched {len(new_data_df)} new data points (processed to UTC).")
            # Ensure both are UTC before combining
            if not existing_df.empty:
                 existing_df.set_index('datetime', inplace=True)
            new_data_df.set_index('datetime', inplace=True)

            # Combine using concat/drop_duplicates on the UTC index
            combined_df = pd.concat([existing_df, new_data_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            combined_df.reset_index(inplace=True) # Make datetime column again

            # Save combined data back to CSV *WITH UTC timezone info*
            try:
                combined_df.to_csv(csv_filepath, index=False, date_format='%Y-%m-%d %H:%M:%S%z') # Use %z
                logger.info(f"Updated UTC data saved to {csv_filepath}")
            except Exception as e:
                logger.error(f"Error saving combined UTC data to {csv_filepath}: {e}")
        else:
            logger.info(f"No new valid data points processed to UTC for {channel_no}.")

    except Exception as loop_error:
        logger.error(f"Unhandled error processing file {file}: {loop_error}", exc_info=True)

logger.info("--- Finished processing gauge files (to UTC) ---")
# --- END OF MODIFIED get_gauge_data2.py ---