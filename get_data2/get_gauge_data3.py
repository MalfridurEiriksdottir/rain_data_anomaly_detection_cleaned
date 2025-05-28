
import os
import ssl
from urllib.request import urlopen
from datetime import datetime # Keep this for strptime
import json
import pandas as pd
from pyproj import Transformer
import re
import numpy as np
import logging
import traceback # For detailed error logging

# --- Logger Setup ---
#LOGGER = logging.getLogger(__name__)
# if not #LOGGER.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - GAUGE_INGEST - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     #LOGGER.addHandler(handler)
#     #LOGGER.setLevel(logging.INFO) # Set to DEBUG for more verbose output during development

#LOGGER.info("Start of the gauge data ingestion script")

# --- Configuration ---
SENSOR_METADATA_FILE = 'SensorOversigt.xlsx'
GAUGE_DATA_FOLDER = "../gauge_data" 
API_SERVICE_URL = "https://nkaflob.watermanager.dk/"
FALLBACK_START_DATE_STR = '2000-01-01 00:00:00' # For new sensors or if file reading fails
DATE_FORMAT_FOR_CSV_SAVE = "%Y-%m-%d %H:%M:%S" # For saving naive UTC strings

# --- Read Sensor Metadata ---
try:
    df_sensors = pd.read_excel(SENSOR_METADATA_FILE)
    #LOGGER.info(f"Successfully read sensor metadata from {SENSOR_METADATA_FILE}")
except FileNotFoundError:
    #LOGGER.critical(f"Fatal: Sensor metadata file '{SENSOR_METADATA_FILE}' not found. Exiting.")
    exit()
except Exception as e:
    #LOGGER.critical(f"Fatal: Error reading {SENSOR_METADATA_FILE}: {e}")
    exit()

ssl._create_default_https_context = ssl._create_unverified_context

def parse_wkt(wkt_string):
    match = re.match(r"Point \(([-\d\.]+) ([-\d\.]+)\)", str(wkt_string)) # Added str() for safety
    if match: return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan

try:
    if 'wkt_geom' not in df_sensors.columns:
        raise KeyError("'wkt_geom' column not found in sensor metadata.")
    df_sensors['longitude'], df_sensors['latitude'] = zip(*df_sensors['wkt_geom'].map(parse_wkt))
    #LOGGER.info("Successfully extracted coordinates from sensor metadata.")
except KeyError as e:
    #LOGGER.critical(f"Fatal: Missing essential column in sensor metadata: {e}")
    exit()
except Exception as e:
    #LOGGER.critical(f"Fatal: Error parsing WKT geometry from sensor metadata: {e}")
    exit()

transformer_filename_to_wgs84 = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
#LOGGER.info("Filename to WGS84 coordinate transformer created.")

if not os.path.exists(GAUGE_DATA_FOLDER):
    #LOGGER.info(f"Gauge data folder {GAUGE_DATA_FOLDER} does not exist. Creating it.")
    os.makedirs(GAUGE_DATA_FOLDER, exist_ok=True)

gauge_files_in_dir = os.listdir(GAUGE_DATA_FOLDER)
#LOGGER.info(f"Found {len(gauge_files_in_dir)} items in {GAUGE_DATA_FOLDER}")

def get_channel_from_coords(df_sensors_ref, lon_val, lat_val, filename_for_log=""):
    tolerance = 1e-5 # Slightly increased tolerance if needed
    required_cols = ['longitude', 'latitude', 'Channel']
    if not all(col in df_sensors_ref.columns for col in required_cols):
        #LOGGER.error(f"Sensor metadata missing one of required columns: {required_cols} for {filename_for_log}")
        return None
        
    match = df_sensors_ref[
        (np.isclose(df_sensors_ref['longitude'], lon_val, atol=tolerance)) &
        (np.isclose(df_sensors_ref['latitude'], lat_val, atol=tolerance))
    ]
    if match.empty:
        #LOGGER.warning(f"No channel match in metadata for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f})")
        return None
    elif len(match) > 1:
        print(f"Multiple channel matches for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f}). Using first: {match.iloc[0]['Channel']}")
        #LOGGER.warning(f"Multiple channel matches for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f}). Using first: {match.iloc[0]['Channel']}")
    return match.iloc[0]['Channel']

# --- Main Loop ---
for file_item_name in gauge_files_in_dir: # Iterate over items found in directory
    #LOGGER.info(f"\nProcessing file: {file_item_name}")
    if not file_item_name.lower().endswith('.csv') or not file_item_name.lower().startswith('gaugedata_'):
        #LOGGER.info(f"Skipping non-standard gauge file: {file_item_name}")
        continue

    filename_no_ext = file_item_name.replace('.csv', '').replace('.CSV','')
    try:
        parts = filename_no_ext.split('_')
        if len(parts) < 3: raise ValueError("Filename format error (not enough parts)")
        x_coord_from_filename, y_coord_from_filename = float(parts[1]), float(parts[2])
    except (ValueError, IndexError) as e_fn:
        #LOGGER.warning(f"Could not parse coordinates from filename '{file_item_name}': {e_fn}. Skipping.")
        continue

    try:
        lon_wgs84, lat_wgs84 = transformer_filename_to_wgs84.transform(x_coord_from_filename, y_coord_from_filename)
    except Exception as e_trans:
        #LOGGER.error(f"Error transforming coordinates for {file_item_name}: {e_trans}. Skipping.")
        continue

    channel_id = get_channel_from_coords(df_sensors, lon_wgs84, lat_wgs84, file_item_name)
    if channel_id is None:
        #LOGGER.warning(f"No channel identified for {file_item_name}. Skipping API fetch for this file.")
        continue
    #LOGGER.info(f"Matched channel {channel_id} for file {file_item_name}")

    # --- Determine Start Date for API Fetch ---
    current_file_path = os.path.join(GAUGE_DATA_FOLDER, file_item_name)
    api_fetch_start_date_utc = pd.Timestamp(FALLBACK_START_DATE_STR).tz_localize('UTC') # Default

    if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
        try:
            df_existing_gauge = pd.read_csv(current_file_path) # Read first
            if 'datetime' in df_existing_gauge.columns and not df_existing_gauge.empty:
                df_existing_gauge['datetime'] = pd.to_datetime(df_existing_gauge['datetime'], errors='coerce')
                df_existing_gauge.dropna(subset=['datetime'], inplace=True)
                
                if not df_existing_gauge.empty:
                    # Assume CSV stores naive strings representing TRUE UTC
                    if df_existing_gauge['datetime'].dt.tz is None:
                        #LOGGER.debug(f"Localizing naive datetimes from {file_item_name} to UTC.")
                        df_existing_gauge['datetime'] = df_existing_gauge['datetime'].dt.tz_localize('UTC', ambiguous='NaT').dropna()
                    elif str(df_existing_gauge['datetime'].dt.tz).upper() != 'UTC':
                        #LOGGER.debug(f"Converting datetimes from {file_item_name} (tz={df_existing_gauge['datetime'].dt.tz}) to UTC.")
                        df_existing_gauge['datetime'] = df_existing_gauge['datetime'].dt.tz_convert('UTC')
                    
                    if not df_existing_gauge.empty: # After potential NaT drop
                        last_ts = df_existing_gauge['datetime'].max()
                        if pd.notna(last_ts):
                            api_fetch_start_date_utc = last_ts + pd.Timedelta(microseconds=1)
            else: 
                print(f"File {current_file_path} is empty or missing 'datetime' column. Using fallback start.")
                #LOGGER.info(f"File {file_item_name} is empty or missing 'datetime' column. Using fallback start.")
        except Exception as e_read_file:
            print(f"Error reading or processing existing file {current_file_path}: {e_read_file}")
            #LOGGER.warning(f"Error reading or processing existing {file_item_name}: {e_read_file}. Using fallback start.")
    else:
        print(f"File {current_file_path} not found or empty. Assuming new sensor.")
        #LOGGER.info(f"File {file_item_name} not found or empty. Assuming new sensor.")

    api_fetch_end_date_utc = pd.Timestamp.now(tz='UTC')
    print(f"API Fetch Range for {channel_id}: From={api_fetch_start_date_utc}, To={api_fetch_end_date_utc}")    
    #LOGGER.info(f"API Fetch Range for {channel_id}: From={api_fetch_start_date_utc}, To={api_fetch_end_date_utc}")

    if api_fetch_start_date_utc >= api_fetch_end_date_utc:
        print(f"Start date {api_fetch_start_date_utc} is not before end date {api_fetch_end_date_utc}. Skipping fetch.")
        #LOGGER.info(f"Data for {channel_id} seems up-to-date. Skipping API fetch."); continue
    if (api_fetch_end_date_utc - api_fetch_start_date_utc) < pd.Timedelta(minutes=5):
        #LOGGER.info(f"Less than 5 mins of data to fetch for {channel_id}. Skipping."); continue
        print(f"Less than 5 mins of data to fetch for {channel_id}. Skipping.")
    api_request_url = (
        f"{API_SERVICE_URL}/Services/DataService.ashx?type=graph&channel={channel_id}"
        f"&DateFrom={api_fetch_start_date_utc.strftime('%Y-%m-%d%%20%H:%M:%S')}"
        f"&DateTo={api_fetch_end_date_utc.strftime('%Y-%m-%d%%20%H:%M:%S')}&Reduction=day"
    )
    #LOGGER.info(f"Fetching data for channel {channel_id}...")
    #LOGGER.debug(f"Request URL (first 150 chars): {api_request_url[:150]}")
    
    try:
        with urlopen(api_request_url) as response:
            response_text = response.read().decode('utf-8').strip()
        if not response_text: 
            print(f"No response content from server for channel {channel_id}. Skipping...")
            #LOGGER.info(f"No response content from API for channel {channel_id}."); continue
        api_data_json = json.loads(response_text)
    except json.JSONDecodeError as e: 
        print(f"Invalid JSON from API for {channel_id}: {e}. Response: '{response_text[:200]}...'")
        #LOGGER.error(f"Invalid JSON from API for {channel_id}: {e}. Response: '{response_text[:200]}...'"); continue
    except Exception as e_fetch_api:
        print(f"Error fetching API data for {channel_id}: {e_fetch_api}")
         #LOGGER.error(f"Error fetching API data for {channel_id}: {e_fetch_api}"); continue

    api_channels_data = api_data_json.get("Channels", [])
    if not api_channels_data or "StringifiedData" not in api_channels_data[0]:
        print(f"API response for {channel_id} missing 'Channels' or 'StringifiedData'.")
        #LOGGER.info(f"API response for {channel_id} missing 'Channels' or 'StringifiedData'."); continue
    
    data_str_from_api = api_channels_data[0]["StringifiedData"]
    if data_str_from_api == "no data" or not data_str_from_api:
        print(f"No actual data string in API response for {channel_id} for the requested period.")
        #LOGGER.info(f"No actual data string in API response for {channel_id} for period."); continue

    try:
        raw_api_data_points = json.loads(data_str_from_api)
    except json.JSONDecodeError as e: 
        print(f"Invalid JSON in 'StringifiedData' for {channel_id}: {e}")
        #LOGGER.error(f"Invalid JSON in 'StringifiedData' for {channel_id}: {e}"); continue
    if not raw_api_data_points: 
        print(f"API 'StringifiedData' is empty list for {channel_id}.")
        #LOGGER.info(f"API 'StringifiedData' is empty list for {channel_id}."); continue

    # --- Process API Timestamps to TRUE UTC ---
    #LOGGER.info('Preparing time series from API, converting to TRUE UTC...')
    newly_fetched_gauge_records = []
    for item in raw_api_data_points:
        timestamp_str, value = item[0], item[1]
        try:
            naive_cph_wall_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            cph_local_aware = pd.Timestamp(naive_cph_wall_time).tz_localize(
                'Europe/Copenhagen', ambiguous='NaT', nonexistent='NaT'
            )
            if pd.isna(cph_local_aware):
                #LOGGER.warning(f"TS '{timestamp_str}' (Ch {channel_id}) NaT after CPH localize. Skipping.")
                continue
            true_utc_timestamp = cph_local_aware.tz_convert('UTC')
            newly_fetched_gauge_records.append({'datetime': true_utc_timestamp, 'value': value})
        except ValueError: 
            print(f"Could not parse timestamp '{timestamp_str}' from API for channel {channel_id}.")
            #LOGGER.warning(f"Parse fail for API TS '{timestamp_str}' (Ch {channel_id}).")
        except Exception as e_ts_proc:
            print(f"Error processing timezone for timestamp '{timestamp_str}' (Ch {channel_id}): {e_ts_proc}") 
            #LOGGER.warning(f"TZ error for API TS '{timestamp_str}' (Ch {channel_id}): {e_ts_proc}")
    
    if not newly_fetched_gauge_records:
        print(f"No valid new records after API processing for {file_item_name}.")
        #LOGGER.info(f"No valid new records after API processing for {file_item_name}."); continue
    
    df_newly_fetched = pd.DataFrame(newly_fetched_gauge_records)
    #LOGGER.info(f"Prepared {len(df_newly_fetched)} new TRUE UTC gauge records for {file_item_name}.")

    # --- Append to CSV ---
    try:
        df_to_write_to_csv = df_newly_fetched
        # Read existing data if file exists and has content
        if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
            try:
                df_existing_from_csv = pd.read_csv(current_file_path) # Read first
                if 'datetime' in df_existing_from_csv.columns and not df_existing_from_csv.empty:
                    df_existing_from_csv['datetime'] = pd.to_datetime(df_existing_from_csv['datetime'], errors='coerce')
                    df_existing_from_csv.dropna(subset=['datetime'], inplace=True)
                    if not df_existing_from_csv.empty:
                        # Ensure existing data is also true UTC-aware
                        if df_existing_from_csv['datetime'].dt.tz is None:
                            df_existing_from_csv['datetime'] = df_existing_from_csv['datetime'].dt.tz_localize('UTC')
                        elif str(df_existing_from_csv['datetime'].dt.tz).upper() != 'UTC':
                            df_existing_from_csv['datetime'] = df_existing_from_csv['datetime'].dt.tz_convert('UTC')
                        
                        if not df_existing_from_csv.empty: # Check after potential tz-related NaT drop
                            df_to_write_to_csv = pd.concat([df_existing_from_csv, df_newly_fetched], ignore_index=True)
            except Exception as e_append_read_csv:
                print(f"Error reading existing CSV for append: {e_append_read_csv}")
                 #LOGGER.error(f"Error reading existing {current_file_path} for append: {e_append_read_csv}. Will attempt to overwrite with new data.")

        if not df_to_write_to_csv.empty:
            df_to_write_to_csv.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
            df_to_write_to_csv.sort_values('datetime', inplace=True)
            
            # Save 'datetime' column as naive UTC strings
            df_to_write_to_csv_final = df_to_write_to_csv.copy()
            if pd.api.types.is_datetime64_any_dtype(df_to_write_to_csv_final['datetime']) and df_to_write_to_csv_final['datetime'].dt.tz is not None:
                df_to_write_to_csv_final['datetime'] = df_to_write_to_csv_final['datetime'].dt.tz_localize(None)
            
            df_to_write_to_csv_final.to_csv(current_file_path, index=False, date_format=DATE_FORMAT_FOR_CSV_SAVE)
            #LOGGER.info(f"Saved data to {current_file_path}. Total records: {len(df_to_write_to_csv_final)}.")
        else:
            print(f"No new data to write for {current_file_path}.")
            #LOGGER.info(f"No data (new or existing) to write for {current_file_path}.")

    except Exception as e_save_logic:
        print(f"Error in save/append logic for {current_file_path}: {e_save_logic}")
        #LOGGER.error(f"Critical error in save/append logic for {current_file_path}: {e_save_logic}")
        traceback.print_exc()

#LOGGER.info("\nGauge data ingestion script finished.")
# --- END OF FILE get_gauge_data3.py ---