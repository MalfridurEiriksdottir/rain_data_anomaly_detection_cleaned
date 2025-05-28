# import os
# import ssl
# from urllib.request import urlopen
# from datetime import datetime # Keep this for strptime
# import json
# import pandas as pd
# from pyproj import Transformer
# import re
# import numpy as np
# import logging
# import traceback # For detailed error logging

# # --- Logger Setup ---
# LOGGER = logging.getLogger(__name__)
# if not LOGGER.hasHandlers(): # Avoid adding multiple handlers if imported or run multiple times
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - GAUGE_INGEST - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     LOGGER.addHandler(handler)
#     LOGGER.setLevel(logging.INFO) # Set to DEBUG for more verbose output during development

# LOGGER.info("Start of the gauge data ingestion script")

# # --- Configuration ---
# SENSOR_METADATA_FILE = 'SensorOversigt.xlsx'
# GAUGE_DATA_FOLDER = "gauge_data"
# API_SERVICE_URL = "https://nkaflob.watermanager.dk/"
# FALLBACK_START_DATE_STR = '2024-01-01 00:00:00' # For new sensors or if file reading fails
# DATE_FORMAT_FOR_CSV_SAVE = "%Y-%m-%d %H:%M:%S" # For saving naive UTC strings
# # HOURS_CORRECTION is no longer needed with dynamic DST adjustment

# # --- Read Sensor Metadata ---
# try:
#     df_sensors = pd.read_excel(SENSOR_METADATA_FILE)
#     LOGGER.info(f"Successfully read sensor metadata from {SENSOR_METADATA_FILE}")
# except FileNotFoundError:
#     LOGGER.critical(f"Fatal: Sensor metadata file '{SENSOR_METADATA_FILE}' not found. Exiting.")
#     exit()
# except Exception as e:
#     LOGGER.critical(f"Fatal: Error reading {SENSOR_METADATA_FILE}: {e}")
#     LOGGER.critical(traceback.format_exc())
#     exit()

# ssl._create_default_https_context = ssl._create_unverified_context

# def parse_wkt(wkt_string):
#     match = re.match(r"Point \(([-\d\.]+) ([-\d\.]+)\)", str(wkt_string)) # Added str() for safety
#     if match: return float(match.group(1)), float(match.group(2))
#     return np.nan, np.nan

# try:
#     if 'wkt_geom' not in df_sensors.columns:
#         raise KeyError("'wkt_geom' column not found in sensor metadata.")
#     df_sensors['longitude'], df_sensors['latitude'] = zip(*df_sensors['wkt_geom'].map(parse_wkt))
#     LOGGER.info("Successfully extracted coordinates from sensor metadata.")
# except KeyError as e:
#     LOGGER.critical(f"Fatal: Missing essential column in sensor metadata: {e}")
#     exit()
# except Exception as e:
#     LOGGER.critical(f"Fatal: Error parsing WKT geometry from sensor metadata: {e}")
#     LOGGER.critical(traceback.format_exc())
#     exit()

# transformer_filename_to_wgs84 = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
# LOGGER.info("Filename to WGS84 coordinate transformer created.")

# if not os.path.exists(GAUGE_DATA_FOLDER):
#     LOGGER.info(f"Gauge data folder {GAUGE_DATA_FOLDER} does not exist. Creating it.")
#     os.makedirs(GAUGE_DATA_FOLDER, exist_ok=True)

# gauge_files_in_dir = os.listdir(GAUGE_DATA_FOLDER)
# LOGGER.info(f"Found {len(gauge_files_in_dir)} items in {GAUGE_DATA_FOLDER}")

# def get_channel_from_coords(df_sensors_ref, lon_val, lat_val, filename_for_log=""):
#     tolerance = 1e-5
#     required_cols = ['longitude', 'latitude', 'Channel']
#     if not all(col in df_sensors_ref.columns for col in required_cols):
#         LOGGER.error(f"Sensor metadata missing one of required columns: {required_cols} for {filename_for_log}")
#         return None

#     match = df_sensors_ref[
#         (np.isclose(df_sensors_ref['longitude'], lon_val, atol=tolerance)) &
#         (np.isclose(df_sensors_ref['latitude'], lat_val, atol=tolerance))
#     ]
#     if match.empty:
#         LOGGER.warning(f"No channel match in metadata for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f})")
#         return None
#     elif len(match) > 1:
#         LOGGER.warning(f"Multiple channel matches for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f}). Using first: {match.iloc[0]['Channel']}")
#     return match.iloc[0]['Channel']

# # --- Main Loop ---
# for file_item_name in gauge_files_in_dir: # Iterate over items found in directory
#     LOGGER.info(f"\nProcessing file: {file_item_name}")
#     if not file_item_name.lower().endswith('.csv') or not file_item_name.lower().startswith('gaugedata_'):
#         LOGGER.info(f"Skipping non-standard gauge file: {file_item_name}")
#         continue

#     filename_no_ext = file_item_name.replace('.csv', '').replace('.CSV','')
#     try:
#         parts = filename_no_ext.split('_')
#         if len(parts) < 3: raise ValueError("Filename format error (not enough parts)")
#         x_coord_from_filename, y_coord_from_filename = float(parts[1]), float(parts[2])
#         LOGGER.debug(f"Parsed coordinates from filename: {x_coord_from_filename}, {y_coord_from_filename}")
#     except (ValueError, IndexError) as e_fn:
#         LOGGER.warning(f"Could not parse coordinates from filename '{file_item_name}': {e_fn}. Skipping.")
#         continue

#     try:
#         lon_wgs84, lat_wgs84 = transformer_filename_to_wgs84.transform(x_coord_from_filename, y_coord_from_filename)
#     except Exception as e_trans:
#         LOGGER.error(f"Error transforming coordinates for {file_item_name}: {e_trans}. Skipping.")
#         LOGGER.debug(traceback.format_exc())
#         continue

#     channel_id = get_channel_from_coords(df_sensors, lon_wgs84, lat_wgs84, file_item_name)
#     if channel_id is None:
#         LOGGER.warning(f"No channel identified for {file_item_name}. Skipping API fetch for this file.")
#         continue
#     LOGGER.info(f"Matched channel {channel_id} for file {file_item_name}")

#     # --- Determine Start Date for API Fetch ---
#     current_file_path = os.path.join(GAUGE_DATA_FOLDER, file_item_name)
#     api_fetch_start_date_utc = pd.Timestamp(FALLBACK_START_DATE_STR, tz='UTC') # Default

#     if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
#         try:
#             df_existing_gauge = pd.read_csv(current_file_path)
#             if 'datetime' in df_existing_gauge.columns and not df_existing_gauge.empty:
#                 df_existing_gauge['datetime'] = pd.to_datetime(df_existing_gauge['datetime'], errors='coerce')
#                 df_existing_gauge.dropna(subset=['datetime'], inplace=True)

#                 if not df_existing_gauge.empty:
#                     # Assume CSV stores naive strings representing TRUE UTC
#                     if df_existing_gauge['datetime'].dt.tz is None:
#                         LOGGER.debug(f"Localizing naive datetimes from {file_item_name} to UTC.")
#                         df_existing_gauge['datetime'] = df_existing_gauge['datetime'].dt.tz_localize('UTC', ambiguous='NaT')
#                         df_existing_gauge.dropna(subset=['datetime'], inplace=True) # Drop NaT if any occurred
#                     elif str(df_existing_gauge['datetime'].dt.tz).upper() != 'UTC':
#                         LOGGER.debug(f"Converting datetimes from {file_item_name} (tz={df_existing_gauge['datetime'].dt.tz}) to UTC.")
#                         df_existing_gauge['datetime'] = df_existing_gauge['datetime'].dt.tz_convert('UTC')

#                     if not df_existing_gauge.empty: # After potential NaT drop
#                         last_ts = df_existing_gauge['datetime'].max() # This is UTC-aware
#                         if pd.notna(last_ts):
#                             api_fetch_start_date_utc = last_ts + pd.Timedelta(microseconds=1) # UTC-aware
#             else:
#                 LOGGER.info(f"File {file_item_name} is empty or missing 'datetime' column. Using fallback start.")
#         except Exception as e_read_file:
#             LOGGER.warning(f"Error reading or processing existing {file_item_name}: {e_read_file}. Using fallback start.")
#             LOGGER.debug(traceback.format_exc())
#     else:
#         LOGGER.info(f"File {file_item_name} not found or empty. Assuming new sensor.")

#     api_fetch_end_date_utc = pd.Timestamp.now(tz='UTC') # Aware UTC
#     LOGGER.info(f"API Fetch Range for {channel_id}: From={api_fetch_start_date_utc}, To={api_fetch_end_date_utc}")

#     if api_fetch_start_date_utc >= api_fetch_end_date_utc:
#         LOGGER.info(f"Data for {channel_id} ({file_item_name}) seems up-to-date. Start date {api_fetch_start_date_utc} is not before end date {api_fetch_end_date_utc}. Skipping API fetch.")
#         continue
#     if (api_fetch_end_date_utc - api_fetch_start_date_utc) < pd.Timedelta(minutes=5):
#         LOGGER.info(f"Less than 5 mins of data to fetch for {channel_id} ({file_item_name}). Skipping API fetch.")
#         continue

#     # Timestamps in URL are formatted as naive strings, but represent the UTC time
#     api_request_url = (
#         f"{API_SERVICE_URL}/Services/DataService.ashx?type=graph&channel={channel_id}"
#         f"&DateFrom={api_fetch_start_date_utc.strftime('%Y-%m-%d%%20%H:%M:%S')}"
#         f"&DateTo={api_fetch_end_date_utc.strftime('%Y-%m-%d%%20%H:%M:%S')}&Reduction=day"
#     )
#     LOGGER.info(f"Fetching data for channel {channel_id}...")
#     LOGGER.debug(f"Request URL (first 150 chars): {api_request_url[:150]}")

#     response_text = ""
#     try:
#         with urlopen(api_request_url) as response:
#             response_text = response.read().decode('utf-8').strip()
#         if not response_text:
#             LOGGER.info(f"No response content from API for channel {channel_id}. Skipping.")
#             continue
#         api_data_json = json.loads(response_text)
#     except json.JSONDecodeError as e:
#         LOGGER.error(f"Invalid JSON from API for {channel_id}: {e}. Response: '{response_text[:200]}...'")
#         continue
#     except Exception as e_fetch_api:
#         LOGGER.error(f"Error fetching API data for {channel_id}: {e_fetch_api}")
#         LOGGER.debug(traceback.format_exc())
#         continue

#     api_channels_data = api_data_json.get("Channels", [])
#     if not api_channels_data or "StringifiedData" not in api_channels_data[0]:
#         LOGGER.info(f"API response for {channel_id} missing 'Channels' or 'StringifiedData'.")
#         continue

#     data_str_from_api = api_channels_data[0]["StringifiedData"]
#     if data_str_from_api == "no data" or not data_str_from_api:
#         LOGGER.info(f"No actual data string in API response for {channel_id} for the requested period.")
#         continue

#     try:
#         raw_api_data_points = json.loads(data_str_from_api)
#     except json.JSONDecodeError as e:
#         LOGGER.error(f"Invalid JSON in 'StringifiedData' for {channel_id}: {e}")
#         continue
#     if not raw_api_data_points:
#         LOGGER.info(f"API 'StringifiedData' is empty list for {channel_id}.")
#         continue

#     # --- Process API Timestamps to TRUE UTC ---
#     LOGGER.info(f'Preparing time series from API for channel {channel_id}, converting to TRUE UTC and applying dynamic DST correction...')
#     newly_fetched_gauge_records = []
#     for item in raw_api_data_points:
#         timestamp_str, value = item[0], item[1]
#         try:
#             # API gives naive local Danish time strings
#             naive_cph_wall_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
#             # Localize to Copenhagen time, handling DST ambiguities by dropping
#             cph_local_aware = pd.Timestamp(naive_cph_wall_time).tz_localize(
#                 'Europe/Copenhagen', ambiguous='NaT', nonexistent='NaT'
#             )
#             if pd.isna(cph_local_aware):
#                 LOGGER.warning(f"Timestamp '{timestamp_str}' (Ch {channel_id}) resulted in NaT after CPH localization (likely DST ambiguity). Skipping record.")
#                 continue
            
#             # Convert to UTC (intermediate step)
#             utc_intermediate = cph_local_aware.tz_convert('UTC')

#             # --- APPLY THE DYNAMIC DST CORRECTION ---
#             # Get the DST offset from the Copenhagen-localized time
#             # cph_local_aware.dst() returns a timedelta (e.g., 1 hour in summer, 0 in winter for CPH)
#             dst_offset = cph_local_aware.dst()
#             dst_offset = None
#             if dst_offset is None: # Should not happen with 'Europe/Copenhagen' for valid dates
#                 LOGGER.warning(f"Could not determine DST offset for timestamp {cph_local_aware} (Ch {channel_id}). Assuming no DST offset.")
#                 dst_offset = pd.Timedelta(hours=0)
            
#             # Add this DST offset to the intermediate UTC time
#             corrected_utc_timestamp = utc_intermediate + dst_offset
#             # ---

#             newly_fetched_gauge_records.append({'datetime': corrected_utc_timestamp, 'value': value})
#         except ValueError:
#             LOGGER.warning(f"Could not parse timestamp '{timestamp_str}' from API for channel {channel_id}. Skipping record.")
#         except Exception as e_ts_proc:
#             LOGGER.warning(f"Error processing timezone for timestamp '{timestamp_str}' (Ch {channel_id}): {e_ts_proc}. Skipping record.")
#             LOGGER.debug(traceback.format_exc())

#     if not newly_fetched_gauge_records:
#         LOGGER.info(f"No valid new records after API processing for {file_item_name}.")
#         continue

#     df_newly_fetched = pd.DataFrame(newly_fetched_gauge_records) # 'datetime' column is corrected UTC-aware
#     LOGGER.info(f"Prepared {len(df_newly_fetched)} new corrected TRUE UTC gauge records for {file_item_name}.")

#     # --- Append to CSV ---
#     try:
#         df_to_write_to_csv = df_newly_fetched # df_newly_fetched has corrected UTC-aware datetimes
#         # Read existing data if file exists and has content
#         if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
#             try:
#                 df_existing_from_csv = pd.read_csv(current_file_path)
#                 if 'datetime' in df_existing_from_csv.columns and not df_existing_from_csv.empty:
#                     df_existing_from_csv['datetime'] = pd.to_datetime(df_existing_from_csv['datetime'], errors='coerce')
#                     df_existing_from_csv.dropna(subset=['datetime'], inplace=True)
#                     if not df_existing_from_csv.empty:
#                         # Ensure existing data is also true UTC-aware
#                         if df_existing_from_csv['datetime'].dt.tz is None: # Naive strings from CSV are assumed UTC
#                             df_existing_from_csv['datetime'] = df_existing_from_csv['datetime'].dt.tz_localize('UTC')
#                         elif str(df_existing_from_csv['datetime'].dt.tz).upper() != 'UTC': # Convert if other TZ
#                             df_existing_from_csv['datetime'] = df_existing_from_csv['datetime'].dt.tz_convert('UTC')
                        
#                         if not df_existing_from_csv.empty: # Check after potential tz-related NaT drop
#                             df_to_write_to_csv = pd.concat([df_existing_from_csv, df_newly_fetched], ignore_index=True)
#             except Exception as e_append_read_csv:
#                  LOGGER.error(f"Error reading existing {current_file_path} for append: {e_append_read_csv}. Will attempt to process with new data only.")
#                  LOGGER.debug(traceback.format_exc())
#                  # Continue with df_to_write_to_csv being just df_newly_fetched

#         if not df_to_write_to_csv.empty:
#             df_to_write_to_csv.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
#             df_to_write_to_csv.sort_values('datetime', inplace=True)

#             # Save 'datetime' column as naive UTC strings
#             df_to_write_to_csv_final = df_to_write_to_csv.copy()
#             # Ensure datetime column is timezone-aware UTC before converting to naive for saving
#             if pd.api.types.is_datetime64_any_dtype(df_to_write_to_csv_final['datetime']):
#                 if df_to_write_to_csv_final['datetime'].dt.tz is None:
#                     LOGGER.warning(f"Datetime column for {current_file_path} was naive before saving; assuming UTC and localizing to None.")
#                 elif str(df_to_write_to_csv_final['datetime'].dt.tz).upper() != 'UTC':
#                     LOGGER.warning(f"Datetime column for {current_file_path} was {df_to_write_to_csv_final['datetime'].dt.tz} before saving; converting to UTC then naive.")
#                     df_to_write_to_csv_final['datetime'] = df_to_write_to_csv_final['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
#                 else: # It is UTC-aware, make it naive
#                     df_to_write_to_csv_final['datetime'] = df_to_write_to_csv_final['datetime'].dt.tz_localize(None)
            
#             df_to_write_to_csv_final.to_csv(current_file_path, index=False, date_format=DATE_FORMAT_FOR_CSV_SAVE)
#             LOGGER.info(f"Saved data to {current_file_path}. Total records: {len(df_to_write_to_csv_final)}.")
#         else:
#             LOGGER.info(f"No data (new or existing) to write for {current_file_path}.")

#     except Exception as e_save_logic:
#         LOGGER.error(f"Critical error in save/append logic for {current_file_path}: {e_save_logic}")
#         LOGGER.error(traceback.format_exc())

# LOGGER.info("\nGauge data ingestion script finished.")












# --- START OF ENTIRE gg.py (Minimal Time Conversion Mode) ---

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
LOGGER = logging.getLogger(__name__)
if not LOGGER.hasHandlers(): # Avoid adding multiple handlers if imported or run multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - GAUGE_INGEST - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO) # Set to DEBUG for more verbose output during development

LOGGER.info("Start of the gauge data ingestion script (Minimal Time Conversion Mode)")

# --- Configuration ---
SENSOR_METADATA_FILE = 'SensorOversigt.xlsx'
GAUGE_DATA_FOLDER = "gauge_data" # Assumes this is a sibling to the script's parent directory (e.g., ../gauge_data if script is in get_data2/)
# If GAUGE_DATA_FOLDER is a subdirectory of the script's directory, use:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# GAUGE_DATA_FOLDER = os.path.join(SCRIPT_DIR, "gauge_data")


API_SERVICE_URL = "https://nkaflob.watermanager.dk/"
FALLBACK_START_DATE_STR = '2024-01-01 00:00:00' # For new sensors or if file reading fails
DATE_FORMAT_FOR_CSV_SAVE = "%Y-%m-%d %H:%M:%S" # For saving naive datetime strings

# **DOCUMENTATION:** Naive timestamps in CSV files will represent local time
# as provided by the API, assumed to be 'Europe/Copenhagen' local time.
NAIVE_CSV_TIMEZONE_ASSUMPTION = 'Europe/Copenhagen'

# --- Read Sensor Metadata ---
try:
    df_sensors = pd.read_excel(SENSOR_METADATA_FILE)
    LOGGER.info(f"Successfully read sensor metadata from {SENSOR_METADATA_FILE}")
except FileNotFoundError:
    LOGGER.critical(f"Fatal: Sensor metadata file '{SENSOR_METADATA_FILE}' not found. Exiting.")
    exit()
except Exception as e:
    LOGGER.critical(f"Fatal: Error reading {SENSOR_METADATA_FILE}: {e}")
    LOGGER.critical(traceback.format_exc())
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
    LOGGER.info("Successfully extracted coordinates from sensor metadata.")
except KeyError as e:
    LOGGER.critical(f"Fatal: Missing essential column in sensor metadata: {e}")
    exit()
except Exception as e:
    LOGGER.critical(f"Fatal: Error parsing WKT geometry from sensor metadata: {e}")
    LOGGER.critical(traceback.format_exc())
    exit()

transformer_filename_to_wgs84 = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)
LOGGER.info("Filename to WGS84 coordinate transformer created.")

# Adjust GAUGE_DATA_FOLDER path to be relative to the script's parent directory if needed
# Example: If script is in 'get_data2' and gauge_data is '../gauge_data'
script_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAUGE_DATA_FOLDER_ABSOLUTE = os.path.join(script_parent_dir, os.path.basename(GAUGE_DATA_FOLDER))

if not os.path.exists(GAUGE_DATA_FOLDER_ABSOLUTE):
    LOGGER.info(f"Gauge data folder {GAUGE_DATA_FOLDER_ABSOLUTE} does not exist. Creating it.")
    os.makedirs(GAUGE_DATA_FOLDER_ABSOLUTE, exist_ok=True)
else:
    LOGGER.info(f"Using existing gauge data folder: {GAUGE_DATA_FOLDER_ABSOLUTE}")


try:
    gauge_files_in_dir = os.listdir(GAUGE_DATA_FOLDER_ABSOLUTE)
    LOGGER.info(f"Found {len(gauge_files_in_dir)} items in {GAUGE_DATA_FOLDER_ABSOLUTE}")
except FileNotFoundError:
    LOGGER.error(f"Could not list files in {GAUGE_DATA_FOLDER_ABSOLUTE}. Check path. Exiting.")
    gauge_files_in_dir = [] # Ensure it's iterable
    exit()


def get_channel_from_coords(df_sensors_ref, lon_val, lat_val, filename_for_log=""):
    tolerance = 1e-5
    required_cols = ['longitude', 'latitude', 'Channel']
    if not all(col in df_sensors_ref.columns for col in required_cols):
        LOGGER.error(f"Sensor metadata missing one of required columns: {required_cols} for {filename_for_log}")
        return None

    match = df_sensors_ref[
        (np.isclose(df_sensors_ref['longitude'], lon_val, atol=tolerance)) &
        (np.isclose(df_sensors_ref['latitude'], lat_val, atol=tolerance))
    ]
    if match.empty:
        LOGGER.warning(f"No channel match in metadata for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f})")
        return None
    elif len(match) > 1:
        LOGGER.warning(f"Multiple channel matches for {filename_for_log} (lon {lon_val:.5f}, lat {lat_val:.5f}). Using first: {match.iloc[0]['Channel']}")
    return match.iloc[0]['Channel']

# --- Main Loop ---
for file_item_name in gauge_files_in_dir: # Iterate over items found in directory
    LOGGER.info(f"\nProcessing file: {file_item_name}")
    if not file_item_name.lower().endswith('.csv') or not file_item_name.lower().startswith('gaugedata_'):
        LOGGER.info(f"Skipping non-standard gauge file: {file_item_name}")
        continue

    filename_no_ext = file_item_name.replace('.csv', '').replace('.CSV','')
    try:
        parts = filename_no_ext.split('_')
        if len(parts) < 3: raise ValueError("Filename format error (not enough parts)")
        x_coord_from_filename, y_coord_from_filename = float(parts[1]), float(parts[2])
        LOGGER.debug(f"Parsed coordinates from filename: {x_coord_from_filename}, {y_coord_from_filename}")
    except (ValueError, IndexError) as e_fn:
        LOGGER.warning(f"Could not parse coordinates from filename '{file_item_name}': {e_fn}. Skipping.")
        continue

    try:
        lon_wgs84, lat_wgs84 = transformer_filename_to_wgs84.transform(x_coord_from_filename, y_coord_from_filename)
    except Exception as e_trans:
        LOGGER.error(f"Error transforming coordinates for {file_item_name}: {e_trans}. Skipping.")
        LOGGER.debug(traceback.format_exc())
        continue

    channel_id = get_channel_from_coords(df_sensors, lon_wgs84, lat_wgs84, file_item_name)
    if channel_id is None:
        LOGGER.warning(f"No channel identified for {file_item_name}. Skipping API fetch for this file.")
        continue
    LOGGER.info(f"Matched channel {channel_id} for file {file_item_name}")

    # --- Determine Start Date for API Fetch ---
    current_file_path = os.path.join(GAUGE_DATA_FOLDER_ABSOLUTE, file_item_name)
    
    api_fetch_start_date_naive = pd.to_datetime(FALLBACK_START_DATE_STR) # Naive datetime

    if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
        try:
            df_existing_gauge = pd.read_csv(current_file_path)
            if 'datetime' in df_existing_gauge.columns and not df_existing_gauge.empty:
                # Assume 'datetime' in CSV is a naive string representing NAIVE_CSV_TIMEZONE_ASSUMPTION
                df_existing_gauge['datetime_parsed'] = pd.to_datetime(df_existing_gauge['datetime'], errors='coerce')
                df_existing_gauge.dropna(subset=['datetime_parsed'], inplace=True)

                if not df_existing_gauge.empty:
                    last_ts_naive = df_existing_gauge['datetime_parsed'].max() # This is a naive datetime
                    if pd.notna(last_ts_naive):
                        api_fetch_start_date_naive = last_ts_naive + pd.Timedelta(microseconds=1) # Still naive
            else:
                LOGGER.info(f"File {file_item_name} is empty or missing 'datetime' column. Using fallback start (naive: {api_fetch_start_date_naive}).")
        except Exception as e_read_file:
            LOGGER.warning(f"Error reading or processing existing {file_item_name}: {e_read_file}. Using fallback start (naive: {api_fetch_start_date_naive}).")
            LOGGER.debug(traceback.format_exc())
    else:
        LOGGER.info(f"File {file_item_name} not found or empty. Assuming new sensor, using fallback start (naive: {api_fetch_start_date_naive}).")

    api_fetch_end_date_naive = pd.Timestamp.now().replace(tzinfo=None) # Current naive local time

    # --- IMPORTANT ASSUMPTION FOR API PARAMETERS ---
    # This section assumes the API's DateFrom and DateTo parameters expect NAIVE LOCAL TIME STRINGS.
    # If the API expects UTC strings for DateFrom/DateTo, you MUST convert
    # api_fetch_start_date_naive and api_fetch_end_date_naive to UTC here.
    # Example conversion to UTC for API parameters (if needed):
    # import pytz
    # cph_tz = pytz.timezone(NAIVE_CSV_TIMEZONE_ASSUMPTION)
    # try:
    #     start_utc_for_api = cph_tz.localize(api_fetch_start_date_naive, is_dst=None).astimezone(pytz.UTC)
    #     end_utc_for_api = cph_tz.localize(api_fetch_end_date_naive, is_dst=None).astimezone(pytz.UTC)
    #     date_from_str_for_api = start_utc_for_api.strftime('%Y-%m-%d%%20%H:%M:%S')
    #     date_to_str_for_api = end_utc_for_api.strftime('%Y-%m-%d%%20%H:%M:%S')
    #     LOGGER.info(f"Using UTC for API params: From={date_from_str_for_api}, To={date_to_str_for_api}")
    # except (pytz.exceptions.AmbiguousTimeError, pytz.exceptions.NonExistentTimeError) as e_tz:
    #     LOGGER.error(f"Timezone conversion error for API params for {channel_id}: {e_tz}. Skipping.")
    #     continue # Skip this sensor if API params can't be formed correctly
    # --- END OF EXAMPLE UTC CONVERSION FOR API ---

    # Using naive local time strings for API parameters (DEFAULT BEHAVIOR OF THIS SCRIPT)
    date_from_str_for_api = api_fetch_start_date_naive.strftime('%Y-%m-%d%%20%H:%M:%S')
    date_to_str_for_api = api_fetch_end_date_naive.strftime('%Y-%m-%d%%20%H:%M:%S')


    LOGGER.info(f"API Fetch Range for {channel_id} (Naive, assumed {NAIVE_CSV_TIMEZONE_ASSUMPTION}, formatted for API): From={date_from_str_for_api}, To={date_to_str_for_api}")

    if api_fetch_start_date_naive >= api_fetch_end_date_naive:
        LOGGER.info(f"Data for {channel_id} ({file_item_name}) seems up-to-date. Start date {api_fetch_start_date_naive} is not before end date {api_fetch_end_date_naive}. Skipping API fetch.")
        continue
    if (api_fetch_end_date_naive - api_fetch_start_date_naive) < pd.Timedelta(minutes=5): # Comparison of naive datetimes
        LOGGER.info(f"Less than 5 mins of data to fetch for {channel_id} ({file_item_name}). Skipping API fetch.")
        continue

    api_request_url = (
        f"{API_SERVICE_URL}/Services/DataService.ashx?type=graph&channel={channel_id}"
        f"&DateFrom={date_from_str_for_api}"
        f"&DateTo={date_to_str_for_api}&Reduction=day" # Consider Reduction=no for higher resolution if needed
    )
    LOGGER.info(f"Fetching data for channel {channel_id}...")
    LOGGER.debug(f"Request URL (first 150 chars): {api_request_url[:150]}")

    response_text = ""
    try:
        with urlopen(api_request_url) as response:
            response_text = response.read().decode('utf-8').strip()
        if not response_text:
            LOGGER.info(f"No response content from API for channel {channel_id}. Skipping.")
            continue
        api_data_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        LOGGER.error(f"Invalid JSON from API for {channel_id}: {e}. Response: '{response_text[:200]}...'")
        continue
    except Exception as e_fetch_api:
        LOGGER.error(f"Error fetching API data for {channel_id}: {e_fetch_api}")
        LOGGER.debug(traceback.format_exc())
        continue

    api_channels_data = api_data_json.get("Channels", [])
    if not api_channels_data or "StringifiedData" not in api_channels_data[0]:
        LOGGER.info(f"API response for {channel_id} missing 'Channels' or 'StringifiedData'.")
        continue

    data_str_from_api = api_channels_data[0]["StringifiedData"]
    if data_str_from_api == "no data" or not data_str_from_api:
        LOGGER.info(f"No actual data string in API response for {channel_id} for the requested period.")
        continue

    try:
        raw_api_data_points = json.loads(data_str_from_api)
    except json.JSONDecodeError as e:
        LOGGER.error(f"Invalid JSON in 'StringifiedData' for {channel_id}: {e}")
        continue
    if not raw_api_data_points:
        LOGGER.info(f"API 'StringifiedData' is empty list for {channel_id}.")
        continue

    # --- Process API Timestamps - Minimal Conversion ---
    LOGGER.info(f'Preparing time series from API for channel {channel_id} (storing as naive datetimes)')
    newly_fetched_gauge_records = []
    for item in raw_api_data_points:
        timestamp_str, value = item[0], item[1]
        try:
            # Parse the API string into a naive datetime object
            # This assumes the API string is in "%Y-%m-%d %H:%M:%S" format
            naive_datetime_from_api = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            
            # We store this naive_datetime_from_api. No tz_localize, no tz_convert.
            newly_fetched_gauge_records.append({'datetime': naive_datetime_from_api, 'value': value})

        except ValueError:
            LOGGER.warning(f"Could not parse timestamp string '{timestamp_str}' from API for channel {channel_id} into a naive datetime. Skipping record.")
        except Exception as e_ts_proc:
            LOGGER.warning(f"Error processing timestamp string '{timestamp_str}' (Ch {channel_id}): {e_ts_proc}. Skipping record.")
            LOGGER.debug(traceback.format_exc())

    if not newly_fetched_gauge_records:
        LOGGER.info(f"No valid new records after API processing for {file_item_name}.")
        continue

    df_newly_fetched = pd.DataFrame(newly_fetched_gauge_records)
    # 'datetime' column in df_newly_fetched is now naive datetime objects
    LOGGER.info(f"Prepared {len(df_newly_fetched)} new naive gauge records for {file_item_name}.")

    # --- Append to CSV ---
    try:
        df_to_write_to_csv = df_newly_fetched
        if os.path.exists(current_file_path) and os.path.getsize(current_file_path) > 0:
            try:
                # When reading existing, parse 'datetime' into naive datetime objects
                df_existing_from_csv = pd.read_csv(current_file_path, parse_dates=['datetime'])
                
                if 'datetime' in df_existing_from_csv.columns and not df_existing_from_csv.empty:
                    # Ensure it's actually parsed to datetime64[ns]
                    if not pd.api.types.is_datetime64_any_dtype(df_existing_from_csv['datetime']):
                         LOGGER.warning(f"Column 'datetime' in {current_file_path} was not parsed as datetime. Attempting conversion.")
                         df_existing_from_csv['datetime'] = pd.to_datetime(df_existing_from_csv['datetime'], errors='coerce')
                         df_existing_from_csv.dropna(subset=['datetime'], inplace=True)

                    if not df_existing_from_csv.empty and pd.api.types.is_datetime64_any_dtype(df_existing_from_csv['datetime']):
                        df_to_write_to_csv = pd.concat([df_existing_from_csv, df_newly_fetched], ignore_index=True)
                    else:
                        LOGGER.warning(f"Could not properly use existing data from {current_file_path}. Overwriting with new data if any.")
                else:
                    LOGGER.info(f"Existing file {current_file_path} has no valid 'datetime' data. Overwriting with new data if any.")
            except Exception as e_append_read_csv:
                 LOGGER.error(f"Error reading existing {current_file_path} for append: {e_append_read_csv}. Will attempt to process with new data only.")
                 LOGGER.debug(traceback.format_exc())

        if not df_to_write_to_csv.empty:
            # Ensure 'datetime' is datetime64 before drop_duplicates and sort
            if not pd.api.types.is_datetime64_any_dtype(df_to_write_to_csv['datetime']):
                df_to_write_to_csv['datetime'] = pd.to_datetime(df_to_write_to_csv['datetime'], errors='coerce')
                df_to_write_to_csv.dropna(subset=['datetime'], inplace=True) # Drop if conversion failed

            if not df_to_write_to_csv.empty: # Check again after potential drop
                df_to_write_to_csv.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
                df_to_write_to_csv.sort_values('datetime', inplace=True)
                
                # The 'datetime' column is already naive datetime objects.
                # to_csv will format them using date_format.
                df_to_write_to_csv.to_csv(current_file_path, index=False, date_format=DATE_FORMAT_FOR_CSV_SAVE)
                LOGGER.info(f"Saved data to {current_file_path}. Total records: {len(df_to_write_to_csv)} (naive datetimes representing {NAIVE_CSV_TIMEZONE_ASSUMPTION}).")
            else:
                LOGGER.info(f"No valid data to write for {current_file_path} after attempting to combine and clean.")
        else:
            LOGGER.info(f"No data (new or existing) to write for {current_file_path}.")

    except Exception as e_save_logic:
        LOGGER.error(f"Critical error in save/append logic for {current_file_path}: {e_save_logic}")
        LOGGER.error(traceback.format_exc())

LOGGER.info(f"\nGauge data ingestion script finished (Minimal Time Conversion Mode - naive times assumed to be {NAIVE_CSV_TIMEZONE_ASSUMPTION}).")
# --- END OF ENTIRE gg.py (Minimal Time Conversion Mode) ---