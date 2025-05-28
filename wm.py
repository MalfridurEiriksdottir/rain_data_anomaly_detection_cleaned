# '''

# 11434 VeVa radar SVK 5920 Parkvej
# 11435 VeVa radar SVK 5930 Jakobshavn
# 11436 VeVa radar Regnmåler - 20. Næstelsø Præstemark
# 11437 VeVa radar Regnmåler - 8: Kærvej 2
# 11438 Justeret radar SVK 5920 Parkvej
# 11439 Justeret radar SVK 5930 Jakobshavn
# 11440 Justeret radar Regnmåler - 20. Næstelsø Præstemark
# 11441 Justeret radar Regnmåler - 8: Kærvej 2
# 11442 Regnmålerfejl SVK 5920 Parkvej
# 11443 Regnmålerfejl SVK 5930 Jakobshavn
# 11444 Regnmålerfejl Regnmåler - 20. Næstelsø Præstemark
# 11445 Regnmålerfejl Regnmåler - 8: Kærvej 2

# '''


# import csv
# import requests
# import certifi
# from datetime import datetime, timezone

# # ── Configuration ────────────────────────────────────────────────────────
# API_URL = "https://watermanager.dk/api/Data/UpdateDataList"
# API_KEY = "%bFavm2i*m%7gp&4r*sGDAL^N@$boZKxnbA$GY1vaG6q2!Q%R3m9hZuQ*hsOpJ5"
# INSTALLATION_ID = "47"
# CSV_FILE = "results_files/detailed_sensor_output/674693_6133172_data_adjustments.csv"

# # Map CSV columns to channel IDs
# CHANNEL_MAP = {
#     "Radar_Data_mm_per_min":   11437,
#     "Final_Adjusted_Rainfall": 11441,
#     "Flagged":                 11445,
# }

# # Required headers for every request
# HEADERS = {
#     "accept":         "*/*",
#     "apiKey":         API_KEY,
#     "InstallationId": INSTALLATION_ID,
#     "Content-Type":   "application/json",
# }

# # ── Helpers ──────────────────────────────────────────────────────────────
# def isoformat_z(ts_str: str) -> str:
#     """
#     Convert 'YYYY-MM-DD HH:MM:SS+0000' → 'YYYY-MM-DDTHH:MM:SSZ'
#     """
#     ts_clean = ts_str.replace(" ", "T")
#     dt = datetime.fromisoformat(ts_clean)
#     return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

# def build_payload(csv_path: str) -> list:
#     print(f"Reading CSV file: {csv_path}")
#     payload = []
#     with open(csv_path, newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         print("CSV columns:", reader.fieldnames)
#         for row in reader:
#             log_date = isoformat_z(row["time"])
#             for col, channel in CHANNEL_MAP.items():
#                 raw = row.get(col, "")
#                 if raw in ("", None):
#                     continue
#                 # Convert Flagged True/False → 1/0
#                 if col == "Flagged":
#                     val = 1 if raw.lower() == "true" else 0
#                 else:
#                     val = float(raw)
#                 payload.append({
#                     "channel":  channel,
#                     "logDate":  log_date,
#                     "value":    val,
#                     "valueOk":  True,
#                     "manual":   True
#                 })
#     print(f"Total records to send: {len(payload)}")
#     return payload

# def post_data(batch, batch_num=None):
#     label = f"[Batch {batch_num}] " if batch_num else ""
#     print(f"{label}Sending {len(batch)} records…")
#     resp = requests.post(
#         API_URL,
#         json=batch,
#         headers=HEADERS,
#         verify=False  # trusts both Mozilla & your Windows store
#     )
#     print(f"{label}Status code: {resp.status_code}")
#     if resp.text:
#         try:
#             print(f"{label}Response JSON:", resp.json())
#         except ValueError:
#             print(f"{label}Non-JSON response:", resp.text)
#     else:
#         print(f"{label}Empty response body; assuming success.")
#     print(f"{label}{'✅ Success' if resp.ok else f'❌ Error {resp.status_code}'}\n")

# # ── Main ─────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("Starting data upload…")
#     records = build_payload(CSV_FILE)

#     BATCH_SIZE = 10_000
#     total = len(records)
#     for i in range(0, total, BATCH_SIZE):
#         batch = records[i : i + BATCH_SIZE]
#         batch_num = (i // BATCH_SIZE) + 1
#         post_data(batch, batch_num)


# --- START OF FILE wm.py (Corrected) ---
import csv
import requests
from datetime import datetime, timezone as dt_timezone # Alias to avoid conflict
import os
import re
import json
from pathlib import Path
import logging
import time
import pandas as pd
import traceback

# --- Logger Setup ---
logger = logging.getLogger("WATERMANAGER_PUSH")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    # Add a FileHandler
    log_file_wm = Path("watermanager_push.log") # Define log file path
    fh = logging.FileHandler(log_file_wm, mode='a') # Append mode
    formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter_fh)
    logger.addHandler(fh)

    # Add a StreamHandler (for console output)
    ch = logging.StreamHandler()
    formatter_ch = logging.Formatter('%(levelname)s: %(message)s') # Simpler for console
    ch.setFormatter(formatter_ch)
    logger.addHandler(ch)


# --- Configuration ---
API_URL = "https://watermanager.dk/api/Data/UpdateDataList"
# IMPORTANT: Store API_KEY securely, e.g., in environment variable or a secure config file NOT committed to Git
API_KEY = os.environ.get("WATERMANAGER_API_KEY", "%bFavm2i*m%7gp&4r*sGDAL^N@$boZKxnbA$GY1vaG6q2!Q%R3m9hZuQ*hsOpJ5") 
INSTALLATION_ID = "47"

# Directory containing the detailed sensor output CSVs
# Make sure this path is correct relative to where wm.py is run OR use absolute paths from a config system
DETAILED_OUTPUT_DIR = "results_files/detailed_sensor_output"
# Example: If main_v3_batch3_live.py saves to "daily_outputs_YYYYMMDD_HHMM"
# you might need to find the latest such folder or process all of them.
# For simplicity, this script will look for files directly in DETAILED_OUTPUT_DIR or its subdirs.

SENSOR_TO_WM_CHANNEL_MAPPING_FILE = "watermanager_channel_config.json" 

DETAILED_OUTPUT_DIR = Path("results_files/detailed_sensor_output")
SENSOR_TO_WM_CHANNEL_MAPPING_FILE = Path("watermanager_channel_config.json")

# These are the *types* of data, the actual channel ID will come from the mapping file
# Keys are CSV column names, values are the keys used in watermanager_channel_config.json for channel types
CSV_COLUMN_TO_WM_TYPE_KEY_MAP = {
    "Radar_Data_mm_per_min":   "raw_radar_channel",
    "Final_Adjusted_Rainfall": "adjusted_rainfall_channel",
    "Flagged":                 "flag_channel"
}

HEADERS = {
    "accept":         "*/*",
    "apiKey":         API_KEY,
    "InstallationId": INSTALLATION_ID,
    "Content-Type":   "application/json",
}
BATCH_SIZE = 10_000 # Records per API call

# --- Helpers ---
def isoformat_z(ts_str: str) -> str | None:
    """Converts datetime string (potentially with or without tz) to 'YYYY-MM-DDTHH:MM:SSZ'."""
    if not ts_str: return None
    try:
        # Try parsing with timezone first (from '%Y-%m-%d %H:%M:%S%z')
        try:
            # Pandas to_datetime is good at inferring various formats including ISO8601 with offset
            dt_aware = pd.to_datetime(ts_str, errors='raise')
            if dt_aware.tzinfo is None: # If it parsed but resulted in naive (e.g. "YYYY-MM-DD HH:MM:SS")
                dt_aware = dt_aware.tz_localize(dt_timezone.utc) # Assume naive is UTC
            else: # Already tz-aware
                dt_aware = dt_aware.tz_convert(dt_timezone.utc) # Ensure it's UTC
            return dt_aware.isoformat(timespec='seconds').replace("+00:00", "Z")

        except ValueError: # Fallback if pd.to_datetime fails for some reason or it was a simpler format
            logger.debug(f"pd.to_datetime failed for '{ts_str}', trying strptime...")
            # Assume naive "YYYY-MM-DD HH:MM:SS" represents UTC numbers if direct fromisoformat failed
            dt_naive = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            dt_utc_aware = dt_naive.replace(tzinfo=dt_timezone.utc)
            return dt_utc_aware.isoformat(timespec='seconds').replace("+00:00", "Z")

    except ValueError as e:
        logger.warning(f"Could not parse date string '{ts_str}' to ISO Z format: {e}.")
        return None

def build_payload_for_sensor(csv_path: Path, sensor_wm_channel_ids: dict) -> list:
    # sensor_wm_channel_ids is expected to be like:
    # {'raw_radar_channel': 123, 'adjusted_rainfall_channel': 456, 'flag_channel': 789}
    # where keys match those in CSV_COLUMN_TO_WM_TYPE_KEY_MAP's values.

    logger.info(f"Reading CSV file: {csv_path.name}")
    payload = []
    
    # Map from CSV column name to the actual Watermanager Channel ID for this sensor
    active_csv_col_to_wm_id_map = {}
    for csv_col, type_key_in_json in CSV_COLUMN_TO_WM_TYPE_KEY_MAP.items():
        print(f"Processing CSV column '{csv_col}' with type key '{type_key_in_json}'")
        wm_channel_id = sensor_wm_channel_ids.get(type_key_in_json)
        if wm_channel_id is not None:
            active_csv_col_to_wm_id_map[csv_col] = wm_channel_id
        else:
            logger.debug(f"Data type key '{type_key_in_json}' (for CSV col '{csv_col}') not found or None in sensor_wm_channel_ids for {csv_path.name}. This column won't be pushed.")

    if not active_csv_col_to_wm_id_map:
        logger.error(f"No valid Watermanager channels could be mapped for {csv_path.name} using mapping: {sensor_wm_channel_ids}. Skipping payload build.")
        return []
    logger.debug(f"Active CSV Col to WM Channel ID map for {csv_path.name}: {active_csv_col_to_wm_id_map}")

    try:
        print(f"Opening CSV file: {csv_path.name}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logger.warning(f"CSV file {csv_path} is empty or has no header. Skipping.")
                return []
            logger.debug(f"CSV columns for {csv_path.name}: {reader.fieldnames}")
            
            # Check if the CSV contains the columns we intend to map AND the 'time' column
            required_cols_in_csv = ["time"] + list(active_csv_col_to_wm_id_map.keys())
            missing_cols_in_csv = [col for col in required_cols_in_csv if col not in reader.fieldnames]
            if missing_cols_in_csv:
                logger.error(f"CSV {csv_path.name} is missing required columns: {missing_cols_in_csv}. Skipping file.")
                return []
            print(f"CSV {csv_path.name} contains all required columns: {required_cols_in_csv}")
            for i, row in enumerate(reader):
                
                log_date_str = row.get("time")
                if not log_date_str:
                    logger.warning(f"Row {i+1} in {csv_path.name} missing 'time' value. Skipping data for this row.")
                    continue
                
                log_date_iso = isoformat_z(log_date_str)
                if not log_date_iso:
                    logger.warning(f"Could not parse 'time' {log_date_str} in row {i+1} of {csv_path.name}. Skipping data for this timestamp.")
                    continue

                for csv_col_name, wm_channel_id in active_csv_col_to_wm_id_map.items():
                    raw_value = row.get(csv_col_name)
                    if raw_value is None or str(raw_value).strip() == "": # Check for empty string explicitly
                        continue 
                    
                    try:
                        if csv_col_name == "Flagged":
                            processed_value = 1 if str(raw_value).strip().lower() in ["true", "1", "1.0", "yes"] else 0
                        else:
                            processed_value = float(raw_value)
                        
                        payload.append({
                            "channel":  wm_channel_id,
                            "logDate":  log_date_iso,
                            "value":    processed_value,
                            "valueOk":  True, 
                            "manual":   True 
                        })
                    except ValueError:
                        logger.warning(f"Could not convert value '{raw_value}' for CSV column '{csv_col_name}' to expected type in row {i+1} ({log_date_iso}). Skipping this data point.")
                        continue
    except FileNotFoundError:
        logger.error(f"CSV file not found during payload build: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading or processing CSV {csv_path}: {e}")
        logger.error(traceback.format_exc())
        return []
        
    logger.info(f"Built payload with {len(payload)} records from {csv_path.name}.")
    return payload

def post_data_batch(batch_to_send, batch_num_info=""):
    label = f"[Batch {batch_num_info}] " if batch_num_info else "[Batch] "
    if not batch_to_send:
        logger.info(f"{label}No records in this batch to send.")
        return
    logger.info(f"{label}Sending {len(batch_to_send)} records to Watermanager...")
    try:
        resp = requests.post(API_URL, json=batch_to_send, headers=HEADERS, timeout=120, verify=False) # Increased timeout
        
        logger.info(f"{label}Status code: {resp.status_code}")
        response_text_to_log = resp.text[:500] + "..." if len(resp.text) > 500 else resp.text
        if resp.text:
            try: logger.debug(f"{label}Response JSON: {resp.json()}")
            except ValueError: logger.debug(f"{label}Non-JSON response: {response_text_to_log}")
        else: logger.info(f"{label}Empty response body from Watermanager.")
        
        if resp.ok: logger.info(f"{label}✅ Success")
        else: logger.error(f"{label}❌ Error {resp.status_code} - Response: {response_text_to_log}")
        # resp.raise_for_status() # Consider if you want to halt on error or log and continue
    except requests.exceptions.Timeout:
        logger.error(f"{label}Request timed out.")
    except requests.exceptions.RequestException as e_req:
        logger.error(f"{label}Request failed: {e_req}")
    except Exception as e_post:
        logger.error(f"{label}An unexpected error occurred during post: {e_post}")

def extract_utm_str_from_filename(filename: str) -> str | None:
    """Extracts (X,Y) string like '(XXXXXX,YYYYYYY)' from various detailed output CSV filename patterns."""
    # Pattern for daily outputs like XXXXXX_YYYYYY_adj_YYYYMMDD_HHMM.csv
    match_daily = re.search(r"(\d+)_(\d+)_adj_\d{8}_\d{4}", filename)
    if match_daily:
        return f"({match_daily.group(1)},{match_daily.group(2)})"
    
    # Pattern for historical outputs like XXXXXX_YYYYYY_data_adjustments.csv
    match_historical = re.search(r"(\d+)_(\d+)_data_adjustments", filename)
    if match_historical:
        return f"({match_historical.group(1)},{match_historical.group(2)})"
        
    logger.debug(f"Could not extract UTM from filename using known patterns: {filename}")
    return None

# --- Main ---
if __name__ == "__main__":
    logger.info("===== Starting Watermanager Data Upload Script =====")

    if not SENSOR_TO_WM_CHANNEL_MAPPING_FILE.exists():
        logger.critical(f"CRITICAL: WM channel mapping file not found: {SENSOR_TO_WM_CHANNEL_MAPPING_FILE}.")
        exit(1)
    try:
        with open(SENSOR_TO_WM_CHANNEL_MAPPING_FILE, 'r') as f:
            master_sensor_to_wm_map = json.load(f) 
        logger.info(f"Successfully loaded WM channel mapping from {SENSOR_TO_WM_CHANNEL_MAPPING_FILE}")
    except Exception as e:
        logger.critical(f"CRITICAL: Error loading WM channel mapping JSON: {e}"); exit(1)

    if not DETAILED_OUTPUT_DIR.exists() or not DETAILED_OUTPUT_DIR.is_dir():
        logger.critical(f"CRITICAL: Detailed sensor output directory not found: {DETAILED_OUTPUT_DIR}")
        exit(1)

    # Find all CSV files: those ending with _data_adjustments.csv (historical)
    # OR those ending with _adj_YYYYMMDD_HHMM.csv (daily)
    # This uses rglob to search in DETAILED_OUTPUT_DIR and its subdirectories.
    historical_pattern = "*_data_adjustments.csv"
    daily_pattern = "*_adj_????????_????.csv" # Matches YYYYMMDD_HHMM

    all_detailed_csvs = list(DETAILED_OUTPUT_DIR.rglob(historical_pattern))
    # Add daily files, ensuring no duplicates if a file somehow matches both
    daily_files = {f for f in DETAILED_OUTPUT_DIR.rglob(daily_pattern)}
    all_detailed_csvs.extend(list(daily_files - set(all_detailed_csvs)))


    if not all_detailed_csvs:
        logger.info(f"No detailed sensor output CSV files found in {DETAILED_OUTPUT_DIR} or subdirectories matching patterns.")
        exit(0)
    
    logger.info(f"Found {len(all_detailed_csvs)} detailed sensor CSV files to process.")

    for csv_file_path in all_detailed_csvs:
        logger.info(f"\n--- Processing CSV: {csv_file_path.name} ---")
        
        utm_coord_string_from_filename = extract_utm_str_from_filename(csv_file_path.name)
        logger.debug(f"Extracted UTM string from filename '{csv_file_path.name}': '{utm_coord_string_from_filename}'")
        
        if not utm_coord_string_from_filename:
            logger.warning(f"Could not extract UTM string from filename: {csv_file_path.name}. Skipping.")
            continue

        sensor_specific_wm_channels_dict = None # Stores {'raw_radar_channel': ID, ...}
        found_mapping_for_file = False
        for sensor_identifier_in_json, mapping_details_json in master_sensor_to_wm_map.items():
            # The keys in your JSON are sensor names like "Regnmåler - 8: Kærvej 2"
            # The value for "utm_coord_str" inside mapping_details_json is what we match against filename
            if mapping_details_json.get("utm_coord_str") == utm_coord_string_from_filename:
                sensor_specific_wm_channels_dict = {
                    # These keys MUST match the keys in CSV_COLUMN_TO_WM_TYPE_KEY_MAP's values
                    "raw_radar_channel": mapping_details_json.get("raw_radar_channel"),
                    "adjusted_rainfall_channel": mapping_details_json.get("adjusted_rainfall_channel"),
                    "flag_channel": mapping_details_json.get("flag_channel")
                }
                # Filter out None values (if a type of channel is not defined for this sensor in JSON)
                sensor_specific_wm_channels_dict = {k: v for k, v in sensor_specific_wm_channels_dict.items() if v is not None}
                logger.info(f"Found WM channel mapping for {utm_coord_string_from_filename} (via JSON key '{sensor_identifier_in_json}'): {sensor_specific_wm_channels_dict}")
                found_mapping_for_file = True
                break 
        
        if not found_mapping_for_file:
            logger.warning(f"No WM channel mapping found in JSON for UTM coordinate {utm_coord_string_from_filename} (from file {csv_file_path.name}). Skipping.")
            continue
        if not sensor_specific_wm_channels_dict or not any(sensor_specific_wm_channels_dict.values()):
            logger.warning(f"WM Channel mapping for {utm_coord_string_from_filename} is empty or all channels are None. Skipping.")
            continue

        records_for_sensor = build_payload_for_sensor(csv_file_path, sensor_specific_wm_channels_dict)

        if not records_for_sensor:
            logger.info(f"No records to send for {csv_file_path.name}.")
            continue

        total_records_for_sensor = len(records_for_sensor)
        for i in range(0, total_records_for_sensor, BATCH_SIZE):
            batch = records_for_sensor[i : i + BATCH_SIZE]
            batch_num_str = f"{ (i // BATCH_SIZE) + 1 } of { (total_records_for_sensor // BATCH_SIZE) + 1 } for {csv_file_path.name}"
            print("Processing batch:", batch_num_str)
            post_data_batch(batch, batch_num_str)
            if i + BATCH_SIZE < total_records_for_sensor:
                logger.info("Pausing briefly between batches for this file...")
                time.sleep(0.5) # Shorter pause between batches of the same file

        logger.info(f"--- Finished processing CSV: {csv_file_path.name} ---")
        if len(all_detailed_csvs) > 1: # If processing multiple files, add a slightly longer pause
            logger.info("Pausing briefly before processing next CSV file...")
            time.sleep(2)


    logger.info("===== Watermanager Data Upload Script Finished =====")
# --- END OF FILE wm.py ---