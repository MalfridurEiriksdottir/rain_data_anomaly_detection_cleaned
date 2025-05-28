# --- START OF FILE get_radar_data.py (Modified for Robust Append) ---
import pandas as pd
import os
import re
import pyproj
import subprocess
from datetime import datetime, timedelta
import time
import warnings
import configparser # To read config easily

import logging

logger = logging.getLogger(__name__)

# --- Define Constants ---
RADAR_FOLDER = "radar_data"
CONFIG_PATH = "get_data2/config.txt"
# CONFIG_PATH = "get_data2/" + CONFIG_PATH
POWERSHELL_SCRIPT_PATH = "get_data2/TimeSeries_API_DownloadKopi2.ps1" # Or actual name
DEFAULT_START_DATE_RADAR = datetime(2024, 6, 21, 8, 10, 0) # Match initial config or reasonable default
DATE_FORMAT_CONFIG = "%Y-%m-%d %H:%M:%S"
# --- Assuming these formats are correct for radar data ---
# DATE_FORMAT_CSV = "%Y-%m-%d %H:%M:%S.%f" # If radar CSV has fractional seconds
DATE_FORMAT_CSV = "%Y-%m-%d %H:%M:%S" # If radar CSV has only seconds
# ---

# Suppress specific warnings if needed
# warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions (get_config_value, update_config, convert_coordinates, parse_coordinates) ---
# Assume these are correct and unchanged
def get_config_value(key, config_path=CONFIG_PATH):
    # config_path = "get_data2/" + config_path
    # print(f"Reading config value for '{key}' from {config_path}...")
    # read the txt file config.txt

    try:
        with open(config_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    if k == key: return v
    except Exception as e: print(f"Warning: Could not read key '{key}': {e}")
    return None

def get_last_timestamp_from_radar_csv(filepath):
    """Reads the last timestamp from the 'time' column of a radar CSV."""
    if not os.path.exists(filepath):
        # print(f"Info: Radar file {filepath} does not exist.")
        logger.info(f"Radar file {filepath} does not exist.")
        return None
    try:
        # Increased robustness: Read only necessary column, handle potential parse errors better
        df = pd.read_csv(filepath, usecols=['time'])
        # Attempt parsing, coercing errors, then drop NaT before finding max
        df['time_dt'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['time_dt'], inplace=True)
        if not df.empty:
            last_time = df['time_dt'].iloc[-1].tz_localize(None) # Ensure naive
            return last_time
    except pd.errors.EmptyDataError:
        # print(f"Info: Radar file {filepath} is empty.")
        logger.info(f"Radar file {filepath} is empty.")
        return None
    except KeyError:
        print(f"Warning: 'time' column not found in {filepath}.")
        return None
    except Exception as e:
        print(f"Error reading last timestamp from {filepath}: {e}")
    return None

def update_config(config_updates, config_path=CONFIG_PATH):
    try:
        lines = []
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                lines = file.readlines()
        else:
            print(f"Warning: Config file {config_path} not found, will create.")

        updated = {key: False for key in config_updates}
        new_lines = []

        for line in lines:
            written = False
            for key, value in config_updates.items():
                if line.startswith(f"{key}="):
                    new_lines.append(f"{key}={value}\n")
                    updated[key] = True
                    written = True
                    break
            if not written:
                new_lines.append(line)

        # Add keys if they were missing entirely
        for key, value in config_updates.items():
            if not updated[key]:
                 print(f"Warning: Key '{key}' not found in config, adding it.")
                 new_lines.append(f"{key}={value}\n")

        with open(config_path, "w") as file:
            file.writelines(new_lines)
        return True

    except Exception as e:
        print(f"ERROR updating config file {config_path}: {e}")
        return False


def convert_coordinates(lon, lat):
    if lon is None or lat is None: return None, None
    try:
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return int(x), int(y)
    except Exception as e: print(f"Error transforming coordinates ({lon}, {lat}): {e}"); return None, None

def parse_coordinates(coord_string):
    try:
        coord_string = str(coord_string).replace("Point (", "").replace(")", "")
        lon, lat = map(float, coord_string.split())
        return lon, lat
    except Exception as e: print(f"Error parsing coordinates '{coord_string}': {e}"); return None, None

# --- FIND POWERSHELL OUTPUT ---
# *** IMPORTANT: Verify this pattern matches ACTUAL PowerShell output filenames ***
def find_powershell_output_file(output_dir, radar_id, x, y):
    """Tries to find the CSV file generated by PowerShell based on naming patterns."""
    # print(f"Searching for PowerShell output in '{output_dir}' for pattern: TimeSeries_{radar_id}_X{x}_Y{y}_*.csv")
    time.sleep(2) # Give filesystem time to settle
    possible_files = []
    try:
        if not os.path.isdir(output_dir):
             print(f" PowerShell output directory '{output_dir}' not found or is not a directory.")
             return None

        for filename in os.listdir(output_dir):
            # --- ADJUST THIS REGEX IF NEEDED ---
            pattern = rf"TimeSeries_{radar_id}_X{x}_Y{y}_.*\.csv"
            # --- --- --- --- --- --- --- --- ---
            match = re.match(pattern, filename, re.IGNORECASE)
            if match:
                 filepath = os.path.join(output_dir, filename)
                 # Check if it's actually a file and was modified recently (e.g., last 10 mins)
                 try:
                     file_mod_time = os.path.getmtime(filepath)
                     if (time.time() - file_mod_time) < 600: # Modified in last 10 minutes
                         possible_files.append(filepath)
                     else:
                         print(f"  - Found matching file, but old: {filename} (modified {datetime.fromtimestamp(file_mod_time)})")
                 except Exception as stat_e:
                      print(f"  - Warning: Could not get modification time for {filename}: {stat_e}")

        if not possible_files:
            # print(f" No recently modified PowerShell output file found matching the pattern.")
            logger.info(f"No recently modified PowerShell output file found matching the pattern.")
            return None

        # Sort by modification time, newest first
        possible_files.sort(key=os.path.getmtime, reverse=True)
        # print(f"Found {len(possible_files)} recent matching file(s). Using newest: {os.path.basename(possible_files[0])}")
        return possible_files[0]

    except Exception as e:
        print(f"ERROR searching for PowerShell output file: {e}")
        return None

# --- Main Radar Data Update Function ---

def get_radar_data_incremental(x, y, name, channel):
    """
    Checks last timestamp, runs PowerShell to fetch new data to a temp file,
    filters temp file data, appends only new data to persistent file.
    Returns the path to the updated persistent radar CSV file or None if failed.
    """
    # print(f"\n--- Processing Radar: {name} (X={x}, Y={y}) ---")
    os.makedirs(RADAR_FOLDER, exist_ok=True)

    # Read necessary static config values
    radar_id = get_config_value("radar_id")
    powershell_output_dir = get_config_value("output_dir") # Where PowerShell saves its file
    # bias = get_config_value("bias", CONFIG_PATH) # Optional: include if needed by PS

    if not radar_id or not powershell_output_dir:
        print(" Missing 'radar_id' or 'output_dir' in config.txt")
        return None

    # Define the target *persistent* file path
    radar_csv_filename = f"VevaRadar_X{x}_Y{y}_.csv" # Check if this is correct
    radar_csv_filepath = os.path.join(RADAR_FOLDER, radar_csv_filename)
    radar_csv_filepath = os.path.normpath(radar_csv_filepath)

    # Determine start date for API call
    last_timestamp = get_last_timestamp_from_radar_csv(radar_csv_filepath)
    if last_timestamp:
        start_date = last_timestamp + timedelta(seconds=1)
        # print(f"Last record found: {last_timestamp}. Requesting data from: {start_date}")
        # start_date = datetime(2025, 4, 21, 8, 10, 0)
        # print(f"Last record found: {last_timestamp}. Requesting data from: {start_date}")
        logger.info(f"Last record found: {last_timestamp}. Requesting data from: {start_date}")
    else:
        start_date_str_config = get_config_value("time_start") # Use start time from config for first run
        try:
             start_date = datetime.strptime(start_date_str_config, DATE_FORMAT_CONFIG)
             print(f"Using 'time_start' from config: {start_date}")
        except:
             print(f"Warning: Could not parse 'time_start' from config. Using fallback: {DEFAULT_START_DATE_RADAR}")
             start_date = DEFAULT_START_DATE_RADAR
        print(f"No existing data found. Requesting data from default/config start: {start_date}")

    end_date = datetime.now()
    # print('END_DATE:', end_date)
    # end_date = datetime(2025, 3, 15, 8, 10, 0)

    if start_date >= end_date:
        print(f"Start date ({start_date}) not before end date ({end_date}). No new data fetch needed.")
        return radar_csv_filepath # Existing path is considered success (no new data)

    # --- Prepare and Run PowerShell ---
    config_updates = {
        "time_start": start_date.strftime(DATE_FORMAT_CONFIG),
        "time_end": end_date.strftime(DATE_FORMAT_CONFIG),
        "x": x,
        "y": y,
    }

    # print(f"Updating config.txt for PowerShell: {config_updates}")
    if not update_config(config_updates): return None # Failed to update config

    # print(f"Running PowerShell script: {POWERSHELL_SCRIPT_PATH}...")
    ps_output_file = None # Ensure ps_output_file is defined for cleanup
    try:
        # Run PowerShell, capture output
        # print(f"Running PowerShell script: {POWERSHELL_SCRIPT_PATH}...")
        # print('Getting data for the following coordinates in the period:')
        # print(f"X: {x}, Y: {y}")
        # print(f"Start: {start_date.strftime(DATE_FORMAT_CONFIG)}")
        # print(f"End: {end_date.strftime(DATE_FORMAT_CONFIG)}")
        ps_result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", POWERSHELL_SCRIPT_PATH],
            check=False, # Don't raise error immediately, check returncode manually
            capture_output=True,
            text=True,
            encoding='utf-8' # Specify encoding if needed
        )
        # print(f"PowerShell stdout:\n{ps_result.stdout or '(No stdout)'}") # Print stdout
        logger.info(f"PowerShell stdout:\n{ps_result.stdout or '(No stdout)'}") # Print stdout
        
        if ps_result.stderr: print(f"PowerShell stderr:\n{ps_result.stderr}") # Print stderr
        if ps_result.returncode != 0:
             print(f" PowerShell script failed! (Return Code: {ps_result.returncode}). Check stderr/stdout above.")
             # Attempt to find output file even on failure? Maybe PS created partial file.
             # Let's try finding it anyway for potential cleanup.
             ps_output_file = find_powershell_output_file(powershell_output_dir, radar_id, x, y)
             return None # Indicate failure

        # print("PowerShell script finished successfully (Return Code: 0).")
        logger.info(f"PowerShell script finished successfully (Return Code: {ps_result.returncode}).")
        # --- Find the output file created by PowerShell ---
        ps_output_file = find_powershell_output_file(powershell_output_dir, radar_id, x, y)
        if not ps_output_file:
            # print(" Python could not locate the output file created by PowerShell.")
            logger.info("Python could not locate the output file created by PowerShell.")
            # Maybe PS didn't create one despite success code? Or filename pattern is wrong.
            return None # Indicate failure

        # print(f"Found PowerShell output file: {ps_output_file}")
        logger.info(f"Found PowerShell output file: {ps_output_file}")

        # --- Read and Process PowerShell Output File ---
        try:
            # Read the *temporary* data
            # Specify dtype to avoid mixed type warnings if columns are sometimes empty
            new_data_df = pd.read_csv(ps_output_file, dtype={'time': str, 'value': object, 'quality': object})
        except pd.errors.EmptyDataError:
            print(f"Info: PowerShell output file {ps_output_file} is empty. No new data to process.")
            # Delete the empty temp file and return success
            try: os.remove(ps_output_file); print(f"Deleted empty temporary file: {ps_output_file}")
            except Exception as e_del: print(f"Warning: could not delete empty temp file {ps_output_file}: {e_del}")
            return radar_csv_filepath # Success, no new data
        except Exception as read_e:
            print(f"ERROR reading PowerShell output file {ps_output_file}: {read_e}")
            # Try to delete corrupt temp file before failing
            try: os.remove(ps_output_file)
            except Exception as e_del: print(f"Warning: could not delete corrupt temp file {ps_output_file}: {e_del}")
            return None # Indicate failure

        # Clean headers (adjust if PS output has different headers)
        new_data_df.columns = [c.strip().strip('"') for c in new_data_df.columns]
        required_cols = ['time', 'value']
        if not all(col in new_data_df.columns for col in required_cols):
             print(f" PowerShell output file {ps_output_file} missing required columns ('time', 'value').")
             try: os.remove(ps_output_file)
             except Exception as e_del: print(f"Warning: could not delete bad temp file {ps_output_file}: {e_del}")
             return None # Indicate failure

        # Parse time
        new_data_df['time_dt'] = pd.to_datetime(new_data_df['time'], errors='coerce')
        original_rows = len(new_data_df)
        new_data_df.dropna(subset=['time_dt'], inplace=True)
        parsed_rows = len(new_data_df)
        if original_rows > parsed_rows:
            print(f"Warning: Dropped {original_rows - parsed_rows} rows due to invalid time format in PowerShell output.")

        if new_data_df.empty:
             print(f"Info: No valid time data found in {ps_output_file} after parsing.")
             try: os.remove(ps_output_file); print(f"Deleted temporary file with invalid time: {ps_output_file}")
             except Exception as e_del: print(f"Warning: could not delete invalid time temp file {ps_output_file}: {e_del}")
             return radar_csv_filepath # Success, no usable new data


        new_data_df['time_dt'] = new_data_df['time_dt'].dt.tz_localize(None) # Ensure naive

        # --- Filter: Keep only data strictly newer than last_timestamp ---
        rows_before_filter = len(new_data_df)
        if last_timestamp:
            # print(f"Filtering PowerShell data newer than {last_timestamp}...")
            logger.info(f"Filtering PowerShell data newer than {last_timestamp}...")
            new_data_df = new_data_df[new_data_df['time_dt'] > last_timestamp].copy() # Use .copy() to avoid SettingWithCopyWarning
            rows_after_filter = len(new_data_df)
            # print(f"Filtering complete: Kept {rows_after_filter} out of {rows_before_filter} rows.")
            logger.info(f"Filtering complete: Kept {rows_after_filter} out of {rows_before_filter} rows.")
        else:
            # print("No previous data found, keeping all fetched data.")
            logger.info("No previous data found, keeping all fetched data.")
            rows_after_filter = rows_before_filter # All data is new

        # --- Append Filtered Data ---
        if new_data_df.empty:
            print(f"No *new* radar data points found in PowerShell output after filtering.")
        else:
            # print(f"Appending {len(new_data_df)} new radar records to {radar_csv_filepath}")
            logger.info(f"Appending {len(new_data_df)} new radar records to {radar_csv_filepath}")
            # Select columns to append (adjust if needed)
            cols_to_append = ['time', 'value']
            if 'quality' in new_data_df.columns: cols_to_append.append('quality')
            # Check if all desired columns exist before selecting
            cols_exist = [col for col in cols_to_append if col in new_data_df.columns]
            if len(cols_exist) != len(cols_to_append):
                 print(f"Warning: Missing some expected columns {set(cols_to_append) - set(cols_exist)} in temp file, appending available: {cols_exist}")
            new_data_to_append = new_data_df[cols_exist]


            # Determine if header is needed for the *persistent* file
            file_exists_and_has_data = (last_timestamp is not None) or (os.path.exists(radar_csv_filepath) and os.path.getsize(radar_csv_filepath) > 50)

            try:
                new_data_to_append.to_csv(
                    radar_csv_filepath,
                    mode='a',
                    header=not file_exists_and_has_data, # Add header only if persistent file is new/empty
                    index=False
                )
                # print(f"Append successful.")
                logger.info(f"Appended {len(new_data_to_append)} new records to {radar_csv_filepath}.")
            except Exception as append_e:
                #  print(f"ERROR appending data to {radar_csv_filepath}: {append_e}")
                 logger.error(f"err: appending data to {radar_csv_filepath}: {append_e}")
                 # Don't delete temp file if append failed, might need manual recovery
                 return None # Indicate failure

        # --- Clean up temporary file ---
        # print(f"Deleting temporary PowerShell output file: {ps_output_file}")
        logger.info(f"Deleting temporary PowerShell output file: {ps_output_file}")
        try:
            os.remove(ps_output_file)
        except Exception as e_del:
            print(f"Warning: Failed to delete temporary file {ps_output_file}: {e_del}")

        return radar_csv_filepath # Return path to updated persistent file

    except FileNotFoundError: # For powershell executable itself
        print(f" PowerShell executable or script '{POWERSHELL_SCRIPT_PATH}' not found.")
        return None
    except Exception as e: # General catch-all for this block
        print(f"ERROR during PowerShell execution or processing: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up temp file if it exists and path is known
        if ps_output_file and os.path.exists(ps_output_file):
            try: os.remove(ps_output_file); print(f"Cleaned up temporary file {ps_output_file} after error.")
            except Exception as e_del: print(f"Warning: could not delete temp file {ps_output_file} after error: {e_del}")
        return None

# --- Main execution part (if run standalone) ---
if __name__ == "__main__":
    # This remains the same, reads SensorOversigt.xlsx and calls get_radar_data_incremental for each
    print("Reading sensor list from SensorOversigt.xlsx...")
    try:
        df_sensors = pd.read_excel('SensorOversigt.xlsx')
    except FileNotFoundError:
        print(" SensorOversigt.xlsx not found. Cannot proceed.")
        exit(1)
    except Exception as e:
        print(f"ERROR reading SensorOversigt.xlsx: {e}")
        exit(1)

    print(f"Found {len(df_sensors)} sensors in the list.")
    successful_files = []
    failed_sensors = []

    # Prepare sensor info (coordinate conversion etc.)
    sensor_coords = []
    for index, row in df_sensors.iterrows():
        sensor_name = row.get('Name', f'Unnamed Sensor Index {index}')
        coordinates = row.get('wkt_geom')
        channel_no_raw = row.get('Channel')

        if pd.isna(coordinates): continue # Skip if no coords
        try: channel_no = int(channel_no_raw)
        except (ValueError, TypeError): continue # Skip if invalid channel

        lon, lat = parse_coordinates(coordinates)
        x, y = convert_coordinates(lon, lat)
        if x is not None and y is not None:
            sensor_coords.append({
                'name': sensor_name, 'channel': channel_no,
                'lon': lon, 'lat': lat, 'x': x, 'y': y
            })
        else:
            print(f"Skipping sensor {sensor_name} due to coordinate error.")
            failed_sensors.append(f"{sensor_name} (Coord Error)")

    # Process each sensor
    for sensor_info in sensor_coords:
        updated_radar_path = get_radar_data_incremental(
            sensor_info['x'], sensor_info['y'], sensor_info['name'], sensor_info['channel']
        )
        if updated_radar_path:
            successful_files.append(updated_radar_path)
        else:
            failed_sensors.append(f"{sensor_info['name']} (Radar Fetch/Process Error)")

    print("\n--- Radar Data Update Summary ---")
    print(f"Successfully processed/updated {len(successful_files)} radar files.")
    if failed_sensors:
        print(f"Failed to process data for {len(failed_sensors)} sensors/stages: {', '.join(failed_sensors)}")
    print("Radar data update process finished.")

# --- END OF FILE get_radar_data.py ---