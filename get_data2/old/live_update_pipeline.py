# --- START OF FILE live_update_pipeline.py ---

import pandas as pd
import time
from datetime import datetime
import sys
from get_gauge_data2 import parse_coordinates as transform_coordinates


# Import functions from your modified scripts
# Ensure these files are in the same directory or Python path
try:
    from get_gauge_data2 import parse_coordinates as parse_gauge_coords, transform_coordinates as transform_gauge_coords
    from get_data2.old.get_radar_data import get_radar_data_incremental
    from get_combined_data import combine_data_for_location
    from save_as_pkl import save_combined_to_pkl, COMBINED_FOLDER as pkl_input_folder, COMBINED_FILENAME_PATTERN as pkl_input_pattern
    import os
    import re # For finding combined files for pickling
except ImportError as e:
    print(f"ERROR: Could not import necessary functions. Make sure all .py files are present.")
    print(e)
    sys.exit(1)

# --- Configuration ---
SENSOR_LIST_FILE = 'SensorOversigt.xlsx'
RUN_INTERVAL_SECONDS = 3600 # Run every hour (3600 seconds)
# Set RUN_INTERVAL_SECONDS = 0 to run only once

# --- Main Pipeline Function ---
def run_update_cycle():
    """Performs one full update cycle for all sensors."""
    print(f"\n--- Starting Update Cycle: {datetime.now()} ---")

    # 1. Read Sensor List
    print(f"Reading sensor list from {SENSOR_LIST_FILE}...")
    try:
        df_sensors = pd.read_excel(SENSOR_LIST_FILE)
    except FileNotFoundError:
        print(f"ERROR: {SENSOR_LIST_FILE} not found. Cannot proceed.")
        return
    except Exception as e:
        print(f"ERROR reading {SENSOR_LIST_FILE}: {e}")
        return

    print(f"Found {len(df_sensors)} sensors in the list.")

    # Prepare list of sensor info with coordinates
    sensors_to_process = []
    for index, row in df_sensors.iterrows():
        lon, lat = parse_gauge_coords(row['wkt_geom']) # Use parser from gauge script
        x, y = transform_gauge_coords(lon, lat)       # Use transformer from gauge script
        if x is not None and y is not None:
            sensors_to_process.append({
                'name': row['Name'],
                'channel': row['Channel'],
                'wkt_geom': row['wkt_geom'],
                'x': x,
                'y': y
            })
        else:
            print(f"Skipping sensor {row['Name']} due to coordinate processing error.")

    if not sensors_to_process:
         print("No valid sensors found after processing coordinates. Exiting cycle.")
         return

    # --- Process Each Sensor ---
    gauge_success = 0
    gauge_fail = 0
    radar_success = 0
    radar_fail = 0
    combine_success = 0
    combine_fail = 0
    # PKL success/fail counted later

    processed_coords = set() # Keep track of coords processed in this cycle

    for sensor in sensors_to_process:
        print(f"\n>>> Processing Sensor: {sensor['name']} (X={sensor['x']}, Y={sensor['y']}) <<<")
        coord_tuple = (sensor['x'], sensor['y'])
        processed_coords.add(coord_tuple)

        # 2. Get Incremental Gauge Data
        print("--- Updating Gauge Data ---")
        gauge_result_path = update_gauge_data_from_last(
            sensor_name=sensor['name'],
            channel_no=sensor['channel'],
            coord_string=sensor['wkt_geom']
            # gauge_folder is taken from constants inside get_gauge_data.py
        )
        if gauge_result_path:
            gauge_success += 1
        else:
            gauge_fail += 1
            print(f"WARNING: Gauge data update failed for {sensor['name']}. Combination might use stale/missing data.")
            # Continue processing radar even if gauge fails? Decide your strategy. Let's continue for now.

        # 3. Get Incremental Radar Data
        print("\n--- Updating Radar Data ---")
        radar_result_path = get_radar_data_incremental(
            x=sensor['x'],
            y=sensor['y'],
            name=sensor['name'],
            channel=sensor['channel']
             # radar_folder is taken from constants inside get_radar_data.py
        )
        if radar_result_path:
            radar_success += 1
        else:
            radar_fail += 1
            print(f"WARNING: Radar data update failed for {sensor['name']}. Combination might use stale/missing data.")
            # Continue processing? Let's continue.

        # 4. Combine Data (only if both might have updated - or always run?)
        # Always run combine for the location to ensure it reflects latest fetched data.
        print("\n--- Combining Data ---")
        if combine_data_for_location(x=sensor['x'], y=sensor['y']):
             combine_success += 1
        else:
             combine_fail +=1
             print(f"WARNING: Combination failed for ({sensor['x']},{sensor['y']}). PKL file will not be updated.")

    # --- Post-Processing After All Sensors ---

    # 5. Save to PKL (for all successfully combined files in this cycle)
    print("\n--- Saving Updated Data to PKL ---")
    pkl_success = 0
    pkl_fail = 0
    processed_combined_files = []
    if os.path.exists(pkl_input_folder):
        for filename in os.listdir(pkl_input_folder):
            match = re.search(pkl_input_pattern, filename)
            if match:
                try:
                    x_pkl, y_pkl = int(match.group(1)), int(match.group(2))
                    # Only process PKL for coordinates handled in *this* cycle
                    if (x_pkl, y_pkl) in processed_coords:
                         filepath = os.path.join(pkl_input_folder, filename)
                         if save_combined_to_pkl(filepath):
                             pkl_success += 1
                         else:
                             pkl_fail += 1
                         processed_combined_files.append(filename)
                except ValueError:
                     print(f"Warning: Could not parse coordinates from combined filename {filename}")
                     continue

    print("\n--- Update Cycle Summary ---")
    print(f"Gauge Data: {gauge_success} updated, {gauge_fail} failed.")
    print(f"Radar Data: {radar_success} updated, {radar_fail} failed.")
    print(f"Combination: {combine_success} updated, {combine_fail} failed.")
    print(f"PKL Saving: {pkl_success} updated, {pkl_fail} failed (for coords processed in this cycle).")
    print(f"--- Update Cycle Finished: {datetime.now()} ---")


# --- Main Execution Loop ---
if __name__ == "__main__":
    if RUN_INTERVAL_SECONDS > 0:
        print(f"Starting live update pipeline. Running every {RUN_INTERVAL_SECONDS} seconds.")
        print("Press Ctrl+C to stop.")
        while True:
            try:
                run_update_cycle()
                print(f"\nSleeping for {RUN_INTERVAL_SECONDS} seconds...")
                time.sleep(RUN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                print("\nCtrl+C detected. Exiting pipeline.")
                break
            except Exception as e:
                print(f"\nFATAL ERROR in main loop: {e}")
                print("Pipeline will attempt to restart after interval.")
                # Optional: Add more robust error handling/logging here
                try:
                    time.sleep(RUN_INTERVAL_SECONDS) # Wait before retrying
                except KeyboardInterrupt:
                     print("\nCtrl+C detected during error sleep. Exiting pipeline.")
                     break
    else:
        print("Running a single update cycle.")
        run_update_cycle()
        print("\nSingle run complete.")

# --- END OF FILE live_update_pipeline.py ---