# --- START OF FILE live_update_pipeline.py (Modified) ---

import pandas as pd
import time
from datetime import datetime
import sys
import subprocess 
import os
import re
import logging
# LOG_FILENAME = "pipeline.log" # Define log file name
# LOG_FOLDER = "." # Or specify a subfolder like "logs"
# LOG_FILEPATH = os.path.join(LOG_FOLDER, LOG_FILENAME)
# logging.basicConfig(
#     level=logging.INFO, # Set default minimum level to log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
#     format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S', # Define date format
#     # --- ONLY specify filename, NO handlers list ---
#     filename=LOG_FILEPATH,
#     filemode='a' # 'a' for append (default), 'w' for overwrite each run
# )

# #logger = logging.getLogger(__name__)

# --- Remove imports from the old gauge script ---
# from get_gauge_data import update_gauge_data_from_last, parse_coordinates as parse_gauge_coords, transform_coordinates as transform_gauge_coords

# --- Keep imports for other pipeline stages ---
try:
    # We still need coordinate parsing/transformation for the RADAR part,
    # assuming get_radar_data.py still needs them passed in.
    # Let's import them from get_radar_data.py or define them here if needed.
    # For simplicity, let's assume get_radar_data handles its own coordinate logic internally or reads config.
    # If get_radar_data_incremental function still *needs* x,y passed from the pipeline,
    # we'll need parse_coordinates and transform_coordinates from *somewhere*.
    # Let's re-add them here for now, assuming they might be needed for radar input prep.
    from pyproj import Transformer # Need pyproj for coordinate helper functions

    # Import functions for other stages
    from get_radar_data2 import get_radar_data_incremental # Keep this
    from get_combined_data import combine_data_for_location # Keep this
    from save_as_pkl import save_combined_to_pkl, COMBINED_FOLDER as pkl_input_folder, COMBINED_FILENAME_PATTERN as pkl_input_pattern # Keep this

except ImportError as e:
    print(f"ERROR: Could not import necessary functions. Make sure all .py files are present.")
    print(f"Import Error: {e}")
    # Try adding specific paths if modules aren't found
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # sys.path.insert(0, script_dir)
    # print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Helper functions for coordinates (needed if preparing input for radar script) ---
# (Copied from previous gauge script - ensure consistency if get_radar_data uses different ones)
def parse_coordinates_pipeline(coord_string):
    """Extracts lon, lat from 'Point (lon lat)' string."""
    try:
        coord_string = str(coord_string).replace("Point (", "").replace(")", "")
        lon, lat = map(float, coord_string.split())
        return lon, lat
    except Exception as e:
        print(f"Pipeline Error parsing coordinates '{coord_string}': {e}")
        return None, None

def transform_coordinates_pipeline(lon, lat):
    """Transforms from WGS84 (EPSG:4326) to ETRS89 / UTM zone 32N (EPSG:25832)."""
    if lon is None or lat is None: return None, None
    try:
        # Ensure consistent CRS definition
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return int(x), int(y)
    except Exception as e:
        print(f"Pipeline Error transforming coordinates ({lon}, {lat}): {e}")
        return None, None


# --- Configuration ---
SENSOR_LIST_FILE = 'SensorOversigt.xlsx'
GAUGE_SCRIPT_NAME = "get_data2/gg.py" # <--- Use the new script name
RADAR_SCRIPT_NAME = "get_radar_data2.py"
COMBINE_SCRIPT_NAME = "get_combined_data.py" # Optional: run as subprocess too
PKL_SCRIPT_NAME = "save_as_pkl.py"          # Optional: run as subprocess too
GAUGE_ADJUST_SCRIPT_NAME = "adjust_gauge_timezone.py"

GAUGE_REPROCESS_SCRIPT_NAME = "adjust_timestamps.py"

RUN_INTERVAL_SECONDS = 60 # Run every hour (3600 seconds). 604800 seconds in a week, 86400 seconds in a day.
RUN_INTERVAL_SECONDS = 86400
RUN_INTERVAL_SECONDS = 0
# Set RUN_INTERVAL_SECONDS = 0 to run only once

# --- Main Pipeline Function ---
def run_update_cycle():
    """Performs one full update cycle for all sensors."""
    print(f"\n--- Starting Update Cycle: {datetime.now()} ---")

    # 1. Update Gauge Data using the new script
    print(f"\n--- Running Gauge Update Script ({GAUGE_SCRIPT_NAME}) ---")
    print(">>> This script will overwrite existing gauge data files. <<<")
    print(">>> Ensure you have a backup if needed. <<<")
    try:
        # Execute get_gauge_data2.py as a subprocess
        print(f"Running {GAUGE_SCRIPT_NAME}...")
        gauge_result = subprocess.run(["python", GAUGE_SCRIPT_NAME], check=True, capture_output=True, text=True)
        print(f"Gauge script stdout:\n{gauge_result.stdout}")
        # print(f"Gauge script stdout:\n{gauge_result.stdout}")
        # #logger.info(f"Gauge script stdout:\n{gauge_result.stdout}")
        if gauge_result.stderr:
            print(f"Gauge script stderr:\n{gauge_result.stderr}")
        # print("--- Gauge Update Script Finished ---")
        # #logger.info(f"Gauge script finished successfully.")
        gauge_success_flag = True
    except FileNotFoundError:
         print(f"ERROR: Python executable or script '{GAUGE_SCRIPT_NAME}' not found.")
         gauge_success_flag = False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {GAUGE_SCRIPT_NAME} failed with return code {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        print(f"Stdout:\n{e.stdout}")
        gauge_success_flag = False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred running {GAUGE_SCRIPT_NAME}: {e}")
        gauge_success_flag = False

    # Decide whether to continue if gauge update failed (optional)
    # if not gauge_success_flag:
    #     print("Exiting cycle due to gauge update failure.")
    #     return

    # 2. Update Radar Data
    # This part requires reading the sensor list again, as the radar script needs sensor info.
    # Alternatively, modify radar script to read the list itself. Assuming current structure:
    print(f"\n--- Preparing and Running Radar Update Script ({RADAR_SCRIPT_NAME}) ---")
    try:
        df_sensors = pd.read_excel(SENSOR_LIST_FILE)
        print(f"Sensor list read successfully from {SENSOR_LIST_FILE}.")
    except Exception as e:
        print(f"ERROR reading {SENSOR_LIST_FILE} for radar processing: {e}")
        # Decide if pipeline should stop if sensor list fails here
        return

    radar_success = 0
    radar_fail = 0
    processed_coords = set() # Keep track for combine/pkl steps later
    import tqdm as tqdm
    for index, row in tqdm.tqdm(df_sensors.iterrows(), desc="Processing Sensors", total=len(df_sensors)):
        sensor_name = row.get('Name', f'Unnamed Sensor Index {index}')
        channel_no_raw = row.get('Channel') # Radar script needs channel too
        coordinates = row.get('wkt_geom')

        # Basic validation (repeat validation as needed for radar input)
        try: channel_no = int(channel_no_raw)
        except (ValueError, TypeError): continue # Skip if invalid channel
        if pd.isna(coordinates): continue # Skip if missing coords

        # Get coordinates needed by get_radar_data_incremental
        lon, lat = parse_coordinates_pipeline(coordinates)
        x, y = transform_coordinates_pipeline(lon, lat)

        if x is None or y is None:
             print(f"Skipping radar for {sensor_name} due to coordinate error.")
             continue

        # print(f"\n>>> Processing Radar for: {sensor_name} (X={x}, Y={y}) <<<")
        # #logger.info(f"Processing Radar for: {sensor_name} (X={x}, Y={y})")
        coord_tuple = (x, y)
        processed_coords.add(coord_tuple) # Add coords processed in this cycle

        # Call the radar update function (assuming it takes these args)
        radar_result_path = get_radar_data_incremental(x, y, sensor_name, channel_no)

        if radar_result_path:
            radar_success += 1
        else:
            radar_fail += 1
            # print(f": Radar data update failed for {sensor_name}.")
            # #logger.info(f"Radar data update failed for {sensor_name}.")
            # Continue processing other sensors?

    print(f"--- Radar Update Stage Finished (Success: {radar_success}, Fail: {radar_fail}) ---")


    # 3. Combine Data
    # This stage depends on the updated gauge and radar files existing
    print("\n--- Combining Data ---")
    combine_success = 0
    combine_fail = 0
    # Use the coordinates processed during the radar stage
    if not processed_coords:
        print("Warning: No coordinates were processed for radar, attempting to find coords from gauge/radar folders...")
        # Add fallback logic to find coords from folders if needed (like in get_combined_data.py standalone)
        # For now, assume processed_coords is populated if radar ran.
        pass

    for x, y in sorted(list(processed_coords)):
        if combine_data_for_location(x, y):
             combine_success += 1
        else:
             combine_fail +=1
             print(f": Combination failed for ({x},{y}).")
            #  #logger.info(f"Combination failed for ({x},{y}).")

    print(f"--- Combination Stage Finished (Success: {combine_success}, Fail: {combine_fail}) ---")


    # 4. Save to PKL
    print("\n--- Saving Updated Data to PKL ---")
    pkl_success = 0
    pkl_fail = 0
    # Use coordinates processed earlier to decide which PKL files to update
    print(pkl_input_folder)
    if os.path.exists(pkl_input_folder):
        # Iterate through combined files corresponding to processed coordinates
        for x,y in processed_coords:
             combined_filename = f"combined_data_({x},{y}).csv"
             combined_filepath = os.path.join(pkl_input_folder, combined_filename)
             if os.path.exists(combined_filepath):
                 if save_combined_to_pkl(combined_filepath):
                     pkl_success += 1
                 else:
                     pkl_fail += 1
             else:
                 print(f"Warning: Combined file not found for ({x},{y}), cannot save PKL.")
                 pkl_fail += 1 # Count as failure if combined file missing

    print(f"--- PKL Saving Stage Finished (Success: {pkl_success}, Fail: {pkl_fail}) ---")


    # --- Final Summary ---
    print("\n--- Update Cycle Summary ---")
    # Gauge summary is now part of its own script's output
    # print(f"Radar Data Fetch: {radar_success} succeeded, {radar_fail} failed.")
    # print(f"Combination: {combine_success} succeeded, {combine_fail} failed.")
    # print(f"PKL Saving: {pkl_success} succeeded, {pkl_fail} failed (for coords processed in this cycle).")
    # print(f"--- Update Cycle Finished: {datetime.now()} ---")
    #logger.info("Update cycle finished successfully.")
    #logger.info(f"Radar Data Fetch: {radar_success} succeeded, {radar_fail} failed.")
    #logger.info(f"Combination: {combine_success} succeeded, {combine_fail} failed.")
    #logger.info(f"PKL Saving: {pkl_success} succeeded, {pkl_fail} failed (for coords processed in this cycle).")
    #logger.info(f"Update cycle finished at {datetime.now()}")


# --- Main Execution Loop ---
# if __name__ == "__main__":
#     if RUN_INTERVAL_SECONDS > 0:
#         print(f"Starting live update pipeline. Running every {RUN_INTERVAL_SECONDS} seconds.")
#         print("Press Ctrl+C to stop.")
#         while True:
#             try:
#                 run_update_cycle()
#                 print(f"\nSleeping for {RUN_INTERVAL_SECONDS} seconds...")
#                 time.sleep(RUN_INTERVAL_SECONDS)
#             except KeyboardInterrupt:
#                 print("\nCtrl+C detected. Exiting pipeline.")
#                 break
#             except Exception as e:
#                 print(f"\nFATAL ERROR in main loop: {e}")
#                 import traceback
#                 traceback.print_exc() # Print full traceback for debugging
#                 print("Pipeline will attempt to restart after interval.")
#                 try:
#                     time.sleep(RUN_INTERVAL_SECONDS) # Wait before retrying
#                 except KeyboardInterrupt:
#                      print("\nCtrl+C detected during error sleep. Exiting pipeline.")
#                      break
#     else:
#         print("Running a single update cycle.")
#         run_update_cycle()
#         print("\nSingle run complete.")

if __name__ == "__main__":
    print("Starting live update pipeline.")
    import argparse
    parser = argparse.ArgumentParser(description="Live update pipeline script.")
    parser.add_argument('--run-once', action='store_true', help='Run the update cycle once and exit.')
    args = parser.parse_args()

    # Check if RUN_INTERVAL_SECONDS is defined, if not, define it here or ensure it's at the top of the script
    # For example: RUN_INTERVAL_SECONDS = 86400 # Default value if not defined elsewhere

    if args.run_once or ('RUN_INTERVAL_SECONDS' in globals() and RUN_INTERVAL_SECONDS == 0):
        print("Running a single update cycle (triggered by --run-once or RUN_INTERVAL_SECONDS=0).")
        run_update_cycle() # Assuming this is your main function for a single cycle
        print("\nSingle run complete.")
    elif 'RUN_INTERVAL_SECONDS' in globals() and RUN_INTERVAL_SECONDS > 0:
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
                import traceback
                traceback.print_exc() 
                print("Pipeline will attempt to restart after interval.")
                try:
                    time.sleep(RUN_INTERVAL_SECONDS) 
                except KeyboardInterrupt:
                     print("\nCtrl+C detected during error sleep. Exiting pipeline.")
                     break
    else:
        # Fallback or default behavior if neither --run-once nor RUN_INTERVAL_SECONDS conditions met as expected.
        # This might indicate a configuration issue in the script itself.
        print("Warning: Script not configured to run once or loop. Running once by default.")
        run_update_cycle()
        print("\nSingle run complete.")

# --- END OF FILE live_update_pipeline.py ---