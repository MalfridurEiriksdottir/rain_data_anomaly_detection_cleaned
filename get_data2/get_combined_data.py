# --- START OF FILE get_combined_data.py ---

import pandas as pd
import os
import re
import subprocess # Keep for now if running standalone is desired

import logging
import tqdm
logger = logging.getLogger(__name__)


# --- Define Constants ---
RADAR_FOLDER = "radar_data"
GAUGE_FOLDER = "gauge_data"
OUTPUT_FOLDER = "combined_data"
RADAR_FILENAME_PATTERN = r"VeVaRadar_.*_X(\d+)_Y(\d+)_.*\.csv" # Regex to find radar files and extract coords
RADAR_FILENAME_PATTERN = r"^VeVaRadar_X(\d+)_Y(\d+)_\.csv$"
GAUGE_FILENAME_PATTERN = r"gaugedata_(\d+)_(\d+)_\.csv" # Regex for gauge files

# --- Helper function to extract coordinates ---
def extract_coords_from_filename(filename, pattern):
    # print(filename)
    match = re.search(pattern, filename)
    if match:
        # Ensure consistency, return as integers
        return int(match.group(1)), int(match.group(2))
    return None, None

# --- Main Combining Function ---
def combine_data_for_location(x, y, radar_folder=RADAR_FOLDER, gauge_folder=GAUGE_FOLDER, output_folder=OUTPUT_FOLDER):
    """Combines radar and gauge data for a specific X, Y location."""
    os.makedirs(output_folder, exist_ok=True)

    # print(f"Combining data for coordinates ({x}, {y})...")
    # print(f"Radar folder: {radar_folder}")

    # Find the matching radar file (might be multiple bias/etc versions, handle this if needed)
    # For now, assume one relevant file per X,Y based on the pattern
    radar_file = None
    for f in os.listdir(radar_folder):

        rx, ry = extract_coords_from_filename(f, RADAR_FILENAME_PATTERN)
        if rx == x and ry == y:
            radar_file = f
            break # Found the first match
    # print(f"Radar file found: {radar_file}") # ADD

    if not radar_file:
        # print(f"Info: Radar file not found for coordinates ({x},{y}) in {radar_folder}")
        return False # Indicate that combination wasn't possible

    # Define expected gauge file name
    gauge_file = f"gaugedata_{x}_{y}_.csv"
    gauge_path = os.path.join(gauge_folder, gauge_file)
    gauge_path = os.path.normpath(gauge_path)

    radar_path = os.path.join(radar_folder, radar_file)
    radar_path = os.path.normpath(radar_path)

    if not os.path.exists(gauge_path):
        # print(f"Info: Gauge file not found for coordinates ({x},{y}) at {gauge_path}")
        return False # Cannot combine

    # print(f"Combining data for ({x},{y}): Radar='{radar_file}', Gauge='{gauge_file}'")
    logger.info(f"Combining data for ({x},{y}): Radar='{radar_file}', Gauge='{gauge_file}'")
    # Inside combine_data_for_location function in get_combined_data.py

    try:
        logger.info("GING THE PROCESS OF COMBINATION") # ADD
        logger.info(f" ({x},{y}): Trying to load radar file: {radar_path}") # ADD
        # Load radar data
        
        # print('GING THE PROCESS OF COMBINATION') # ADD
        # print(f" ({x},{y}): Trying to load radar file: {radar_path}") # ADD
        radar_df = pd.read_csv(radar_path)
        logger.info(f" ({x},{y}): Loaded radar. Columns: {radar_df.columns.tolist()}, Shape: {radar_df.shape}") # ADD
        # print(f" ({x},{y}): Loaded radar. Columns: {radar_df.columns.tolist()}, Shape: {radar_df.shape}") # ADD

        # ... clean headers ...
        radar_df.columns = [c.strip().strip('"') for c in radar_df.columns]
        # print(f" ({x},{y}): Cleaned radar headers.") # ADD
        logger.info(f" ({x},{y}): Cleaned radar headers.") # ADD

        # ... parse radar time ...
        # print(f" ({x},{y}): Parsing radar time...") # ADD
        logger.info(f" ({x},{y}): Parsing radar time...") # ADD
        radar_df['time'] = pd.to_datetime(radar_df['time'], errors='coerce')
        radar_df.dropna(subset=['time'], inplace=True)
        radar_df['time'] = radar_df['time'].dt.tz_localize(None)
        # print(f" ({x},{y}): Parsed radar time. Valid rows: {len(radar_df)}") # ADD
        logger.info(f" ({x},{y}): Parsed radar time. Valid rows: {len(radar_df)}") # ADD

        # ... parse radar value ...
        # print(f" ({x},{y}): Parsing radar value...") # ADD
        logger.info(f" ({x},{y}): Parsing radar value...") # ADD
        radar_df['Radar Data'] = pd.to_numeric(radar_df['value'].astype(str).str.replace(",", "."), errors='coerce')
        radar_df = radar_df[['time', 'Radar Data']].copy()
        radar_df.dropna(subset=['Radar Data'], inplace=True)
        # print(f" ({x},{y}): Parsed radar value. Valid rows: {len(radar_df)}") # ADD
        logger.info(f" ({x},{y}): Parsed radar value. Valid rows: {len(radar_df)}") # ADD


        # print(f" ({x},{y}): Trying to load gauge file: {gauge_path}") # ADD
        logger.info(f" ({x},{y}): Trying to load gauge file: {gauge_path}") # ADD
        gauge_df = pd.read_csv(gauge_path)
        # print(f" ({x},{y}): Loaded gauge. Columns: {gauge_df.columns.tolist()}, Shape: {gauge_df.shape}") # ADD
        logger.info(f" ({x},{y}): Loaded gauge. Columns: {gauge_df.columns.tolist()}, Shape: {gauge_df.shape}") # ADD

        # ... parse gauge time ...
        # print(f" ({x},{y}): Parsing gauge time...") # ADD
        logger.info(f" ({x},{y}): Parsing gauge time...") # ADD
        gauge_df['time'] = pd.to_datetime(gauge_df['datetime'], errors='coerce')
        gauge_df.dropna(subset=['time'], inplace=True)
        gauge_df['time'] = gauge_df['time'].dt.tz_localize(None)
        # print(f" ({x},{y}): Parsed gauge time. Valid rows: {len(gauge_df)}") # ADD
        logger.info(f" ({x},{y}): Parsed gauge time. Valid rows: {len(gauge_df)}") # ADD

        # ... parse gauge value ...
        # print(f" ({x},{y}): Parsing gauge value...") # ADD
        logger.info(f" ({x},{y}): Parsing gauge value...") # ADD
        gauge_df['Gauge Data'] = pd.to_numeric(gauge_df['value'], errors='coerce')
        gauge_df = gauge_df[['time', 'Gauge Data']].copy()
        gauge_df.dropna(subset=['Gauge Data'], inplace=True)
        # print(f" ({x},{y}): Parsed gauge value. Valid rows: {len(gauge_df)}") # ADD
        logger.info(f" ({x},{y}): Parsed gauge value. Valid rows: {len(gauge_df)}") # ADD


        # print(f" ({x},{y}): Merging dataframes...") # ADD
        logger.info(f" ({x},{y}): Merging dataframes...") # ADD
        combined_df = pd.merge(radar_df, gauge_df, on='time', how='outer')
        # print(f" ({x},{y}): Merged. Shape: {combined_df.shape}") # ADD
        logger.info(f" ({x},{y}): Merged. Shape: {combined_df.shape}") # ADD

        # ... sort and drop duplicates ...
        combined_df = combined_df.sort_values('time').drop_duplicates(subset=['time'], keep='first')
        # print(f" ({x},{y}): Sorted and dropped duplicates. Shape: {combined_df.shape}") # ADD
        logger.info(f" ({x},{y}): Sorted and dropped duplicates. Shape: {combined_df.shape}") # ADD

        # ... save combined file ...
        output_filename = f"combined_data_({x},{y}).csv"
        output_path = os.path.join(output_folder, output_filename)
        # print(f" ({x},{y}): Saving combined file to {output_path}") # ADD
        logger.info(f" ({x},{y}): Saving combined file to {output_path}") # ADD
        combined_df.to_csv(output_path, index=False)
        # print(f"Saved combined data to: {output_filename}")
        logger.info(f"Saved combined data to: {output_filename}") # ADD
        return True

    except Exception as e:
        # Error handling remains the same
        # print(f"ERROR combining data for ({x},{y}): {e}")
        # print(f"  Radar file: {radar_path}")
        # print(f"  Gauge file: {gauge_path}")
        logger.error(f" combining data for ({x},{y}): {e}")
        logger.error(f"  Radar file: {radar_path}")
        logger.error(f"  Gauge file: {gauge_path}")
        import traceback # Add traceback for more detail
        traceback.print_exc() # Add traceback
        return False



# --- Main execution part (if run standalone) ---
if __name__ == "__main__":



    print("\n--- Starting Data Combination Process ---")
    # Find all unique coordinates from gauge files (or radar files)
    coords_found = set()
    if os.path.exists(GAUGE_FOLDER):
        for f in os.listdir(GAUGE_FOLDER):
            x, y = extract_coords_from_filename(f, GAUGE_FILENAME_PATTERN)
            if x is not None and y is not None:
                coords_found.add((x, y))

    if not coords_found and os.path.exists(RADAR_FOLDER):
         print("No gauge files found, checking radar files for coordinates...")
         for f in os.listdir(RADAR_FOLDER):
             x, y = extract_coords_from_filename(f, RADAR_FILENAME_PATTERN)
             if x is not None and y is not None:
                 coords_found.add((x, y))

    if not coords_found:
        print(": No gauge or radar data files found to determine coordinates for combination.")
        exit()

    print(f"Found {len(coords_found)} unique coordinates to process for combination.")

    logger.info(f"Found {len(coords_found)} unique coordinates to process for combination.")

    success_count = 0
    fail_count = 0
    for x, y in sorted(list(coords_found)):
        if combine_data_for_location(x, y):
            success_count += 1
        else:
            fail_count += 1

    # print("\n--- Combination Summary ---")
    logger.info(f"Successfully combined data for {success_count} locations.")
    # print(f"Successfully combined data for {success_count} locations.")
    if fail_count > 0:
        # print(f"Failed or skipped combination for {fail_count} locations (check logs for details).")
        logger.warning(f"Failed or skipped combination for {fail_count} locations (check logs for details).")
    # print("Combination process finished.")
    logger.info("Combination process finished.")

# --- END OF FILE get_combined_data.py ---