# --- START OF FILE save_as_pkl.py ---

import os
import pandas as pd
import re
import subprocess # Keep for standalone run option
import logging
logger = logging.getLogger(__name__)

# --- Define Constants ---
COMBINED_FOLDER = "combined_data"
OUTPUT_FOLDER = "all_data_pkl2" # Or your desired PKL output folder
COMBINED_FILENAME_PATTERN = r"combined_data_\((\d+),(\d+)\)\.csv"

# --- Helper function ---
def extract_coords_from_combined(filename):
    match = re.search(COMBINED_FILENAME_PATTERN, filename)
    if match:
        return match.group(1), match.group(2) # Return as strings initially
    return None, None

# --- Main Pickling Function ---
def save_combined_to_pkl(combined_csv_path, output_folder=OUTPUT_FOLDER):
    """Reads a combined CSV, processes it, and saves it as a PKL file."""
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.basename(combined_csv_path)
    x_str, y_str = extract_coords_from_combined(filename)

    if not x_str or not y_str:
        print(f"Warning: Could not extract coordinates from filename '{filename}'. Skipping pickling.")
        return False # Indicate failure

    # print(f"Processing combined file for pickling: {filename}")
    logger.info(f"Processing combined file for pickling: {filename}")

    try:
        df = pd.read_csv(combined_csv_path)

        # Convert time, set index, sort
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['time'], inplace=True) # Drop rows where time conversion failed
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        # Define output PKL filename
        output_filename = f"all_data_({x_str},{y_str}).pkl"
        output_path = os.path.join(output_folder, output_filename)

        # Save as .pkl
        df.to_pickle(output_path)
        # print(f"Saved PKL file: {output_filename}")
        logger.info(f"Saved PKL file: {output_filename}")

        return True # Indicate success

    except FileNotFoundError:
        print(f"ERROR: Combined CSV file not found at {combined_csv_path}")
        return False
    except Exception as e:
        print(f"ERROR processing or saving PKL for {filename}: {e}")
        return False

# --- Main execution part (if run standalone) ---
if __name__ == "__main__":
    # This part runs if the script is executed directly
    # It assumes the combined data CSV files already exist

    # --- Optional: Run prerequisite script if needed when run standalone ---
    # print("Running get_combined_data.py...")
    # try:
    #      subprocess.run(["python", "get_combined_data.py"], check=True, capture_output=True, text=True)
    # except subprocess.CalledProcessError as e:
    #      print(f"ERROR running get_combined_data.py: {e}")
    #      print(e.stderr)
    # print("get_combined_data script finished.\n")
    # --- End Optional Prerequisite Run ---

    print("\n--- Starting PKL Saving Process ---")
    if not os.path.exists(COMBINED_FOLDER):
        print(f"ERROR: Combined data folder '{COMBINED_FOLDER}' not found.")
        exit()

    combined_files = [f for f in os.listdir(COMBINED_FOLDER) if re.match(COMBINED_FILENAME_PATTERN, f)]

    if not combined_files:
        print(f"No combined data files found in '{COMBINED_FOLDER}' matching the pattern.")
        exit()

    print(f"Found {len(combined_files)} combined CSV files to process.")

    success_count = 0
    fail_count = 0
    for filename in combined_files:
        filepath = os.path.join(COMBINED_FOLDER, filename)
        if save_combined_to_pkl(filepath):
            success_count += 1
        else:
            fail_count += 1

    print("\n--- PKL Saving Summary ---")
    print(f"Successfully saved {success_count} PKL files.")
    if fail_count > 0:
        print(f"Failed or skipped saving {fail_count} PKL files (check logs for details).")
    print("PKL saving process finished.")

# --- END OF FILE save_as_pkl.py ---