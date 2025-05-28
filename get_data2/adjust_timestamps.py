import os
import pandas as pd
from datetime import datetime, timedelta, date
import calendar # Needed for finding last Sunday
import logging
from tqdm import tqdm

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

# --- Configuration ---
GAUGE_DATA_FOLDER = "../gauge_data"
FILE_PREFIX = "gaugedata_"
FILE_SUFFIX = ".csv"
# ---

# --- MANUAL TIME SHIFT HELPER FUNCTIONS ---

def find_last_sunday(year, month):
    """Finds the date of the last Sunday of a given month and year."""
    last_day = calendar.monthrange(year, month)[1]
    date_obj = date(year, month, last_day)
    offset = (date_obj.weekday() - 6) % 7 # Monday is 0, Sunday is 6
    last_sunday_date = date_obj - timedelta(days=offset)
    # Return the datetime at the very start of that Sunday (naive)
    return datetime(year, last_sunday_date.month, last_sunday_date.day, 0, 0, 0)

def is_danish_summer_time(dt_naive):
    """
    Checks if a naive datetime falls within Danish Summer Time (DST).
    DST runs from the last Sunday of March until the last Sunday of October.
    Assumes input dt_naive is a naive datetime object.
    NOTE: Ignores the exact transition hour (e.g., 2/3 AM).
    """
    if not isinstance(dt_naive, datetime):
        # Handle cases where input might not be a datetime (e.g., pd.NaT)
        return False

    year = dt_naive.year
    try:
        dst_start = find_last_sunday(year, 3)
        dst_end = find_last_sunday(year, 10)
        # Check if the naive datetime is within the period [start, end)
        return dst_start <= dt_naive < dst_end
    except ValueError: # Handle potential errors with date calculation
        logger.error(f"Error calculating DST bounds for year {year}. Assuming winter time for {dt_naive}.")
        return False


def manual_time_shift(dt_naive):
    """
    Applies a manual time shift based on Danish DST periods.
    Adds +4 hours during summer, +2 hours during winter.
    Input and Output are naive datetime objects. Handles NaT gracefully.
    """
    if pd.isna(dt_naive) or not isinstance(dt_naive, datetime):
        return pd.NaT # Return Not-a-Time if input is invalid

    if is_danish_summer_time(dt_naive):
        # Summer: Add 4 hours
        return dt_naive + timedelta(hours=4)
    else:
        # Winter: Add 2 hours
        return dt_naive + timedelta(hours=2)

# --- End MANUAL TIME SHIFT HELPER FUNCTIONS ---


# --- Main Reprocessing Script ---
logger.info(f"--- Starting Reprocessing Script for Gauge Files in '{GAUGE_DATA_FOLDER}' ---")
logger.warning(">>> This script will OVERWRITE original files with manually time-shifted data. <<<")
logger.warning(">>> Ensure you have a backup if needed. <<<")
logger.info("Applying manual shift: +2h winter / +4h summer (naive output).")

# Find relevant files
try:
    all_files = os.listdir(GAUGE_DATA_FOLDER)
    gauge_files_to_process = [
        f for f in all_files
        if f.startswith(FILE_PREFIX) and f.endswith(FILE_SUFFIX)
    ]
    if not gauge_files_to_process:
        logger.warning("No gauge data files found matching the pattern. Exiting.")
        exit()
    logger.info(f"Found {len(gauge_files_to_process)} gauge files to reprocess.")
except FileNotFoundError:
    logger.error(f"Error: Directory not found '{GAUGE_DATA_FOLDER}'. Exiting.")
    exit()
except Exception as e:
    logger.error(f"Error listing files in '{GAUGE_DATA_FOLDER}': {e}. Exiting.")
    exit()


# Process each file
files_processed = 0
files_skipped = 0
for filename in tqdm(gauge_files_to_process, desc="Reprocessing Gauge Files"):
    filepath = os.path.join(GAUGE_DATA_FOLDER, filename)
    logger.debug(f"Processing: {filepath}")

    try:
        # Read CSV - crucial to parse dates here
        df = pd.read_csv(filepath, parse_dates=['datetime'])

        # --- Datetime Column Validation and Preparation ---
        if 'datetime' not in df.columns:
            logger.warning(f"Skipping '{filename}': Missing 'datetime' column.")
            files_skipped += 1
            continue

        # Check if parsing worked
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            logger.warning(f"Skipping '{filename}': Could not parse 'datetime' column correctly.")
            files_skipped += 1
            continue

        # Create a working column for naive datetime
        df['datetime_naive'] = df['datetime']

        # Handle existing timezones: Standardize to naive UTC if present
        if df['datetime_naive'].dt.tz is not None:
            original_tz = df['datetime_naive'].dt.tz
            logger.warning(f"'{filename}' contains timezone-aware datetimes ({original_tz}). Standardizing to naive UTC before applying manual shift.")
            try:
                # Convert to UTC, then remove timezone info
                df['datetime_naive'] = df['datetime_naive'].dt.tz_convert('UTC').dt.tz_localize(None)
                logger.info(f"Standardized timezone-aware data in '{filename}' to naive UTC.")
            except Exception as tz_strip_err:
                logger.error(f"Error standardizing timezone in '{filename}': {tz_strip_err}. Skipping file.")
                files_skipped += 1
                continue

        # --- Apply the manual shift ---
        # Apply the shift to the prepared naive datetime column
        df['datetime_shifted_naive'] = df['datetime_naive'].apply(manual_time_shift)

        # Check for issues during shift (e.g., all NaT)
        if df['datetime_shifted_naive'].isnull().all():
             logger.warning(f"Manual shift resulted in all NaT values for '{filename}'. Check input data or shift logic. Skipping save.")
             files_skipped += 1
             continue

        # --- Prepare for Saving ---
        # Replace the original datetime column with the shifted one
        df['datetime'] = df['datetime_shifted_naive']

        # Select original columns + the potentially modified 'datetime'
        # Get original columns excluding temporary ones
        original_columns = [col for col in pd.read_csv(filepath, nrows=0).columns] # Read only header
        if 'datetime' not in original_columns: original_columns.append('datetime') # ensure it's there
        columns_to_save = [col for col in original_columns if col in df.columns]


        df_to_save = df[columns_to_save]

        # --- Overwrite the original file ---
        df_to_save.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S') # Save naive strings
        logger.info(f"Successfully reprocessed and overwrote '{filename}' with {len(df_to_save)} rows.")
        files_processed += 1

    except FileNotFoundError:
        logger.error(f"File not found during processing: '{filepath}'. Skipping.")
        files_skipped += 1
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping empty file: '{filename}'.")
        files_skipped += 1
    except Exception as e:
        logger.error(f"Failed to reprocess '{filename}': {e}", exc_info=True) # Log traceback for unexpected errors
        files_skipped += 1

# --- Final Summary ---
logger.info("--- Gauge File Reprocessing Complete ---")
logger.info(f"Successfully processed and overwrote: {files_processed} files.")
logger.info(f"Skipped due to errors or empty:      {files_skipped} files.")
# ---