# --- START OF FILE fix_gauge_to_utc.py ---
import os
import pandas as pd
from datetime import datetime, timedelta, date
import calendar
import logging
from tqdm import tqdm
import pytz # Use pytz for robust timezone handling

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ---

# --- Configuration ---
GAUGE_DATA_FOLDER = "../gauge_data"
FILE_PREFIX = "gaugedata_"
FILE_SUFFIX = ".csv"
EXPECTED_NAIVE_TIMEZONE = 'Europe/Copenhagen' # The original local timezone
# ---

# --- DST Helper Functions (Copied from manual shift script) ---
def find_last_sunday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    date_obj = date(year, month, last_day)
    offset = (date_obj.weekday() - 6) % 7
    last_sunday_date = date_obj - timedelta(days=offset)
    return datetime(year, last_sunday_date.month, last_sunday_date.day, 0, 0, 0)

def is_danish_summer_time(dt_naive):
    if not isinstance(dt_naive, datetime): return False
    year = dt_naive.year
    try:
        dst_start = find_last_sunday(year, 3)
        dst_end = find_last_sunday(year, 10)
        return dst_start <= dt_naive < dst_end
    except ValueError: return False

def reverse_manual_shift(dt_shifted_naive):
    """
    Reverses the +2h winter / +4h summer manual shift.
    Input is the naive datetime *after* the incorrect shift was applied.
    Output is the estimated original naive *local* datetime.
    Handles NaT gracefully.
    """
    if pd.isna(dt_shifted_naive) or not isinstance(dt_shifted_naive, datetime):
        return pd.NaT

    # We need to guess if the *original* time was summer/winter based on the *shifted* time.
    # This might be slightly off near transitions, but it's the best estimate.
    # Heuristic: Subtract the likely offset and check the DST status of the *result*.
    potential_winter_original = dt_shifted_naive - timedelta(hours=2)
    potential_summer_original = dt_shifted_naive - timedelta(hours=4)

    # If the time resulting from subtracting 4h is summer time, that was likely the original
    if is_danish_summer_time(potential_summer_original):
        return potential_summer_original
    # Otherwise, assume the original was winter time (or the summer check was wrong)
    else:
        # Double-check: if subtracting 2h results in winter time, use that
        if not is_danish_summer_time(potential_winter_original):
             return potential_winter_original
        else:
             # Ambiguous case - shifted time is near a transition.
             # Defaulting to winter subtraction might be safer, or log a warning.
             logger.warning(f"Ambiguous DST transition near {dt_shifted_naive}. Defaulting to reversing winter shift.")
             return potential_winter_original


# --- Main Reprocessing Script to UTC ---
logger.info(f"--- Starting UTC Fix Script for Gauge Files in '{GAUGE_DATA_FOLDER}' ---")
logger.warning(">>> This script will OVERWRITE original files with corrected UTC timestamps. <<<")
logger.warning(">>> Ensure you have a backup if needed. <<<")
logger.info(f"Interpreting naive times as previously manually shifted, reversing shift, localizing to '{EXPECTED_NAIVE_TIMEZONE}', converting to UTC.")

# Find relevant files
try:
    all_files = os.listdir(GAUGE_DATA_FOLDER)
    gauge_files_to_process = [f for f in all_files if f.startswith(FILE_PREFIX) and f.endswith(FILE_SUFFIX)]
    if not gauge_files_to_process: logger.warning("No gauge files found. Exiting."); exit()
    logger.info(f"Found {len(gauge_files_to_process)} gauge files to fix.")
except Exception as e: logger.error(f"Error listing files: {e}. Exiting."); exit()

files_processed = 0; files_skipped = 0
for filename in tqdm(gauge_files_to_process, desc="Fixing Gauge Files to UTC"):
    filepath = os.path.join(GAUGE_DATA_FOLDER, filename)
    logger.debug(f"Processing: {filepath}")

    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])

        if 'datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            logger.warning(f"Skipping '{filename}': Invalid/missing 'datetime' column.")
            files_skipped += 1; continue

        # --- Assume current 'datetime' is the naive, manually shifted time ---
        df['datetime_shifted_naive'] = df['datetime']

        # Warn if it already has timezone info (unexpected)
        if df['datetime_shifted_naive'].dt.tz is not None:
            logger.warning(f"'{filename}' already has timezone {df['datetime_shifted_naive'].dt.tz}. Attempting to make naive UTC first.")
            try:
                 df['datetime_shifted_naive'] = df['datetime_shifted_naive'].dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception as tz_err:
                 logger.error(f"Could not handle existing timezone in '{filename}': {tz_err}. Skipping.")
                 files_skipped += 1; continue

        # --- Reverse the shift ---
        df['datetime_original_naive_local'] = df['datetime_shifted_naive'].apply(reverse_manual_shift)

        # Drop rows where reversal failed (returned NaT)
        original_count = len(df)
        df.dropna(subset=['datetime_original_naive_local'], inplace=True)
        if len(df) < original_count:
            logger.warning(f"Dropped {original_count - len(df)} rows in '{filename}' due to shift reversal errors.")

        if df.empty:
            logger.warning(f"No valid data remaining in '{filename}' after reversing shift. Skipping save.")
            files_skipped += 1; continue

        # --- Localize to Original Timezone and Convert to UTC ---
        try:
             # 1. Localize the estimated original naive time
             df['datetime_localized'] = df['datetime_original_naive_local'].dt.tz_localize(EXPECTED_NAIVE_TIMEZONE, ambiguous='infer', nonexistent='shift_forward')
             # 2. Convert to UTC
             df['datetime_utc'] = df['datetime_localized'].dt.tz_convert('UTC')
        except Exception as tz_conv_err:
             logger.error(f"Error converting time to UTC for '{filename}': {tz_conv_err}. Skipping file.")
             files_skipped += 1; continue

        # --- Prepare for Saving ---
        # Replace the original datetime column with the CORRECT UTC one
        df['datetime'] = df['datetime_utc']

        # Select original columns + the corrected 'datetime'
        original_columns = [col for col in pd.read_csv(filepath, nrows=0).columns] # Read header
        if 'datetime' not in original_columns: original_columns.append('datetime')
        columns_to_save = [col for col in original_columns if col in df.columns]
        df_to_save = df[columns_to_save]

        # --- Overwrite the original file with UTC data ---
        # Save WITH timezone information
        df_to_save.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S%z')
        logger.info(f"Successfully fixed and overwrote '{filename}' with UTC timestamps ({len(df_to_save)} rows).")
        files_processed += 1

    except pd.errors.EmptyDataError: logger.warning(f"Skipping empty file: '{filename}'."); files_skipped += 1
    except Exception as e: logger.error(f"Failed to fix '{filename}': {e}", exc_info=True); files_skipped += 1

# --- Final Summary ---
logger.info("--- Gauge File UTC Fixing Complete ---")
logger.info(f"Successfully processed and overwrote: {files_processed} files.")
logger.info(f"Skipped due to errors or empty:      {files_skipped} files.")
# --- END OF FILE fix_gauge_to_utc.py ---