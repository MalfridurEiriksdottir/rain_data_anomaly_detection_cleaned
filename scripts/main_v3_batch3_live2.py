
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging
import datetime

# Custom module imports
from config import CFG
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
from anomaly import flag_anomalies
from plotting3_batches2 import create_plots_with_error_markers, generate_html_dashboard
from network import get_nearest_neighbors
from batch_adjustment import compute_regional_adjustment

import json
from pathlib import Path
import time


# set up logging in a seperate text file
logging.basicConfig(
    filename=CFG.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger
logger = logging.getLogger(__name__)
# Set the logging level to INFO
logger.setLevel(logging.INFO)



# =============================================================================
# Helper Function Definition
# =============================================================================
def initialize_adjustment_columns(df):
    """
    Ensure the DataFrame contains all necessary columns for batch adjustment,
    initializing them with the appropriate data types if missing.
    
    Parameters:
        df (pd.DataFrame): The input data frame to initialize.
    
    Returns:
        pd.DataFrame: DataFrame with initialized columns.
    """
    cols_to_init = {
        'Alpha': float,
        'Rolling_Gauge': float,
        'Rolling_Radar': float,
        'Median_Neighbor_Alpha': float,
        'Network_Adjusted_Radar': float,
        'Adjusted_Diff_from_network': float,
        'Adjusted_Ratio_From_Network': float,
        'Flagged': bool,
        'Batch_Flag': bool,
        'Radar_Freg_Adjusted': float,
        'Final_Adjusted_Rainfall': float,
        'Rolling_Abs_Error': float,
        'Rolling_Prop_Flagged': float
    }
    
    for col, dtype in cols_to_init.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype, index=df.index)
    return df


import zoneinfo
# import pandas as pd # Already imported

STATE_FILE_PATH = CFG.BASE_DIR / "live_processing_state_v4.json" 
DEFAULT_HISTORICAL_START_STR = "2024-01-01 00:00:00" 


# --- State Management Functions ---
def load_last_processing_end_date():
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                state = json.load(f)
            last_end_str = state.get("last_successful_processing_end_date_utc")
            if last_end_str:
                dt_obj = pd.Timestamp(last_end_str)
                return dt_obj.tz_localize('UTC') if dt_obj.tzinfo is None else dt_obj.tz_convert('UTC')
            logger.info("No 'last_successful_processing_end_date_utc' in state file. Using default historical start.")
        except Exception as e:
            logger.error(f"Error loading state from {STATE_FILE_PATH}: {e}. Using default historical start.")
    else:
        logger.info(f"State file {STATE_FILE_PATH} not found. Assuming first run or state lost.")
    
    default_start_dt = pd.Timestamp(DEFAULT_HISTORICAL_START_STR)
    return default_start_dt.tz_localize('UTC') if default_start_dt.tzinfo is None else default_start_dt.tz_convert('UTC')

def save_last_processing_end_date(end_date_utc):
    try:
        if end_date_utc.tzinfo is None: # Ensure UTC awareness before saving
            end_date_utc = end_date_utc.tz_localize('UTC')
        elif str(end_date_utc.tzinfo).upper() != 'UTC':
            end_date_utc = end_date_utc.tz_convert('UTC')

        state = {"last_successful_processing_end_date_utc": end_date_utc.isoformat()}
        Path(STATE_FILE_PATH.parent).mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Saved last successful processing end date to state file: {end_date_utc.isoformat()}")
    except Exception as e:
        logger.error(f"Error saving state to {STATE_FILE_PATH}: {e}")

# =============================================================================
# Main Batch Adjustment Function
# =============================================================================
def run_daily_adjustment_cycle():
    """
    Run the batch adjustment process. This function loads sensor data, applies preprocessing,
    divides the data into 24-hour batches, computes adjustments (including anomaly detection),
    aggregates results, and generates visualization dashboards.
    """
    all_data = {}

    try:
        # ---------------------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------------------
        print("--- Loading Initial Data ---")
        logger.info("--- Loading Initial Data ---")
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        utm_to_channel_desc_map, svk_coords_utm = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        
        if not initial_all_data or not valid_coordinate_locations_utm:
            logger.error("Initial data loading failed or returned no data. Aborting cycle.")
            return True 
        logger.info(f"Loaded {len(initial_all_data)} sensor data frames initially.")
        
        all_data = {str(k): v for k, v in initial_all_data.items()}
        coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
        valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}
        logger.info(f"Converted keys to strings. {len(all_data)} sensor data frames with valid coordinates.")

        # ---------------------------------------------------------------------
        # 2. Preprocessing: Rename Columns, Standardize Timezone, Initialize Columns,
        #    and Apply Feature Engineering
        # ---------------------------------------------------------------------
        print("Preprocessing...")
        logger.info("--- Starting Preprocessing Step ---")
        processed_data_step2 = {}
        for coord, df_orig in tqdm(all_data.items(), desc="Preprocessing"):
            df = df_orig.copy()
            logger.info(f"Preprocessing sensor: {coord}")
            
            if df.empty:
                logger.warning(f"DataFrame for {coord} is empty before preprocessing. Skipping.")
                processed_data_step2[coord] = df_orig # Keep original empty df
                continue

            # Ensure index is a DatetimeIndex first
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    logger.info(f"Index for {coord} is not DatetimeIndex. Attempting conversion to pd.to_datetime.")
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    logger.error(f"Error converting index to DatetimeIndex for {coord}: {e}. Skipping this sensor.")
                    processed_data_step2[coord] = df_orig 
                    continue
            
            # Now, ensure it's UTC-aware
            try:
                if df.index.tz is None:
                    logger.info(f"Index for {coord} is naive. Localizing to UTC.")
                    df.index = df.index.tz_localize('UTC') # UNCOMMENTED
                elif str(df.index.tz).upper() != 'UTC': # More robust check for UTC
                    logger.info(f"Index timezone for {coord} is {df.index.tz}. Converting to UTC.")
                    df.index = df.index.tz_convert('UTC') # UNCOMMENTED
                # If already UTC, no action needed
            except Exception as e:
                logger.error(f"Error standardizing index timezone for {coord} to UTC: {e}. Skipping this sensor.")
                processed_data_step2[coord] = df_orig
                continue

            rename_dict = {}
            if 'Radar Data' in df.columns and 'Radar_Data_mm_per_min' not in df.columns:
                rename_dict['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns and 'Gauge_Data_mm_per_min' not in df.columns:
                rename_dict['Gauge Data'] = 'Gauge_Data_mm_per_min'
            
            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)
                logger.info(f"Renamed columns for {coord}: {rename_dict}")

            logger.info(f"[{coord}] Skipping application of shift_radar_to_utc in main loop, assuming data is already true UTC from ingestion.")
            
            gauge_col = 'Gauge_Data_mm_per_min'
            if gauge_col in df.columns:
                nan_count_before = df[gauge_col].isnull().sum()
                if nan_count_before > 0:
                     logger.info(f"Found {nan_count_before} NaNs in {gauge_col} for {coord}. Filling with 0.")
                df[gauge_col] = df[gauge_col].fillna(0.0)
            else:
                 logger.warning(f"Gauge column '{gauge_col}' not found for {coord}. Cannot fill NaNs.")

            df = initialize_adjustment_columns(df)

            if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
                if not df.empty:
                    df = apply_feature_engineering({coord: df})[coord]
                else:
                    logger.warning(f"DataFrame for {coord} is empty before feature engineering. Skipping feature_engineering.")
            
            processed_data_step2[coord] = df
            if not df.empty:
                logger.info(f"Finished preprocessing for {coord}. Rows: {len(df)}, Index TZ: {df.index.tz}, Min Time: {df.index.min()}, Max Time: {df.index.max()}")
            else:
                logger.info(f"Finished preprocessing for {coord}. DataFrame is empty.")

        all_data = processed_data_step2
        del processed_data_step2
        logger.info(f"--- Finished Preprocessing Step. Processed {len(all_data)} sensors. ---")

        # ---------------------------------------------------------------------
        # 3. Batch Processing Loop: Process Data in 24-Hour Batches
        # ---------------------------------------------------------------------
        print("Steps 3b, 4, 5: Processing in 24h batches...")
        logger.info("--- Starting Batch Processing Loop ---")
        
        global_start = pd.Timestamp.max.tz_localize('UTC') # Initialize with UTC timezone
        global_end = pd.Timestamp.min.tz_localize('UTC')   # Initialize with UTC timezone
        valid_indices_found = False

        for coord, df in all_data.items(): # Iterate through the processed data
            if (df is not None and isinstance(df, pd.DataFrame) and 
                isinstance(df.index, pd.DatetimeIndex) and 
                df.index.tz is not None and str(df.index.tz).upper() == 'UTC' and # Check for UTC explicitly
                not df.empty):
                
                current_start = df.index.min()
                current_end = df.index.max()
                
                # Additional check for valid timestamp values (not NaT)
                if pd.notna(current_start) and pd.notna(current_end):
                    logger.debug(f"Valid data for {coord}: Min Time {current_start}, Max Time {current_end}, TZ {df.index.tz}")
                    global_start = min(global_start, current_start)
                    global_end = max(global_end, current_end)
                    valid_indices_found = True
                else:
                    logger.warning(f"Data for {coord} has NaT in min/max time despite being DatetimeIndex. Min: {current_start}, Max: {current_end}")
            else:
                tz_info = df.index.tz if isinstance(df.index, pd.DatetimeIndex) else "Not DatetimeIndex"
                logger.warning(f"Invalid or empty DataFrame for {coord} during global time range calculation. Empty: {df.empty if isinstance(df, pd.DataFrame) else 'N/A'}, Index TZ: {tz_info}")
        
        logger.info(f"Calculated Global start: {global_start}, Global end: {global_end}, Valid indices found: {valid_indices_found}")

        if not valid_indices_found or global_start >= global_end or pd.isna(global_start) or pd.isna(global_end):
            logger.error("No valid data with UTC DatetimeIndex found or invalid global time range. Aborting batch processing.")
            # Potentially, you might still want to run plotting or CSV saving if some data was processed up to this point.
            # For now, returning True to indicate cycle "completed" but without batching.
            # Or, if this is critical, return False.
            # If returning True, subsequent plotting steps might fail or produce empty plots.
            # Consider what should happen if no batches are processed.
            # For robust pipeline, may need to handle this more gracefully in plotting/saving.
            # For now, let's try to proceed to plotting.
            if not all_data: # if no data was even loaded/preprocessed
                 return False # Indicate a more fundamental failure

        batch_periods = []
        if valid_indices_found and global_start < global_end and pd.notna(global_start) and pd.notna(global_end):
            batch_start_times = pd.date_range(start=global_start.floor(CFG.BATCH_DURATION),
                                              end=global_end, freq=CFG.BATCH_DURATION, tz='UTC')
            batch_periods = [(start, start + pd.Timedelta(CFG.BATCH_DURATION)) for start in batch_start_times]
        
        if not batch_periods:
            logger.warning("No batch periods generated. This might be due to a very short global time range or all data being in a single partial batch period. Skipping batch-specific processing.")
            # If no batches, the loop below won't run. Data in `all_data` will retain its state
            # from after preprocessing. The rolling error metrics and CSV saving will operate on this.
        else:
            logger.info(f"Generated {len(batch_periods)} batch periods.")

            for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):
                logger.info(f"--- Processing Batch: {batch_start} to {batch_end} ---")

                batch_avg_alphas = {}
                for coord, df in all_data.items():
                    if df.empty: continue
                    batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                    if batch_mask.any() and 'Alpha' in df.columns:
                        batch_alpha_series = df.loc[batch_mask, 'Alpha']
                        avg_alpha = batch_alpha_series.mean(skipna=True)
                        batch_avg_alphas[coord] = avg_alpha if pd.notna(avg_alpha) else 1.0
                    else:
                        batch_avg_alphas[coord] = 1.0

                for coord, df_target in all_data.items():
                    if df_target.empty: continue
                    batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                    if not batch_mask.any(): continue
                    
                    neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                    relevant_coords = [coord] + [n for n in neighbors if n in batch_avg_alphas]
                    batch_alpha_values = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords]
                    median_batch_avg_alpha = np.nanmedian(batch_alpha_values) if batch_alpha_values else 1.0
                    if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0

                    if 'Radar_Data_mm_per_min' in df_target.columns:
                        df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = \
                            (df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0) + CFG.EPSILON) * median_batch_avg_alpha # fillna(0) for safety
                    else:
                        df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = np.nan

                for coord, df in all_data.items():
                    if df.empty: continue
                    batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                    if not batch_mask.any(): continue

                    req_cols = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                    if all(c in df.columns for c in req_cols) and \
                       not df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'].isnull().all() and \
                       not df.loc[batch_mask, 'Rolling_Gauge'].isnull().all():
                        
                        rolling_batch_alpha_adj = df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar']\
                            .rolling(CFG.ROLLING_WINDOW, center=True, min_periods=1).mean().ffill().bfill()
                        rolling_gauge_batch = df.loc[batch_mask, 'Rolling_Gauge']
                        
                        if rolling_gauge_batch.empty or rolling_batch_alpha_adj.empty:
                            df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                            df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan
                            continue

                        # Ensure alignment if indices are not perfectly matching
                        rolling_gauge_aligned, rolling_batch_alpha_adj_aligned = rolling_gauge_batch.align(rolling_batch_alpha_adj, join='inner')

                        if not rolling_gauge_aligned.empty and not rolling_batch_alpha_adj_aligned.empty:
                            diff = rolling_batch_alpha_adj_aligned - rolling_gauge_aligned
                            ratio = (rolling_batch_alpha_adj_aligned + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                            ratio.replace([np.inf, -np.inf], 3.0, inplace=True) 
                            ratio_clipped = ratio.clip(upper=3.0).fillna(1.0) 

                            df.loc[diff.index, 'Adjusted_Diff_from_network'] = diff # Use aligned index
                            df.loc[ratio_clipped.index, 'Adjusted_Ratio_From_Network'] = ratio_clipped # Use aligned index
                        else: # After alignment, one or both are empty
                            # Get the original indices within the batch_mask to assign NaNs
                            original_indices_in_batch = df.loc[batch_mask].index
                            df.loc[original_indices_in_batch, 'Adjusted_Diff_from_network'] = np.nan
                            df.loc[original_indices_in_batch, 'Adjusted_Ratio_From_Network'] = np.nan
                    else:
                        df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                        df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan


                batch_flags_this_batch = {}
                for coord, df in all_data.items():
                    if df.empty: 
                        batch_flags_this_batch[coord] = False
                        continue
                    batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                    if not batch_mask.any():
                        batch_flags_this_batch[coord] = False 
                        continue

                    df_slice_for_anomaly = df.loc[batch_mask].copy() # Operate on a copy
                    if df_slice_for_anomaly.empty:
                         df.loc[batch_mask, 'Flagged'] = False
                         batch_flags_this_batch[coord] = False
                         continue

                    df_slice_dict = {coord: df_slice_for_anomaly} 
                    flagged_slice_dict = flag_anomalies(df_slice_dict) 
                    
                    if coord in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord].columns:
                        df.loc[batch_mask, 'Flagged'] = flagged_slice_dict[coord]['Flagged'].reindex(df.loc[batch_mask].index).fillna(False)
                    else:
                        df.loc[batch_mask, 'Flagged'] = False 

                    batch_flag = False
                    df_batch_flagged_slice = df.loc[batch_mask] 
                    if 'Flagged' in df_batch_flagged_slice.columns and not df_batch_flagged_slice.empty:
                        flagged_points = df_batch_flagged_slice['Flagged'].sum()
                        total_points = len(df_batch_flagged_slice)
                        percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                        if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT:
                            batch_flag = True
                    
                    batch_flags_this_batch[coord] = batch_flag
                    df.loc[batch_mask, 'Batch_Flag'] = batch_flag

                f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_this_batch)
                logger.info(f"Regional adjustment factor f_reg for batch {batch_start}-{batch_end}: {f_reg} (based on {valid_count} sensors)")

                batch_data_cache = {}
                for coord, df in all_data.items():
                    if df.empty: continue
                    batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                    if batch_mask.any():
                        cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                        batch_data_cache[coord] = df.loc[batch_mask, [c for c in cols_to_cache if c in df.columns]].copy()

                for coord, df_target in all_data.items():
                    if df_target.empty: continue
                    batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                    if not batch_mask.any(): continue

                    if 'Radar_Data_mm_per_min' in df_target.columns:
                        df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0) * f_reg
                    else:
                        df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = np.nan

                    neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                    current_batch_index = df_target.loc[batch_mask].index
                    if current_batch_index.empty: continue

                    num_valid_neighbors_series = pd.Series(0, index=current_batch_index, dtype=int)
                    local_factor_series = pd.Series(1.0, index=current_batch_index, dtype=float)
                    valid_neighbor_data = {}
                    neighbor_is_valid_mask_df = pd.DataFrame(index=current_batch_index)

                    for n in neighbors:
                        if n == coord: continue
                        is_batch_faulty_neighbor = batch_flags_this_batch.get(n, True)
                        current_neighbor_valid_mask = pd.Series(False, index=current_batch_index)

                        if not is_batch_faulty_neighbor:
                            neighbor_batch_df = batch_data_cache.get(n)
                            if neighbor_batch_df is not None and not neighbor_batch_df.empty:
                                neighbor_alpha = neighbor_batch_df.get('Alpha').reindex(current_batch_index)
                                neighbor_flagged = neighbor_batch_df.get('Flagged').reindex(current_batch_index)
                                
                                temp_valid_mask_n = pd.Series(True, index=current_batch_index)
                                if neighbor_flagged is not None: temp_valid_mask_n &= ~neighbor_flagged.fillna(True)
                                if neighbor_alpha is not None:
                                    temp_valid_mask_n &= neighbor_alpha.notna()
                                    temp_valid_mask_n &= (neighbor_alpha > 0)
                                else: temp_valid_mask_n = pd.Series(False, index=current_batch_index)
                                
                                current_neighbor_valid_mask = temp_valid_mask_n
                                if current_neighbor_valid_mask.any():
                                    valid_neighbor_data[n] = neighbor_alpha.where(current_neighbor_valid_mask)
                        neighbor_is_valid_mask_df[n] = current_neighbor_valid_mask

                    if not neighbor_is_valid_mask_df.empty:
                        num_valid_neighbors_series = neighbor_is_valid_mask_df.sum(axis=1)

                    if valid_neighbor_data:
                        df_valid_neighbor_alphas = pd.DataFrame(valid_neighbor_data)
                        local_factor_series = df_valid_neighbor_alphas.median(axis=1, skipna=True).fillna(1.0).clip(lower=0.1)
                    else: # Ensure local_factor_series is defined even if no valid_neighbor_data
                        local_factor_series = pd.Series(1.0, index=current_batch_index)


                    conditions = [num_valid_neighbors_series == 1, num_valid_neighbors_series == 2, num_valid_neighbors_series >= 3]
                    choices = [1.0 / 3.0, 2.0 / 3.0, 1.0]
                    weight_series = pd.Series(np.select(conditions, choices, default=0.0), index=current_batch_index)
                    
                    f_reg_series = pd.Series(f_reg, index=current_batch_index)
                    factor_combined_series = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)

                    if 'Radar_Data_mm_per_min' in df_target.columns:
                        raw_radar_series = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0)
                        final_adjusted_batch_series = raw_radar_series * factor_combined_series
                    else:
                        final_adjusted_batch_series = pd.Series(np.nan, index=current_batch_index)
                    
                    df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series.values
            logger.info("--- Finished Batch Processing Loop ---")
        
        # ---------------------------------------------------------------------
        # 7. Post-processing: Calculate Rolling Error Metrics (Outside Batch Loop)
        #    This will run even if no batch_periods were generated.
        # ---------------------------------------------------------------------
        print("Calculating final rolling error metrics...")
        logger.info("--- Calculating Final Rolling Error Metrics ---")
        for coord, df_orig in tqdm(all_data.items(), desc="Calculating Rolling Errors"):
            df = df_orig.copy()
            if df.empty: 
                logger.warning(f"DataFrame for {coord} is empty. Skipping rolling error calculation.")
                all_data[coord] = df # Keep it empty
                continue

            if 'Network_Adjusted_Radar' in df.columns and 'Gauge_Data_mm_per_min' in df.columns:
                gauge_filled = df['Gauge_Data_mm_per_min'].ffill().bfill() 
                net_adj_filled = df['Network_Adjusted_Radar'].ffill().bfill()

                if not net_adj_filled.isnull().all() and not gauge_filled.isnull().all():
                    abs_error = abs(net_adj_filled - gauge_filled)
                    df['Rolling_Abs_Error'] = abs_error.rolling(window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1).mean()
                    
                    if 'Flagged' in df.columns:
                        flag_points_numeric = df['Flagged'].astype(float) 
                        df['Rolling_Prop_Flagged'] = flag_points_numeric.rolling(window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1).mean()
                    else:
                        df['Rolling_Prop_Flagged'] = np.nan
                        logger.warning(f"'Flagged' column missing for {coord} when calculating Rolling_Prop_Flagged.")
                else:
                    df['Rolling_Abs_Error'] = np.nan
                    df['Rolling_Prop_Flagged'] = np.nan
                    logger.info(f"Insufficient data for rolling error metrics for {coord} (all NaNs in key series).")
            else:
                df['Rolling_Abs_Error'] = np.nan
                df['Rolling_Prop_Flagged'] = np.nan
                logger.warning(f"Missing 'Network_Adjusted_Radar' or 'Gauge_Data_mm_per_min' for {coord}. Cannot calculate rolling error metrics.")
            all_data[coord] = df
        logger.info("--- Finished Calculating Final Rolling Error Metrics ---")

        # --- 8. Save Detailed Output CSV per Coordinate ---
        print("Saving detailed output CSV files...")
        logger.info("--- Saving Detailed Output CSV Files ---")
        output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output" 
        output_csv_dir.mkdir(parents=True, exist_ok=True)

        cols_to_save = [
            'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Batch_Alpha_Adjusted_Radar',
            'Radar_Freg_Adjusted', 'Final_Adjusted_Rainfall', 'Flagged', 'Batch_Flag'
        ]
        
        for coord, df_final in tqdm(all_data.items(), desc="Saving Output CSVs"):
            if df_final.empty:
                logger.warning(f"DataFrame for {coord} is empty. Skipping CSV save.")
                continue

            coord_str = str(coord)
            save_cols_present = [col for col in cols_to_save if col in df_final.columns]
            if not save_cols_present:
                logger.warning(f"No columns to save for {coord_str}.")
                continue

            df_to_save = df_final[save_cols_present].copy()
            
            channel_desc_value = "Unknown_Channel"
            try:
                utm_tuple = coordinate_locations_utm.get(coord_str) 
                if utm_tuple:
                    channel_desc_value = utm_to_channel_desc_map.get(utm_tuple, "Description_Not_Found")
                else:
                    logger.warning(f"Could not find UTM tuple for coord '{coord_str}' in coordinate_locations_utm map.")
                    channel_desc_value = "UTM_Coord_Not_Found"
            except Exception as e:
                logger.error(f"Error looking up channel description for {coord_str}: {e}")
                channel_desc_value = "Lookup_Error"
            
            df_to_save['Channel'] = str(channel_desc_value)
            
            safe_coord_fname = coord_str.replace('(','').replace(')','').replace(',','_').replace(' ','')
            output_filename = output_csv_dir / f"{safe_coord_fname}_data_adjustments.csv"

            try:
                first_cols = ['Channel']
                remaining_cols = [col for col in df_to_save.columns if col not in first_cols]
                final_cols_order = first_cols + remaining_cols
                df_to_save[final_cols_order].to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z') 
                logger.info(f"Saved detailed data for {coord_str} to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save CSV for {coord_str}: {e}")
        
        logger.info(f"--- Finished Saving Detailed Output CSV Files to {output_csv_dir} ---")

        # ---------------------------------------------------------------------
        # 9. Final Aggregation, Saving, and Plotting
        # ---------------------------------------------------------------------
        events_df = None 
        all_data_iter0 = None 
        print("Generating final visualizations...")
        logger.info("--- Generating Final Visualizations ---")

        coord_str_to_channel_desc = {}
        if 'valid_coordinate_locations_utm' in locals() and 'utm_to_channel_desc_map' in locals():
            for str_coord_key, utm_tuple_val in valid_coordinate_locations_utm.items():
                 desc = utm_to_channel_desc_map.get(utm_tuple_val, "Unknown Channel")
                 coord_str_to_channel_desc[str_coord_key] = desc
        else:
            logger.warning("Could not create coordinate to channel map for plotting - required data missing.")

        logger.debug(f"Coordinate to channel description mapping for plots: {coord_str_to_channel_desc}")
        
        all_data_plotting = {str(k): v for k, v in all_data.items() if not v.empty} # Plot only non-empty
        
        if all_data_plotting: # Only plot if there's data
            create_plots_with_error_markers(all_data_plotting, valid_coordinate_locations_utm, coord_str_to_channel_desc,
                                            events_df=events_df, all_data_iter0=all_data_iter0)
            
            if gdf_wgs84 is not None and not gdf_wgs84.empty:
                generate_html_dashboard(all_data_plotting, valid_coordinate_locations_utm, gdf_wgs84,
                                        svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
            else:
                logger.warning("gdf_wgs84 is empty or None. Skipping HTML dashboard generation.")
        else:
            logger.warning("No data available for plotting. Skipping visualization generation.")
            
        logger.info("--- Finished Generating Final Visualizations ---")
        return True
    
    except Exception as e:
        logger.error("--- An error occurred during batch adjustment execution ---")
        logger.error(traceback.format_exc()) 
        return False
    
    finally:
        logger.info("\n--- Batch adjustment process finished for this cycle. ---")

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    RUN_INTERVAL_SECONDS = 24 * 60 * 60  # Default: every 24 hours
    # RUN_INTERVAL_SECONDS = 10 * 60     # For quicker testing: every 10 minutes
    # RUN_INTERVAL_SECONDS = 60          # For very quick testing: every 1 minute

    logger.info("===== Live adjustment pipeline started =====")
    logger.info("This script assumes external data ingestion scripts (gauge/radar) run periodically to update source files.")
    logger.info(f"Processing will occur approximately every {RUN_INTERVAL_SECONDS / 3600:.1f} hours.")

    while True:
        cycle_start_time = datetime.datetime.now()
        logger.info(f"===== New Live Update Cycle Triggered at {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        
        cycle_completed_successfully = run_daily_adjustment_cycle()

        cycle_end_time = datetime.datetime.now()
        if cycle_completed_successfully:
            logger.info(f"Daily adjustment cycle completed successfully at {cycle_end_time.strftime('%Y-%m-%d %H:%M:%S')}.")
        else:
            logger.error(f"Daily adjustment cycle FAILED at {cycle_end_time.strftime('%Y-%m-%d %H:%M:%S')}. Check logs for errors.")
        
        elapsed_time = (cycle_end_time - cycle_start_time).total_seconds()
        sleep_duration = max(0, RUN_INTERVAL_SECONDS - elapsed_time)

        logger.info(f"Cycle duration: {elapsed_time:.2f} seconds. Sleeping for {sleep_duration:.2f} seconds until the next cycle...")
        try:
            time.sleep(sleep_duration)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Exiting live processing loop.")
            break
        except Exception as e_slp:
            logger.error(f"An error occurred during sleep: {e_slp}. Retrying after a shorter delay (1 hour).")
            time.sleep(3600) 
    
    logger.info("===== Live adjustment pipeline shutting down. =====")
    logging.shutdown()