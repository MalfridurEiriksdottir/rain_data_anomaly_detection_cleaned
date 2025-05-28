
# import pandas as pd
# from tqdm import tqdm
# import traceback
# import numpy as np
# import logging
# import datetime

# # Custom module imports
# from config import CFG
# from data_loading import (load_sensor_data, load_target_coordinates,
#                           process_sensor_metadata, load_time_series_data)
# from feature_engineering import apply_feature_engineering
# from anomaly import flag_anomalies
# from plotting3_batches2 import create_plots_with_error_markers, generate_html_dashboard
# from network import get_nearest_neighbors
# from batch_adjustment import compute_regional_adjustment

# import json
# from pathlib import Path
# import time


# # set up logging in a seperate text file
# logging.basicConfig(
#     filename=CFG.LOG_FILE,
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# # Create a logger
# logger = logging.getLogger(__name__)
# # Set the logging level to INFO
# logger.setLevel(logging.INFO)



# # =============================================================================
# # Helper Function Definition
# # =============================================================================
# def initialize_adjustment_columns(df):
#     """
#     Ensure the DataFrame contains all necessary columns for batch adjustment,
#     initializing them with the appropriate data types if missing.
    
#     Parameters:
#         df (pd.DataFrame): The input data frame to initialize.
    
#     Returns:
#         pd.DataFrame: DataFrame with initialized columns.
#     """
#     cols_to_init = {
#         'Alpha': float,
#         'Rolling_Gauge': float,
#         'Rolling_Radar': float,
#         'Median_Neighbor_Alpha': float,
#         'Network_Adjusted_Radar': float,
#         'Adjusted_Diff_from_network': float,
#         'Adjusted_Ratio_From_Network': float,
#         'Flagged': bool,
#         'Batch_Flag': bool,
#         'Radar_Freg_Adjusted': float,
#         'Final_Adjusted_Rainfall': float,
#         'Rolling_Abs_Error': float,
#         'Rolling_Prop_Flagged': float
#     }
    
#     for col, dtype in cols_to_init.items():
#         if col not in df.columns:
#             df[col] = pd.Series(dtype=dtype, index=df.index)
#     return df


# import zoneinfo
# # import pandas as pd # Already imported

# STATE_FILE_PATH = CFG.BASE_DIR / "live_processing_state_v4.json" 
# DEFAULT_HISTORICAL_START_STR = "2024-01-01 00:00:00" 


# # --- State Management Functions ---
# def load_last_processing_end_date():
#     if STATE_FILE_PATH.exists():
#         try:
#             with open(STATE_FILE_PATH, 'r') as f:
#                 state = json.load(f)
#             last_end_str = state.get("last_successful_processing_end_date_utc")
#             if last_end_str:
#                 dt_obj = pd.Timestamp(last_end_str)
#                 return dt_obj.tz_localize('UTC') if dt_obj.tzinfo is None else dt_obj.tz_convert('UTC')
#             logger.info("No 'last_successful_processing_end_date_utc' in state file. Using default historical start.")
#         except Exception as e:
#             logger.error(f"Error loading state from {STATE_FILE_PATH}: {e}. Using default historical start.")
#     else:
#         logger.info(f"State file {STATE_FILE_PATH} not found. Assuming first run or state lost.")
    
#     default_start_dt = pd.Timestamp(DEFAULT_HISTORICAL_START_STR)
#     return default_start_dt.tz_localize('UTC') if default_start_dt.tzinfo is None else default_start_dt.tz_convert('UTC')

# def save_last_processing_end_date(end_date_utc):
#     try:
#         if end_date_utc.tzinfo is None: # Ensure UTC awareness before saving
#             end_date_utc = end_date_utc.tz_localize('UTC')
#         elif str(end_date_utc.tzinfo).upper() != 'UTC':
#             end_date_utc = end_date_utc.tz_convert('UTC')

#         state = {"last_successful_processing_end_date_utc": end_date_utc.isoformat()}
#         Path(STATE_FILE_PATH.parent).mkdir(parents=True, exist_ok=True)
#         with open(STATE_FILE_PATH, 'w') as f:
#             json.dump(state, f, indent=4)
#         logger.info(f"Saved last successful processing end date to state file: {end_date_utc.isoformat()}")
#     except Exception as e:
#         logger.error(f"Error saving state to {STATE_FILE_PATH}: {e}")

# # =============================================================================
# # Main Batch Adjustment Function
# # =============================================================================
# def run_daily_adjustment_cycle():
#     """
#     Run the batch adjustment process. This function loads sensor data, applies preprocessing,
#     divides the data into 24-hour batches, computes adjustments (including anomaly detection),
#     aggregates results, and generates visualization dashboards.
#     """
#     all_data = {}

#     try:
#         # ---------------------------------------------------------------------
#         # 1. Load Data
#         # ---------------------------------------------------------------------
#         print("--- Loading Initial Data ---")
#         logger.info("--- Loading Initial Data ---")
#         target_coords, coordinate_locations_utm = load_target_coordinates()
#         sensordata, gdf_wgs84 = load_sensor_data()
#         utm_to_channel_desc_map, svk_coords_utm = process_sensor_metadata(sensordata)
#         initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        
#         if not initial_all_data or not valid_coordinate_locations_utm:
#             logger.error("Initial data loading failed or returned no data. Aborting cycle.")
#             return True 
#         logger.info(f"Loaded {len(initial_all_data)} sensor data frames initially.")
        
#         all_data = {str(k): v for k, v in initial_all_data.items()}
#         coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
#         valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}
#         logger.info(f"Converted keys to strings. {len(all_data)} sensor data frames with valid coordinates.")

#         # ---------------------------------------------------------------------
#         # 2. Preprocessing: Rename Columns, Standardize Timezone, Initialize Columns,
#         #    and Apply Feature Engineering
#         # ---------------------------------------------------------------------
#         print("Preprocessing...")
#         logger.info("--- Starting Preprocessing Step ---")
#         processed_data_step2 = {}
#         for coord, df_orig in tqdm(all_data.items(), desc="Preprocessing"):
#             df = df_orig.copy()
#             logger.info(f"Preprocessing sensor: {coord}")
            
#             if df.empty:
#                 logger.warning(f"DataFrame for {coord} is empty before preprocessing. Skipping.")
#                 processed_data_step2[coord] = df_orig # Keep original empty df
#                 continue

#             # Ensure index is a DatetimeIndex first
#             if not isinstance(df.index, pd.DatetimeIndex):
#                 try:
#                     logger.info(f"Index for {coord} is not DatetimeIndex. Attempting conversion to pd.to_datetime.")
#                     df.index = pd.to_datetime(df.index)
#                 except Exception as e:
#                     logger.error(f"Error converting index to DatetimeIndex for {coord}: {e}. Skipping this sensor.")
#                     processed_data_step2[coord] = df_orig 
#                     continue
            
#             # Now, ensure it's UTC-aware
#             try:
#                 if df.index.tz is None:
#                     logger.info(f"Index for {coord} is naive. Localizing to UTC.")
#                     df.index = df.index.tz_localize('UTC') # UNCOMMENTED
#                 elif str(df.index.tz).upper() != 'UTC': # More robust check for UTC
#                     logger.info(f"Index timezone for {coord} is {df.index.tz}. Converting to UTC.")
#                     df.index = df.index.tz_convert('UTC') # UNCOMMENTED
#                 # If already UTC, no action needed
#             except Exception as e:
#                 logger.error(f"Error standardizing index timezone for {coord} to UTC: {e}. Skipping this sensor.")
#                 processed_data_step2[coord] = df_orig
#                 continue

#             rename_dict = {}
#             if 'Radar Data' in df.columns and 'Radar_Data_mm_per_min' not in df.columns:
#                 rename_dict['Radar Data'] = 'Radar_Data_mm_per_min'
#             if 'Gauge Data' in df.columns and 'Gauge_Data_mm_per_min' not in df.columns:
#                 rename_dict['Gauge Data'] = 'Gauge_Data_mm_per_min'
            
#             if rename_dict:
#                 df.rename(columns=rename_dict, inplace=True)
#                 logger.info(f"Renamed columns for {coord}: {rename_dict}")

#             logger.info(f"[{coord}] Skipping application of shift_radar_to_utc in main loop, assuming data is already true UTC from ingestion.")
            
#             gauge_col = 'Gauge_Data_mm_per_min'
#             if gauge_col in df.columns:
#                 nan_count_before = df[gauge_col].isnull().sum()
#                 if nan_count_before > 0:
#                      logger.info(f"Found {nan_count_before} NaNs in {gauge_col} for {coord}. Filling with 0.")
#                 df[gauge_col] = df[gauge_col].fillna(0.0)
#             else:
#                  logger.warning(f"Gauge column '{gauge_col}' not found for {coord}. Cannot fill NaNs.")

#             df = initialize_adjustment_columns(df)

#             if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
#                 if not df.empty:
#                     df = apply_feature_engineering({coord: df})[coord]
#                 else:
#                     logger.warning(f"DataFrame for {coord} is empty before feature engineering. Skipping feature_engineering.")
            
#             processed_data_step2[coord] = df
#             if not df.empty:
#                 logger.info(f"Finished preprocessing for {coord}. Rows: {len(df)}, Index TZ: {df.index.tz}, Min Time: {df.index.min()}, Max Time: {df.index.max()}")
#             else:
#                 logger.info(f"Finished preprocessing for {coord}. DataFrame is empty.")

#         all_data = processed_data_step2
#         del processed_data_step2
#         logger.info(f"--- Finished Preprocessing Step. Processed {len(all_data)} sensors. ---")

#         # ---------------------------------------------------------------------
#         # 3. Batch Processing Loop: Process Data in 24-Hour Batches
#         # ---------------------------------------------------------------------
#         print("Steps 3b, 4, 5: Processing in 24h batches...")
#         logger.info("--- Starting Batch Processing Loop ---")
        
#         global_start = pd.Timestamp.max.tz_localize('UTC') # Initialize with UTC timezone
#         global_end = pd.Timestamp.min.tz_localize('UTC')   # Initialize with UTC timezone
#         valid_indices_found = False

#         for coord, df in all_data.items(): # Iterate through the processed data
#             if (df is not None and isinstance(df, pd.DataFrame) and 
#                 isinstance(df.index, pd.DatetimeIndex) and 
#                 df.index.tz is not None and str(df.index.tz).upper() == 'UTC' and # Check for UTC explicitly
#                 not df.empty):
                
#                 current_start = df.index.min()
#                 current_end = df.index.max()
                
#                 # Additional check for valid timestamp values (not NaT)
#                 if pd.notna(current_start) and pd.notna(current_end):
#                     logger.debug(f"Valid data for {coord}: Min Time {current_start}, Max Time {current_end}, TZ {df.index.tz}")
#                     global_start = min(global_start, current_start)
#                     global_end = max(global_end, current_end)
#                     valid_indices_found = True
#                 else:
#                     logger.warning(f"Data for {coord} has NaT in min/max time despite being DatetimeIndex. Min: {current_start}, Max: {current_end}")
#             else:
#                 tz_info = df.index.tz if isinstance(df.index, pd.DatetimeIndex) else "Not DatetimeIndex"
#                 logger.warning(f"Invalid or empty DataFrame for {coord} during global time range calculation. Empty: {df.empty if isinstance(df, pd.DataFrame) else 'N/A'}, Index TZ: {tz_info}")
        
#         logger.info(f"Calculated Global start: {global_start}, Global end: {global_end}, Valid indices found: {valid_indices_found}")

#         if not valid_indices_found or global_start >= global_end or pd.isna(global_start) or pd.isna(global_end):
#             logger.error("No valid data with UTC DatetimeIndex found or invalid global time range. Aborting batch processing.")
#             # Potentially, you might still want to run plotting or CSV saving if some data was processed up to this point.
#             # For now, returning True to indicate cycle "completed" but without batching.
#             # Or, if this is critical, return False.
#             # If returning True, subsequent plotting steps might fail or produce empty plots.
#             # Consider what should happen if no batches are processed.
#             # For robust pipeline, may need to handle this more gracefully in plotting/saving.
#             # For now, let's try to proceed to plotting.
#             if not all_data: # if no data was even loaded/preprocessed
#                  return False # Indicate a more fundamental failure

#         batch_periods = []
#         if valid_indices_found and global_start < global_end and pd.notna(global_start) and pd.notna(global_end):
#             batch_start_times = pd.date_range(start=global_start.floor(CFG.BATCH_DURATION),
#                                               end=global_end, freq=CFG.BATCH_DURATION, tz='UTC')
#             batch_periods = [(start, start + pd.Timedelta(CFG.BATCH_DURATION)) for start in batch_start_times]
        
#         if not batch_periods:
#             logger.warning("No batch periods generated. This might be due to a very short global time range or all data being in a single partial batch period. Skipping batch-specific processing.")
#             # If no batches, the loop below won't run. Data in `all_data` will retain its state
#             # from after preprocessing. The rolling error metrics and CSV saving will operate on this.
#         else:
#             logger.info(f"Generated {len(batch_periods)} batch periods.")

#             for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):
#                 logger.info(f"--- Processing Batch: {batch_start} to {batch_end} ---")

#                 batch_avg_alphas = {}
#                 for coord, df in all_data.items():
#                     if df.empty: continue
#                     batch_mask = (df.index >= batch_start) & (df.index < batch_end)
#                     if batch_mask.any() and 'Alpha' in df.columns:
#                         batch_alpha_series = df.loc[batch_mask, 'Alpha']
#                         avg_alpha = batch_alpha_series.mean(skipna=True)
#                         batch_avg_alphas[coord] = avg_alpha if pd.notna(avg_alpha) else 1.0
#                     else:
#                         batch_avg_alphas[coord] = 1.0

#                 for coord, df_target in all_data.items():
#                     if df_target.empty: continue
#                     batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
#                     if not batch_mask.any(): continue
                    
#                     neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
#                     relevant_coords = [coord] + [n for n in neighbors if n in batch_avg_alphas]
#                     batch_alpha_values = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords]
#                     median_batch_avg_alpha = np.nanmedian(batch_alpha_values) if batch_alpha_values else 1.0
#                     if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0

#                     if 'Radar_Data_mm_per_min' in df_target.columns:
#                         df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = \
#                             (df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0) + CFG.EPSILON) * median_batch_avg_alpha # fillna(0) for safety
#                     else:
#                         df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = np.nan

#                 for coord, df in all_data.items():
#                     if df.empty: continue
#                     batch_mask = (df.index >= batch_start) & (df.index < batch_end)
#                     if not batch_mask.any(): continue

#                     req_cols = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
#                     if all(c in df.columns for c in req_cols) and \
#                        not df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'].isnull().all() and \
#                        not df.loc[batch_mask, 'Rolling_Gauge'].isnull().all():
                        
#                         rolling_batch_alpha_adj = df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar']\
#                             .rolling(CFG.ROLLING_WINDOW, center=True, min_periods=1).mean().ffill().bfill()
#                         rolling_gauge_batch = df.loc[batch_mask, 'Rolling_Gauge']
                        
#                         if rolling_gauge_batch.empty or rolling_batch_alpha_adj.empty:
#                             df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
#                             df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan
#                             continue

#                         # Ensure alignment if indices are not perfectly matching
#                         rolling_gauge_aligned, rolling_batch_alpha_adj_aligned = rolling_gauge_batch.align(rolling_batch_alpha_adj, join='inner')

#                         if not rolling_gauge_aligned.empty and not rolling_batch_alpha_adj_aligned.empty:
#                             diff = rolling_batch_alpha_adj_aligned - rolling_gauge_aligned
#                             ratio = (rolling_batch_alpha_adj_aligned + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
#                             ratio.replace([np.inf, -np.inf], 3.0, inplace=True) 
#                             ratio_clipped = ratio.clip(upper=3.0).fillna(1.0) 

#                             df.loc[diff.index, 'Adjusted_Diff_from_network'] = diff # Use aligned index
#                             df.loc[ratio_clipped.index, 'Adjusted_Ratio_From_Network'] = ratio_clipped # Use aligned index
#                         else: # After alignment, one or both are empty
#                             # Get the original indices within the batch_mask to assign NaNs
#                             original_indices_in_batch = df.loc[batch_mask].index
#                             df.loc[original_indices_in_batch, 'Adjusted_Diff_from_network'] = np.nan
#                             df.loc[original_indices_in_batch, 'Adjusted_Ratio_From_Network'] = np.nan
#                     else:
#                         df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
#                         df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan


#                 batch_flags_this_batch = {}
#                 for coord, df in all_data.items():
#                     if df.empty: 
#                         batch_flags_this_batch[coord] = False
#                         continue
#                     batch_mask = (df.index >= batch_start) & (df.index < batch_end)
#                     if not batch_mask.any():
#                         batch_flags_this_batch[coord] = False 
#                         continue

#                     df_slice_for_anomaly = df.loc[batch_mask].copy() # Operate on a copy
#                     if df_slice_for_anomaly.empty:
#                          df.loc[batch_mask, 'Flagged'] = False
#                          batch_flags_this_batch[coord] = False
#                          continue

#                     df_slice_dict = {coord: df_slice_for_anomaly} 
#                     flagged_slice_dict = flag_anomalies(df_slice_dict) 
                    
#                     if coord in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord].columns:
#                         df.loc[batch_mask, 'Flagged'] = flagged_slice_dict[coord]['Flagged'].reindex(df.loc[batch_mask].index).fillna(False)
#                     else:
#                         df.loc[batch_mask, 'Flagged'] = False 

#                     batch_flag = False
#                     df_batch_flagged_slice = df.loc[batch_mask] 
#                     if 'Flagged' in df_batch_flagged_slice.columns and not df_batch_flagged_slice.empty:
#                         flagged_points = df_batch_flagged_slice['Flagged'].sum()
#                         total_points = len(df_batch_flagged_slice)
#                         percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
#                         if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT:
#                             batch_flag = True
                    
#                     batch_flags_this_batch[coord] = batch_flag
#                     df.loc[batch_mask, 'Batch_Flag'] = batch_flag

#                 f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_this_batch)
#                 logger.info(f"Regional adjustment factor f_reg for batch {batch_start}-{batch_end}: {f_reg} (based on {valid_count} sensors)")

#                 batch_data_cache = {}
#                 for coord, df in all_data.items():
#                     if df.empty: continue
#                     batch_mask = (df.index >= batch_start) & (df.index < batch_end)
#                     if batch_mask.any():
#                         cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
#                         batch_data_cache[coord] = df.loc[batch_mask, [c for c in cols_to_cache if c in df.columns]].copy()

#                 for coord, df_target in all_data.items():
#                     if df_target.empty: continue
#                     batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
#                     if not batch_mask.any(): continue

#                     if 'Radar_Data_mm_per_min' in df_target.columns:
#                         df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0) * f_reg
#                     else:
#                         df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = np.nan

#                     neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
#                     current_batch_index = df_target.loc[batch_mask].index
#                     if current_batch_index.empty: continue

#                     num_valid_neighbors_series = pd.Series(0, index=current_batch_index, dtype=int)
#                     local_factor_series = pd.Series(1.0, index=current_batch_index, dtype=float)
#                     valid_neighbor_data = {}
#                     neighbor_is_valid_mask_df = pd.DataFrame(index=current_batch_index)

#                     for n in neighbors:
#                         if n == coord: continue
#                         is_batch_faulty_neighbor = batch_flags_this_batch.get(n, True)
#                         current_neighbor_valid_mask = pd.Series(False, index=current_batch_index)

#                         if not is_batch_faulty_neighbor:
#                             neighbor_batch_df = batch_data_cache.get(n)
#                             if neighbor_batch_df is not None and not neighbor_batch_df.empty:
#                                 neighbor_alpha = neighbor_batch_df.get('Alpha').reindex(current_batch_index)
#                                 neighbor_flagged = neighbor_batch_df.get('Flagged').reindex(current_batch_index)
                                
#                                 temp_valid_mask_n = pd.Series(True, index=current_batch_index)
#                                 if neighbor_flagged is not None: temp_valid_mask_n &= ~neighbor_flagged.fillna(True)
#                                 if neighbor_alpha is not None:
#                                     temp_valid_mask_n &= neighbor_alpha.notna()
#                                     temp_valid_mask_n &= (neighbor_alpha > 0)
#                                 else: temp_valid_mask_n = pd.Series(False, index=current_batch_index)
                                
#                                 current_neighbor_valid_mask = temp_valid_mask_n
#                                 if current_neighbor_valid_mask.any():
#                                     valid_neighbor_data[n] = neighbor_alpha.where(current_neighbor_valid_mask)
#                         neighbor_is_valid_mask_df[n] = current_neighbor_valid_mask

#                     if not neighbor_is_valid_mask_df.empty:
#                         num_valid_neighbors_series = neighbor_is_valid_mask_df.sum(axis=1)

#                     if valid_neighbor_data:
#                         df_valid_neighbor_alphas = pd.DataFrame(valid_neighbor_data)
#                         local_factor_series = df_valid_neighbor_alphas.median(axis=1, skipna=True).fillna(1.0).clip(lower=0.1)
#                     else: # Ensure local_factor_series is defined even if no valid_neighbor_data
#                         local_factor_series = pd.Series(1.0, index=current_batch_index)


#                     conditions = [num_valid_neighbors_series == 1, num_valid_neighbors_series == 2, num_valid_neighbors_series >= 3]
#                     choices = [1.0 / 3.0, 2.0 / 3.0, 1.0]
#                     weight_series = pd.Series(np.select(conditions, choices, default=0.0), index=current_batch_index)
                    
#                     f_reg_series = pd.Series(f_reg, index=current_batch_index)
#                     factor_combined_series = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)

#                     if 'Radar_Data_mm_per_min' in df_target.columns:
#                         raw_radar_series = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'].fillna(0)
#                         final_adjusted_batch_series = raw_radar_series * factor_combined_series
#                     else:
#                         final_adjusted_batch_series = pd.Series(np.nan, index=current_batch_index)
                    
#                     df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series.values
#             logger.info("--- Finished Batch Processing Loop ---")
        
#         # ---------------------------------------------------------------------
#         # 7. Post-processing: Calculate Rolling Error Metrics (Outside Batch Loop)
#         #    This will run even if no batch_periods were generated.
#         # ---------------------------------------------------------------------
#         print("Calculating final rolling error metrics...")
#         logger.info("--- Calculating Final Rolling Error Metrics ---")
#         for coord, df_orig in tqdm(all_data.items(), desc="Calculating Rolling Errors"):
#             df = df_orig.copy()
#             if df.empty: 
#                 logger.warning(f"DataFrame for {coord} is empty. Skipping rolling error calculation.")
#                 all_data[coord] = df # Keep it empty
#                 continue

#             if 'Network_Adjusted_Radar' in df.columns and 'Gauge_Data_mm_per_min' in df.columns:
#                 gauge_filled = df['Gauge_Data_mm_per_min'].ffill().bfill() 
#                 net_adj_filled = df['Network_Adjusted_Radar'].ffill().bfill()

#                 if not net_adj_filled.isnull().all() and not gauge_filled.isnull().all():
#                     abs_error = abs(net_adj_filled - gauge_filled)
#                     df['Rolling_Abs_Error'] = abs_error.rolling(window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1).mean()
                    
#                     if 'Flagged' in df.columns:
#                         flag_points_numeric = df['Flagged'].astype(float) 
#                         df['Rolling_Prop_Flagged'] = flag_points_numeric.rolling(window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1).mean()
#                     else:
#                         df['Rolling_Prop_Flagged'] = np.nan
#                         logger.warning(f"'Flagged' column missing for {coord} when calculating Rolling_Prop_Flagged.")
#                 else:
#                     df['Rolling_Abs_Error'] = np.nan
#                     df['Rolling_Prop_Flagged'] = np.nan
#                     logger.info(f"Insufficient data for rolling error metrics for {coord} (all NaNs in key series).")
#             else:
#                 df['Rolling_Abs_Error'] = np.nan
#                 df['Rolling_Prop_Flagged'] = np.nan
#                 logger.warning(f"Missing 'Network_Adjusted_Radar' or 'Gauge_Data_mm_per_min' for {coord}. Cannot calculate rolling error metrics.")
#             all_data[coord] = df
#         logger.info("--- Finished Calculating Final Rolling Error Metrics ---")

#         # --- 8. Save Detailed Output CSV per Coordinate ---
#         print("Saving detailed output CSV files...")
#         logger.info("--- Saving Detailed Output CSV Files ---")
#         output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output" 
#         output_csv_dir.mkdir(parents=True, exist_ok=True)

#         cols_to_save = [
#             'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Batch_Alpha_Adjusted_Radar',
#             'Radar_Freg_Adjusted', 'Final_Adjusted_Rainfall', 'Flagged', 'Batch_Flag'
#         ]
        
#         for coord, df_final in tqdm(all_data.items(), desc="Saving Output CSVs"):
#             if df_final.empty:
#                 logger.warning(f"DataFrame for {coord} is empty. Skipping CSV save.")
#                 continue

#             coord_str = str(coord)
#             save_cols_present = [col for col in cols_to_save if col in df_final.columns]
#             if not save_cols_present:
#                 logger.warning(f"No columns to save for {coord_str}.")
#                 continue

#             df_to_save = df_final[save_cols_present].copy()
            
#             channel_desc_value = "Unknown_Channel"
#             try:
#                 utm_tuple = coordinate_locations_utm.get(coord_str) 
#                 if utm_tuple:
#                     channel_desc_value = utm_to_channel_desc_map.get(utm_tuple, "Description_Not_Found")
#                 else:
#                     logger.warning(f"Could not find UTM tuple for coord '{coord_str}' in coordinate_locations_utm map.")
#                     channel_desc_value = "UTM_Coord_Not_Found"
#             except Exception as e:
#                 logger.error(f"Error looking up channel description for {coord_str}: {e}")
#                 channel_desc_value = "Lookup_Error"
            
#             df_to_save['Channel'] = str(channel_desc_value)
            
#             safe_coord_fname = coord_str.replace('(','').replace(')','').replace(',','_').replace(' ','')
#             output_filename = output_csv_dir / f"{safe_coord_fname}_data_adjustments.csv"

#             try:
#                 first_cols = ['Channel']
#                 remaining_cols = [col for col in df_to_save.columns if col not in first_cols]
#                 final_cols_order = first_cols + remaining_cols
#                 df_to_save[final_cols_order].to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z') 
#                 logger.info(f"Saved detailed data for {coord_str} to {output_filename}")
#             except Exception as e:
#                 logger.error(f"Failed to save CSV for {coord_str}: {e}")
        
#         logger.info(f"--- Finished Saving Detailed Output CSV Files to {output_csv_dir} ---")

#         # ---------------------------------------------------------------------
#         # 9. Final Aggregation, Saving, and Plotting
#         # ---------------------------------------------------------------------
#         events_df = None 
#         all_data_iter0 = None 
#         print("Generating final visualizations...")
#         logger.info("--- Generating Final Visualizations ---")

#         coord_str_to_channel_desc = {}
#         if 'valid_coordinate_locations_utm' in locals() and 'utm_to_channel_desc_map' in locals():
#             for str_coord_key, utm_tuple_val in valid_coordinate_locations_utm.items():
#                  desc = utm_to_channel_desc_map.get(utm_tuple_val, "Unknown Channel")
#                  coord_str_to_channel_desc[str_coord_key] = desc
#         else:
#             logger.warning("Could not create coordinate to channel map for plotting - required data missing.")

#         logger.debug(f"Coordinate to channel description mapping for plots: {coord_str_to_channel_desc}")
        
#         all_data_plotting = {str(k): v for k, v in all_data.items() if not v.empty} # Plot only non-empty
        
#         if all_data_plotting: # Only plot if there's data
#             create_plots_with_error_markers(all_data_plotting, valid_coordinate_locations_utm, coord_str_to_channel_desc,
#                                             events_df=events_df, all_data_iter0=all_data_iter0)
            
#             if gdf_wgs84 is not None and not gdf_wgs84.empty:
#                 generate_html_dashboard(all_data_plotting, valid_coordinate_locations_utm, gdf_wgs84,
#                                         svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
#             else:
#                 logger.warning("gdf_wgs84 is empty or None. Skipping HTML dashboard generation.")
#         else:
#             logger.warning("No data available for plotting. Skipping visualization generation.")
            
#         logger.info("--- Finished Generating Final Visualizations ---")
#         return True
    
#     except Exception as e:
#         logger.error("--- An error occurred during batch adjustment execution ---")
#         logger.error(traceback.format_exc()) 
#         return False
    
#     finally:
#         logger.info("\n--- Batch adjustment process finished for this cycle. ---")

# # =============================================================================
# # Script Entry Point
# # =============================================================================
# if __name__ == "__main__":
#     RUN_INTERVAL_SECONDS = 24 * 60 * 60  # Default: every 24 hours
#     # RUN_INTERVAL_SECONDS = 10 * 60     # For quicker testing: every 10 minutes
#     # RUN_INTERVAL_SECONDS = 60          # For very quick testing: every 1 minute

#     logger.info("===== Live adjustment pipeline started =====")
#     logger.info("This script assumes external data ingestion scripts (gauge/radar) run periodically to update source files.")
#     logger.info(f"Processing will occur approximately every {RUN_INTERVAL_SECONDS / 3600:.1f} hours.")

#     while True:
#         cycle_start_time = datetime.datetime.now()
#         logger.info(f"===== New Live Update Cycle Triggered at {cycle_start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        
#         cycle_completed_successfully = run_daily_adjustment_cycle()

#         cycle_end_time = datetime.datetime.now()
#         if cycle_completed_successfully:
#             logger.info(f"Daily adjustment cycle completed successfully at {cycle_end_time.strftime('%Y-%m-%d %H:%M:%S')}.")
#         else:
#             logger.error(f"Daily adjustment cycle FAILED at {cycle_end_time.strftime('%Y-%m-%d %H:%M:%S')}. Check logs for errors.")
        
#         elapsed_time = (cycle_end_time - cycle_start_time).total_seconds()
#         sleep_duration = max(0, RUN_INTERVAL_SECONDS - elapsed_time)

#         logger.info(f"Cycle duration: {elapsed_time:.2f} seconds. Sleeping for {sleep_duration:.2f} seconds until the next cycle...")
#         try:
#             time.sleep(sleep_duration)
#         except KeyboardInterrupt:
#             logger.info("KeyboardInterrupt received. Exiting live processing loop.")
#             break
#         except Exception as e_slp:
#             logger.error(f"An error occurred during sleep: {e_slp}. Retrying after a shorter delay (1 hour).")
#             time.sleep(3600) 
    
#     logger.info("===== Live adjustment pipeline shutting down. =====")
#     logging.shutdown()

import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging
import datetime
import time  # For sleeping
from pathlib import Path # For path management
import json  # For state file
import zoneinfo # For potential timezone operations if needed elsewhere
import sys

# Custom module imports
from config import CFG # Ensure CFG.LOG_FILE is defined
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
from anomaly import flag_anomalies
from plotting3_batches2 import create_plots_with_error_markers, generate_html_dashboard
from network2 import get_nearest_neighbors
from batch_adjustment import compute_regional_adjustment

# --- Logger Setup ---
# Ensure CFG.LOG_FILE is defined in your config.py
# Example: LOG_FILE = BASE_DIR / "logs" / "main_live_pipeline_v3.log"
log_file_path = getattr(CFG, 'LOG_FILE', Path("main_live_pipeline_v3.log"))
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Also print to console
    ],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MAIN_LIVE_V3") # Specific logger name
logger.setLevel(logging.INFO) # Ensure this logger also respects the level


# --- State File and Constants for Live Operation ---
STATE_FILE_PATH = CFG.BASE_DIR / "live_processing_state_main_v3.json" 
DEFAULT_HISTORICAL_START_STR = "2024-01-01 00:00:00" # Adjust as needed

# =============================================================================
# Helper Function Definition (initialize_adjustment_columns)
# =============================================================================
def initialize_adjustment_columns(df):
    cols_to_init = {
        'Alpha': float, 'Rolling_Gauge': float, 'Rolling_Radar': float,
        'Median_Neighbor_Alpha': float, 'Network_Adjusted_Radar': float,
        'Adjusted_Diff_from_network': float, 'Adjusted_Ratio_From_Network': float,
        'Flagged': bool, 'Batch_Flag': bool, 'Radar_Freg_Adjusted': float,
        'Final_Adjusted_Rainfall': float, 'Rolling_Abs_Error': float,
        'Rolling_Prop_Flagged': float, 'Batch_Alpha_Adjusted_Radar': float
    }
    for col, dtype in cols_to_init.items():
        if col not in df.columns:
            if dtype == bool: df[col] = pd.Series(False, index=df.index, dtype=dtype)
            else: df[col] = pd.Series(np.nan, index=df.index, dtype=dtype)
    return df

# The shift_radar_to_utc function provided in the input is NOT used in this recommended live strategy,
# as we aim for both get_gauge_data and get_radar_data to provide TRUE UTC.
# If a display shift is needed for radar, it's better done in plotting or via a specific
# conditional shift if intermediate "local numbers as UTC" are truly required.

# --- State Management Functions ---
def load_last_processing_end_date():
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, 'r') as f: state = json.load(f)
            last_end_str = state.get("last_successful_processing_end_date_utc")
            if last_end_str:
                dt_obj = pd.Timestamp(last_end_str)
                return dt_obj.tz_localize('UTC') if dt_obj.tzinfo is None else dt_obj.tz_convert('UTC')
            logger.info("No 'last_successful_processing_end_date_utc' in state. Using default.")
        except Exception as e: logger.error(f"Error loading state: {e}. Using default.")
    else: logger.info(f"State file {STATE_FILE_PATH} not found. Assuming first run.")
    default_start_dt = pd.Timestamp(DEFAULT_HISTORICAL_START_STR)
    return default_start_dt.tz_localize('UTC') if default_start_dt.tzinfo is None else default_start_dt.tz_convert('UTC')

# --- State Management Functions ---
def load_last_actual_data_timestamp(): # Renamed for clarity
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                state = json.load(f)
            # Load the actual max timestamp of data that was processed
            last_actual_max_ts_str = state.get("last_actual_max_data_timestamp_utc")
            if last_actual_max_ts_str:
                dt_obj = pd.Timestamp(last_actual_max_ts_str)
                return dt_obj.tz_localize('UTC') if dt_obj.tzinfo is None else dt_obj.tz_convert('UTC')
            logger.info("No 'last_actual_max_data_timestamp_utc' in state. Using default.")
        except Exception as e:
            logger.error(f"Error loading state: {e}. Using default.")
    else:
        logger.info(f"State file {STATE_FILE_PATH} not found. Assuming first run.")
    
    default_start_dt = pd.Timestamp(DEFAULT_HISTORICAL_START_STR)
    return default_start_dt.tz_localize('UTC') if default_start_dt.tzinfo is None else default_start_dt.tz_convert('UTC')

def save_last_actual_data_timestamp(actual_max_timestamp_utc: pd.Timestamp): # Renamed for clarity
    if pd.isna(actual_max_timestamp_utc):
        logger.warning("Attempted to save NaT as last actual data timestamp. Skipping state save.")
        return
    try:
        actual_max_timestamp_utc_aware = actual_max_timestamp_utc
        if actual_max_timestamp_utc.tzinfo is None: 
            actual_max_timestamp_utc_aware = actual_max_timestamp_utc.tz_localize('UTC')
        elif str(actual_max_timestamp_utc.tzinfo).upper() != 'UTC': 
            actual_max_timestamp_utc_aware = actual_max_timestamp_utc.tz_convert('UTC')
        
        state = {"last_actual_max_data_timestamp_utc": actual_max_timestamp_utc_aware.isoformat()}
        STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Saved last actual max data timestamp: {actual_max_timestamp_utc_aware.isoformat()}")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def save_last_processing_end_date(end_date_utc: pd.Timestamp):
    try:
        end_date_utc_aware = end_date_utc
        if end_date_utc.tzinfo is None: end_date_utc_aware = end_date_utc.tz_localize('UTC')
        elif str(end_date_utc.tzinfo).upper() != 'UTC': end_date_utc_aware = end_date_utc.tz_convert('UTC')
        state = {"last_successful_processing_end_date_utc": end_date_utc_aware.isoformat()}
        STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE_PATH, 'w') as f: json.dump(state, f, indent=4)
        logger.info(f"Saved last successful processing end date: {end_date_utc_aware.isoformat()}")
    except Exception as e: logger.error(f"Error saving state: {e}")


def load_full_historical_data_from_pkl(
    list_of_coords_to_plot: list, # List of coordinate tuples or stringified tuples
    pkl_data_directory: Path,     # Directory where PKL files are stored (e.g., CFG.ADJUSTED_DATA_PKL_DIR)
    pkl_filename_pattern: str = "all_data_({x},{y}).pkl" # Default pattern based on your save_as_pkl.py
    ) -> dict:
    """
    Loads the full historical data for each specified coordinate from PKL files.
    Ensures the DataFrame index is a timezone-aware UTC DatetimeIndex.

    Args:
        list_of_coords_to_plot (list): List of coordinates.
                                       Can be tuples (x,y) or stringified tuples like "(x,y)".
        pkl_data_directory (Path): The Path object pointing to the directory containing PKL files.
        pkl_filename_pattern (str): The filename pattern for PKL files.
                                    Use {x} and {y} as placeholders.

    Returns:
        dict: A dictionary where keys are stringified coordinate tuples
              and values are pandas DataFrames with the full historical data.
    """
    full_plot_data = {}
    logger.info(f"Loading full historical data from PKL files in: {pkl_data_directory}")

    if not pkl_data_directory.is_dir():
        logger.error(f"PKL data directory does not exist or is not a directory: {pkl_data_directory}")
        return full_plot_data

    for coord_item in tqdm(list_of_coords_to_plot, desc="Loading PKL data for plots"):
        # Standardize coordinate to string and extract x, y
        if isinstance(coord_item, tuple) and len(coord_item) == 2:
            coord_str = str(coord_item) # e.g., "(661433, 6131423)"
            x, y = coord_item
        elif isinstance(coord_item, str):
            coord_str = coord_item
            try:
                # Attempt to parse string like "(x,y)" or "x,y"
                parsed_coords = coord_item.strip("()").split(',')
                if len(parsed_coords) == 2:
                    x = int(parsed_coords[0].strip())
                    y = int(parsed_coords[1].strip())
                else:
                    logger.warning(f"Could not parse coordinate string '{coord_item}'. Skipping.")
                    continue
            except ValueError:
                logger.warning(f"Could not parse coordinate string '{coord_item}' into integers. Skipping.")
                continue
        else:
            logger.warning(f"Invalid coordinate item type '{type(coord_item)}': {coord_item}. Skipping.")
            continue

        # Construct PKL filename using the provided pattern
        try:
            current_pkl_filename = pkl_filename_pattern.format(x=x, y=y)
        except KeyError:
            logger.error(f"Invalid pkl_filename_pattern '{pkl_filename_pattern}'. Must contain {{x}} and {{y}}. Skipping {coord_str}.")
            continue
            
        pkl_filepath = pkl_data_directory / current_pkl_filename

        if pkl_filepath.exists():
            try:
                df = pd.read_pickle(pkl_filepath)
                if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                    # Ensure timezone consistency: Make it UTC aware for plotting
                    if df.index.tz is None:
                        # If naive, assume it's UTC (or the timezone it was saved as)
                        # Best practice is that save_as_pkl.py ensures it's UTC naive before saving
                        # or saves with UTC timezone info if pickling aware objects.
                        # For this function, we'll assume naive means UTC.
                        df.index = df.index.tz_localize('UTC')
                        logger.debug(f"Localized naive DatetimeIndex to UTC for {pkl_filepath}")
                    elif df.index.tz != datetime.timezone.utc:
                        df.index = df.index.tz_convert('UTC')
                        logger.debug(f"Converted DatetimeIndex from {df.index.tz} to UTC for {pkl_filepath}")
                    
                    full_plot_data[coord_str] = df
                    logger.debug(f"Loaded {len(df)} points from PKL '{current_pkl_filename}' for {coord_str}.")
                elif df.empty:
                    logger.warning(f"PKL file '{current_pkl_filename}' for {coord_str} is empty.")
                else: # Not a DatetimeIndex
                    logger.warning(f"Data in PKL file '{current_pkl_filename}' for {coord_str} does not have a DatetimeIndex.")
            except Exception as e:
                logger.error(f"Error loading or processing PKL file '{current_pkl_filename}' for {coord_str}: {e}", exc_info=True)
        else:
            logger.warning(f"PKL file '{current_pkl_filename}' not found for {coord_str} at {pkl_filepath}.")
            
    logger.info(f"Finished loading PKL data. Loaded data for {len(full_plot_data)} coordinates.")
    return full_plot_data

def load_and_initialize_master_data(
    list_of_target_coords: list, # list of stringified coord tuples "(x,y)"
    pkl_dir: Path,
    pkl_pattern: str,
    combined_csv_dir: Path, # e.g., CFG.COMBINED_DATA_DIR
    combined_csv_pattern: str = "combined_data_({x},{y}).csv"
    ) -> dict:
    """
    Loads data for target coordinates.
    1. Tries to load from PKL. If PKL exists, it's assumed to be the most up-to-date.
    2. If PKL doesn't exist, loads from combined CSV and initializes.
    3. Ensures essential columns for processing and plotting are present.
    """
    master_data_dict = {}
    logger.info(f"Initializing master data for {len(list_of_target_coords)} coordinates...")

    for coord_str in tqdm(list_of_target_coords, desc="Loading/Initializing Master Data"):
        try:
            parsed_coords = coord_str.strip("()").split(',')
            x = int(parsed_coords[0].strip())
            y = int(parsed_coords[1].strip())
        except Exception:
            logger.warning(f"Could not parse coord_str '{coord_str}'. Skipping.")
            continue

        df = None
        pkl_filename = pkl_pattern.format(x=x, y=y)
        pkl_filepath = pkl_dir / pkl_filename

        if pkl_filepath.exists():
            try:
                df = pd.read_pickle(pkl_filepath)
                logger.debug(f"Loaded existing PKL for {coord_str} from {pkl_filepath}")
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Data for {coord_str} from PKL {pkl_filepath} does not have DatetimeIndex. Attempting to fix.")
                    # Attempt to reset index if 'time' column exists from old format
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'], errors='coerce')
                        df = df.set_index('time').sort_index()
                    else: # Cannot fix
                        logger.error(f"Cannot fix index for {coord_str} from PKL {pkl_filepath}. Data might be unusable.")
                        df = pd.DataFrame() # Make it empty to signify issue
            except Exception as e:
                logger.error(f"Error loading PKL {pkl_filepath} for {coord_str}: {e}. Will try combined CSV.")
                df = None
        
        if df is None or df.empty: # If PKL didn't exist, was empty, or failed to load properly
            logger.info(f"PKL for {coord_str} not found or empty/invalid. Loading from combined CSV.")
            combined_filename = combined_csv_pattern.format(x=x, y=y)
            combined_filepath = combined_csv_dir / combined_filename
            if combined_filepath.exists():
                try:
                    df_combined = pd.read_csv(combined_filepath)
                    if 'time' in df_combined.columns and 'Radar Data' in df_combined.columns and 'Gauge Data' in df_combined.columns:
                        df_combined['time'] = pd.to_datetime(df_combined['time'], errors='coerce')
                        df_combined.dropna(subset=['time'], inplace=True)
                        df_combined = df_combined.set_index('time').sort_index()
                        
                        # This is the DataFrame as per your current PKL content
                        df = df_combined[['Radar Data', 'Gauge Data']].copy()
                        logger.debug(f"Loaded and indexed combined CSV for {coord_str}")
                    else:
                        logger.warning(f"Combined CSV {combined_filepath} for {coord_str} missing required columns (time, Radar Data, Gauge Data).")
                        df = pd.DataFrame() # Create empty df
                except Exception as e:
                    logger.error(f"Error loading combined CSV {combined_filepath} for {coord_str}: {e}")
                    df = pd.DataFrame() # Create empty df
            else:
                logger.warning(f"Combined CSV {combined_filepath} not found for {coord_str}.")
                df = pd.DataFrame() # Create empty df

        if df.empty:
            logger.warning(f"No data loaded for {coord_str}. It will be skipped in processing.")
            master_data_dict[coord_str] = pd.DataFrame() # Store empty DF
            continue

        # --- Ensure all required columns for processing & plotting exist ---
        # These are the raw values in mm (if not already mm/min)
        # We need to define how to get to mm/min
        # Assuming 'Radar Data' and 'Gauge Data' are in mm per original interval (e.g., per minute if data is 1-min)
        # If they need conversion (e.g. from cumulative or different units/time steps), that logic goes here.
        # For simplicity, let's assume they are already effectively mm/min for now, or this conversion is simple.
        
        if 'Radar_Data_mm_per_min' not in df.columns:
            if 'Radar Data' in df.columns: # Assuming 'Radar Data' is the source for mm/min
                df['Radar_Data_mm_per_min'] = df['Radar Data'] # Placeholder: Replace with actual conversion if needed
            else:
                df['Radar_Data_mm_per_min'] = pd.Series(index=df.index, dtype='float64') # Empty series

        if 'Gauge_Data_mm_per_min' not in df.columns:
            if 'Gauge Data' in df.columns: # Assuming 'Gauge Data' is the source for mm/min
                df['Gauge_Data_mm_per_min'] = df['Gauge Data'] # Placeholder: Replace with actual conversion if needed
            else:
                df['Gauge_Data_mm_per_min'] = pd.Series(index=df.index, dtype='float64') # Empty series

        # Initialize other adjustment/flag columns if they don't exist
        # These will be populated by your adjustment pipeline
        adjustment_cols = [
            'Radar_Freg_Adjusted', 'Batch_Alpha_Adjusted_Radar',
            'Final_Adjusted_Rainfall', 'Flagged', 'Batch_Flag'
            # Add any other intermediate columns your plotting might reference from df_final
        ]
        for col in adjustment_cols:
            if col not in df.columns:
                df[col] = pd.Series(index=df.index, dtype='float64') # Initialize with NaNs or appropriate dtype
        
        if 'Flagged' not in df.columns: df['Flagged'] = pd.Series(index=df.index, dtype='bool')
        if 'Batch_Flag' not in df.columns: df['Batch_Flag'] = pd.Series(index=df.index, dtype='bool')

        # Ensure DatetimeIndex is UTC for internal processing consistency
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            elif df.index.tz != datetime.timezone.utc:
                df.index = df.index.tz_convert('UTC')
        
        master_data_dict[coord_str] = df

    logger.info(f"Master data initialization complete. Loaded/initialized data for {len(master_data_dict)} coordinates.")
    return master_data_dict

# =============================================================================
# Core Processing Cycle Function
# =============================================================================
def run_daily_adjustment_cycle(processing_start_utc: pd.Timestamp, 
                                 processing_end_for_this_cycle: pd.Timestamp):
    logger.info(f"--- Starting Adjustment Cycle for Window ---")
    logger.info(f"Processing data FROM (inclusive): {processing_start_utc} UP TO (exclusive): {processing_end_for_this_cycle}")

    data_was_actually_processed_in_this_window = False
    all_data_for_batches = {} # Holds data for the current window after preprocessing

    try:
        # ---------------------------------------------------------------------
        # 1. Load Data (Assumes load_time_series_data loads all relevant history)
        # ---------------------------------------------------------------------
        logger.info("--- Loading Initial Data (full history) ---")
        target_coords, coordinate_locations_utm_all = load_target_coordinates()
        sensordata, gdf_wgs84_all = load_sensor_data()
        utm_to_channel_desc_map_all, svk_coords_utm_all = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm_all, _ = load_time_series_data(
            target_coords, coordinate_locations_utm_all
        )
        if not initial_all_data: 
            logger.warning("No initial data loaded by load_time_series_data. No data to process in this cycle.")
            return True, False # Cycle "succeeded" (no error), but no data processed
        
        all_data_loaded_full_history = {str(k): v for k, v in initial_all_data.items()}
        
        # --- Filter data to the current processing window ---
        # logger.info(f"Filtering loaded data to window: {processing_start_utc} <= time < {processing_end_for_this_cycle}")
        # all_data_current_window = {}
        # for coord, df_full in all_data_loaded_full_history.items():
        #     if not isinstance(df_full, pd.DataFrame) or df_full.empty: continue
        #     df_indexed_utc = df_full.copy()
        #     try:
        #         if not isinstance(df_indexed_utc.index, pd.DatetimeIndex):
        #             df_indexed_utc.index = pd.to_datetime(df_indexed_utc.index, errors='coerce')
        #             df_indexed_utc.dropna(subset=[df_indexed_utc.index.name or 'index'], inplace=True)
        #         if df_indexed_utc.empty: continue

        #         if df_indexed_utc.index.tz is None:
        #             df_indexed_utc.index = df_indexed_utc.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
        #         elif str(df_indexed_utc.index.tz).upper() != 'UTC': 
        #             df_indexed_utc.index = df_indexed_utc.index.tz_convert('UTC')
        #         df_indexed_utc.dropna(subset=[df_indexed_utc.index.name or 'index'], inplace=True)
        #         if df_indexed_utc.empty: continue
                
        #         mask = (df_indexed_utc.index >= processing_start_utc) & (df_indexed_utc.index < processing_end_for_this_cycle)
        #         if mask.any():
        #             all_data_current_window[coord] = df_indexed_utc.loc[mask].copy()
        #     except Exception as e_filter: logger.warning(f"Could not filter {coord} for window: {e_filter}")
        # Inside run_daily_adjustment_cycle()

        # ... (previous code for loading initial_all_data) ...
        
        # --- Filter data to the current processing window ---
        logger.info(f"Filtering loaded data to window: {processing_start_utc} <= time < {processing_end_for_this_cycle}")
        all_data_current_window = {}
        for coord, df_full_orig in all_data_loaded_full_history.items():
            if not isinstance(df_full_orig, pd.DataFrame) or df_full_orig.empty:
                logger.debug(f"Skipping {coord}: Original DataFrame is not valid or empty.")
                continue
            
            df_to_filter = df_full_orig.copy()

            try:
                # --- Step 1: Ensure 'datetime' column is the index and is pd.DatetimeIndex ---
                # This section assumes your time column in CSVs/PKLs is named 'datetime'
                # If it's 'time', change 'datetime' to 'time' below.
                time_column_name = None
                if 'datetime' in df_to_filter.columns:
                    time_column_name = 'datetime'
                elif 'time' in df_to_filter.columns: # Fallback if 'time' is the column name
                    time_column_name = 'time'

                if time_column_name:
                    logger.debug(f"Found time column '{time_column_name}' for {coord}. Converting and setting as index.")
                    df_to_filter[time_column_name] = pd.to_datetime(df_to_filter[time_column_name], errors='coerce')
                    df_to_filter.dropna(subset=[time_column_name], inplace=True) # Drop rows where time is NaT
                    if df_to_filter.empty:
                        logger.debug(f"DataFrame for {coord} empty after converting '{time_column_name}' column and dropping NaTs. Skipping.")
                        continue
                    df_to_filter.set_index(time_column_name, inplace=True)
                elif not isinstance(df_to_filter.index, pd.DatetimeIndex):
                    # If no 'datetime' or 'time' column, try to convert the existing index
                    logger.debug(f"Index for {coord} is not DatetimeIndex and no 'datetime'/'time' column. Converting current index.")
                    df_to_filter.index = pd.to_datetime(df_to_filter.index, errors='coerce')
                    # Drop NaTs directly from index if conversion failed for some
                    if df_to_filter.index.hasnans:
                        df_to_filter = df_to_filter[df_to_filter.index.notna()]
                
                if df_to_filter.empty or not isinstance(df_to_filter.index, pd.DatetimeIndex):
                    logger.warning(f"Could not establish a valid DatetimeIndex for {coord}. Skipping.")
                    continue
                
                # --- Step 2: Ensure index is UTC-aware ---
                if df_to_filter.index.tz is None:
                    df_to_filter.index = df_to_filter.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                elif str(df_to_filter.index.tz).upper() != 'UTC': 
                    df_to_filter.index = df_to_filter.index.tz_convert('UTC')
                
                # Drop NaTs again that might have been introduced by tz_localize specifically for ambiguous/nonexistent
                if df_to_filter.index.hasnans:
                    df_to_filter = df_to_filter[df_to_filter.index.notna()]

                if df_to_filter.empty: 
                    logger.debug(f"DataFrame for {coord} empty after timezone handling for filtering. Skipping.")
                    continue
                
                # --- Step 3: Apply the window mask ---
                mask = (df_to_filter.index >= processing_start_utc) & (df_to_filter.index < processing_end_for_this_cycle)
                if mask.any():
                    all_data_current_window[coord] = df_to_filter.loc[mask].copy()
                else:
                    logger.debug(f"No data for {coord} within the current processing window.")

            except Exception as e_filter_window:
                logger.warning(f"MAJOR ERROR during filtering/indexing setup for {coord}: {e_filter_window}")
                logger.warning(traceback.format_exc()) # Log the full traceback
        # ... rest of run_daily_adjustment_cycle ...

        if not all_data_current_window:
            logger.info("No data found within the current processing window after filtering.")
            return True, False 

        data_was_actually_processed_in_this_window = True
        logger.info(f"Found {len(all_data_current_window)} sensors with data in the current processing window.")
        valid_coords_in_window = list(all_data_current_window.keys())
        coordinate_locations_utm_window = {k: v for k, v in coordinate_locations_utm_all.items() if k in valid_coords_in_window}
        valid_coordinate_locations_utm_window = {k: v for k, v in valid_coordinate_locations_utm_all.items() if k in valid_coords_in_window}

        # ---------------------------------------------------------------------
        # 2. Preprocessing (Applied to `all_data_current_window`)
        # ---------------------------------------------------------------------
        logger.info("--- Preprocessing Windowed Data ---")
        for coord, df_orig_windowed in tqdm(all_data_current_window.items(), desc="Preprocessing Windowed Sensors"):
            df = df_orig_windowed.copy() # df.index is already TRUE UTC

            # Gauge data and Radar data are assumed to be TRUE UTC from their ingestion scripts.
            # No `shift_radar_to_utc` is called here for that reason.
            # If you need radar display to show local numbers on a UTC axis, that's a plotting choice
            # or a separate conditional shift (like Step 2b in previous examples, controlled by CFG flag).
            # For this version, we keep everything TRUE UTC.
            logger.debug(f"Preprocessing {coord}. Index is UTC: {str(df.index.tz).upper() == 'UTC'}. Min: {df.index.min()}, Max: {df.index.max()}")

            rename_map = {}
            if 'Radar Data' in df.columns: rename_map['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns: rename_map['Gauge Data'] = 'Gauge_Data_mm_per_min'
            if rename_map: df.rename(columns=rename_map, inplace=True)

                        # --- *** NEW: ALIGN GAUGE AND RADAR DATA TO COMMON PERIOD *** ---
            if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
                logger.debug(f"Aligning Gauge and Radar data for {coord} based on non-NaN periods.")
                
                # Identify where both gauge and radar have valid (non-NaN) data
                # This assumes 0 in gauge is valid, but NaN in radar might mean no data.
                # If 0 in radar is also valid, adjust accordingly.
                # For this example, let's assume NaN indicates missing data for either.
                gauge_valid_idx = df[df['Gauge_Data_mm_per_min'].notna()].index
                radar_valid_idx = df[df['Radar_Data_mm_per_min'].notna()].index
                
                if gauge_valid_idx.empty or radar_valid_idx.empty:
                    logger.warning(f"Sensor {coord}: Either gauge or radar has no valid data points after NaNs. Skipping alignment, will likely result in no adjustments.")
                    # Keep df as is, feature engineering will produce NaNs where one is missing
                else:
                    common_start = max(gauge_valid_idx.min(), radar_valid_idx.min())
                    common_end = min(gauge_valid_idx.max(), radar_valid_idx.max())
                    
                    if common_start >= common_end:
                        logger.warning(f"Sensor {coord}: No overlapping period with valid data for gauge and radar. Gauge: {gauge_valid_idx.min()}-{gauge_valid_idx.max()}, Radar: {radar_valid_idx.min()}-{radar_valid_idx.max()}. Skipping alignment.")
                        # To effectively process nothing for this sensor's combined features:
                        # df = pd.DataFrame(index=df.index) # Keep index, clear data, or handle as below
                        df = df.loc[common_start:common_end] # This will make it empty
                    else:
                        logger.info(f"Sensor {coord}: Common data period found: {common_start} to {common_end}. Slicing data.")
                        df = df.loc[common_start:common_end].copy() # Slice to the common period
                        # It's possible that even within this common_start/end, some intermediate rows might be all NaN
                        # for one of the columns if the data is sparse. Pandas handles this in rolling operations.
            
            if df.empty:
                logger.warning(f"DataFrame for {coord} became empty after attempting to find common data period. Storing empty.")
                all_data_for_batches[coord] = df # Store the (now possibly empty) df
                continue
            # --- *** END OF ALIGNMENT *** ---
            
            if 'Gauge_Data_mm_per_min' in df.columns: df['Gauge_Data_mm_per_min'] = df['Gauge_Data_mm_per_min'].fillna(0.0)
            
            df = initialize_adjustment_columns(df)
            
            can_do_fe = ('Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns) or \
                        ('Radar_Data_mm_per_min' in df.columns) or \
                        ('Gauge_Data_mm_per_min' in df.columns)
            if can_do_fe and not df.empty:
                df = apply_feature_engineering({coord: df})[coord]
            elif not df.empty:
                logger.warning(f"Skipping feature engineering for {coord}, missing data columns or df empty.")
            
            if not df.empty: all_data_for_batches[coord] = df
            else: logger.warning(f"DataFrame for {coord} became empty during preprocessing.")
        
        if not all_data_for_batches:
            logger.info("No data survived preprocessing for current window.")
            return True, data_was_actually_processed_in_this_window # data_was_processed could still be True

        # ---------------------------------------------------------------------
        # 3. Batch Processing Loop (Operates on `all_data_for_batches`)
        # ---------------------------------------------------------------------
        logger.info("--- Starting Batch Processing for Current Window ---")
        # ... (The rest of your batch processing logic from Step 3 to Step 6
        #      from your original script goes here, ensuring it uses
        #      `all_data_for_batches` and `valid_coordinate_locations_utm_window`)
        # --- Start of Pasted Batch Processing Logic ---
        window_actual_min_time = pd.Timestamp.max.tz_localize('UTC')
        window_actual_max_time = pd.Timestamp.min.tz_localize('UTC')
        found_data_for_batch_range = False
        for df_check_batch in all_data_for_batches.values():
            if not df_check_batch.empty:
                window_actual_min_time = min(window_actual_min_time, df_check_batch.index.min())
                window_actual_max_time = max(window_actual_max_time, df_check_batch.index.max())
                found_data_for_batch_range = True
        
        if not found_data_for_batch_range or pd.isna(window_actual_min_time) or \
           pd.isna(window_actual_max_time) or window_actual_min_time >= window_actual_max_time:
            logger.info("No actual data points to create batches in current window after preprocessing.")
            return True, data_was_actually_processed_in_this_window

        batch_start_times = pd.date_range(
            start=window_actual_min_time.floor(CFG.BATCH_DURATION),
            end=window_actual_max_time, freq=CFG.BATCH_DURATION, tz='UTC'
        )
        batch_periods = []
        for bs in batch_start_times:
            be = min(bs + CFG.BATCH_DURATION, processing_end_for_this_cycle, window_actual_max_time + pd.Timedelta(microseconds=1))
            if bs < be: batch_periods.append((bs, be))
        
        if not batch_periods:
            logger.info("No valid batch periods for current data window.")
            return True, data_was_actually_processed_in_this_window
        logger.info(f"Generated {len(batch_periods)} batch periods for current window.")

        for batch_start, batch_end in tqdm(batch_periods, desc="Processing Window Batches"):
            logger.debug(f"Batch: {batch_start} to {batch_end}")
            batch_avg_alphas = {}
            for coord_loop, df_loop in all_data_for_batches.items():
                batch_mask_loop = (df_loop.index >= batch_start) & (df_loop.index < batch_end)
                if batch_mask_loop.any() and 'Alpha' in df_loop.columns:
                    avg_alpha = df_loop.loc[batch_mask_loop, 'Alpha'].mean(skipna=True)
                    batch_avg_alphas[coord_loop] = avg_alpha if pd.notna(avg_alpha) else 1.0
                else: batch_avg_alphas[coord_loop] = 1.0
            for coord_loop_target, df_target_loop in all_data_for_batches.items(): 
                batch_mask_target = (df_target_loop.index >= batch_start) & (df_target_loop.index < batch_end)
                if not batch_mask_target.any(): continue
                neighbors = get_nearest_neighbors(coord_loop_target, valid_coordinate_locations_utm_window, n_neighbors=CFG.N_NEIGHBORS)
                relevant_coords_alpha = [coord_loop_target] + [n for n in neighbors if n in batch_avg_alphas]
                batch_alpha_values_median = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords_alpha]
                median_batch_avg_alpha = np.nanmedian(batch_alpha_values_median) if batch_alpha_values_median else 1.0
                if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0
                if 'Radar_Data_mm_per_min' in df_target_loop.columns:
                    df_target_loop.loc[batch_mask_target, 'Batch_Alpha_Adjusted_Radar'] = \
                        (df_target_loop.loc[batch_mask_target, 'Radar_Data_mm_per_min'].fillna(0) + CFG.EPSILON) * median_batch_avg_alpha
            for coord_loop_flagprep, df_loop_flagprep in all_data_for_batches.items():
                batch_mask_flagprep = (df_loop_flagprep.index >= batch_start) & (df_loop_flagprep.index < batch_end)
                if not batch_mask_flagprep.any(): continue
                req_cols_flagprep = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                if all(c in df_loop_flagprep.columns for c in req_cols_flagprep) and \
                   not df_loop_flagprep.loc[batch_mask_flagprep, 'Rolling_Gauge'].isnull().all() and \
                   not df_loop_flagprep.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar'].isnull().all():
                    rolling_batch_alpha_adj = df_loop_flagprep.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar']\
                        .rolling(str(CFG.ROLLING_WINDOW), center=True, min_periods=1).mean().ffill().bfill()
                    rolling_gauge_batch = df_loop_flagprep.loc[batch_mask_flagprep, 'Rolling_Gauge']
                    rolling_gauge_aligned, rolling_batch_alpha_adj_aligned = rolling_gauge_batch.align(rolling_batch_alpha_adj, join='inner')
                    if not rolling_gauge_aligned.empty and not rolling_batch_alpha_adj_aligned.empty:
                        diff = rolling_batch_alpha_adj_aligned - rolling_gauge_aligned
                        ratio_raw = (rolling_batch_alpha_adj_aligned + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                        ratio_no_inf = ratio_raw.replace([np.inf, -np.inf], 3.0)
                        ratio_clipped = ratio_no_inf.clip(lower=0.0, upper=3.0).fillna(1.0)
                        df_loop_flagprep.loc[diff.index, 'Adjusted_Diff_from_network'] = diff
                        df_loop_flagprep.loc[ratio_clipped.index, 'Adjusted_Ratio_From_Network'] = ratio_clipped
                    else:
                        original_indices_in_batch_fgp = df_loop_flagprep.loc[batch_mask_flagprep].index
                        df_loop_flagprep.loc[original_indices_in_batch_fgp, 'Adjusted_Diff_from_network'] = np.nan
                        df_loop_flagprep.loc[original_indices_in_batch_fgp, 'Adjusted_Ratio_From_Network'] = np.nan
                else:
                    df_loop_flagprep.loc[batch_mask_flagprep, 'Adjusted_Diff_from_network'] = np.nan
                    df_loop_flagprep.loc[batch_mask_flagprep, 'Adjusted_Ratio_From_Network'] = np.nan
            batch_flags_this_batch = {} 
            for coord_loop_flag, df_loop_flag in all_data_for_batches.items():
                batch_mask_flagging = (df_loop_flag.index >= batch_start) & (df_loop_flag.index < batch_end)
                if not batch_mask_flagging.any():
                    batch_flags_this_batch[coord_loop_flag] = False 
                    df_loop_flag.loc[batch_mask_flagging, 'Batch_Flag'] = False; continue
                df_slice_for_flagging = {coord_loop_flag: df_loop_flag.loc[batch_mask_flagging].copy()}
                if df_slice_for_flagging[coord_loop_flag].empty:
                     df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = False
                     batch_flags_this_batch[coord_loop_flag] = False; continue
                flagged_slice_dict = flag_anomalies(df_slice_for_flagging) 
                if coord_loop_flag in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord_loop_flag].columns:
                    df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = flagged_slice_dict[coord_loop_flag]['Flagged'].reindex(df_loop_flag.loc[batch_mask_flagging].index).fillna(False)
                else: df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = False
                is_batch_faulty = False
                if 'Flagged' in df_loop_flag.columns:
                    df_batch_current_sensor = df_loop_flag.loc[batch_mask_flagging]
                    if not df_batch_current_sensor.empty:
                        flagged_points = df_batch_current_sensor['Flagged'].sum()
                        total_points = len(df_batch_current_sensor)
                        percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                        if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT: is_batch_faulty = True
                batch_flags_this_batch[coord_loop_flag] = is_batch_faulty
                df_loop_flag.loc[batch_mask_flagging, 'Batch_Flag'] = is_batch_faulty
            f_reg, valid_sensor_count_for_freg = compute_regional_adjustment(
                all_data_for_batches, batch_start, batch_end, batch_flags_this_batch)
            logger.debug(f"Batch {batch_start}-{batch_end}: f_reg = {f_reg:.4f} from {valid_sensor_count_for_freg} valid sensors.")
            batch_data_cache_for_final = {}
            for coord_cache, df_cache in all_data_for_batches.items():
                batch_mask_cache = (df_cache.index >= batch_start) & (df_cache.index < batch_end)
                if batch_mask_cache.any():
                    cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                    batch_data_cache_for_final[coord_cache] = df_cache.loc[batch_mask_cache, [c for c in cols_to_cache if c in df_cache.columns]].copy()
            for coord_loop_final_target, df_target_loop_final in all_data_for_batches.items(): 
                batch_mask_final = (df_target_loop_final.index >= batch_start) & (df_target_loop_final.index < batch_end)
                if not batch_mask_final.any(): continue
                if 'Radar_Data_mm_per_min' in df_target_loop_final.columns:
                    df_target_loop_final.loc[batch_mask_final, 'Radar_Freg_Adjusted'] = \
                        df_target_loop_final.loc[batch_mask_final, 'Radar_Data_mm_per_min'].fillna(0) * f_reg
                neighbors_final = get_nearest_neighbors(coord_loop_final_target, valid_coordinate_locations_utm_window, n_neighbors=CFG.N_NEIGHBORS)
                current_batch_index_final = df_target_loop_final.loc[batch_mask_final].index
                if current_batch_index_final.empty: continue
                num_valid_neighbors_series = pd.Series(0, index=current_batch_index_final, dtype=int)
                local_factor_series = pd.Series(1.0, index=current_batch_index_final, dtype=float)
                valid_neighbor_alphas_for_local = {}
                neighbor_is_valid_timestep_mask = pd.DataFrame(index=current_batch_index_final)
                for n_coord in neighbors_final:
                    if n_coord == coord_loop_final_target: continue 
                    if batch_flags_this_batch.get(n_coord, True):
                        neighbor_is_valid_timestep_mask[n_coord] = False; continue
                    neighbor_df_from_cache = batch_data_cache_for_final.get(n_coord)
                    if neighbor_df_from_cache is None or neighbor_df_from_cache.empty:
                        neighbor_is_valid_timestep_mask[n_coord] = False; continue
                    n_alpha = neighbor_df_from_cache.get('Alpha').reindex(current_batch_index_final)
                    n_flagged = neighbor_df_from_cache.get('Flagged').reindex(current_batch_index_final)
                    valid_mask_for_n_timestep = pd.Series(True, index=current_batch_index_final)
                    if n_flagged is not None: valid_mask_for_n_timestep &= ~n_flagged.fillna(True)
                    if n_alpha is not None:
                        valid_mask_for_n_timestep &= n_alpha.notna(); valid_mask_for_n_timestep &= (n_alpha > 0)
                    else: valid_mask_for_n_timestep = pd.Series(False, index=current_batch_index_final)
                    neighbor_is_valid_timestep_mask[n_coord] = valid_mask_for_n_timestep
                    if n_alpha is not None and valid_mask_for_n_timestep.any():
                        valid_neighbor_alphas_for_local[n_coord] = n_alpha
                if not neighbor_is_valid_timestep_mask.empty:
                    num_valid_neighbors_series = neighbor_is_valid_timestep_mask.sum(axis=1).astype(int)
                if valid_neighbor_alphas_for_local:
                    df_valid_n_alphas = pd.DataFrame(valid_neighbor_alphas_for_local)
                    valid_keys_for_mask = [key for key in valid_neighbor_alphas_for_local.keys() if key in neighbor_is_valid_timestep_mask.columns]
                    if valid_keys_for_mask:
                        masked_n_alphas = df_valid_n_alphas.where(neighbor_is_valid_timestep_mask[valid_keys_for_mask])
                        local_factor_series = masked_n_alphas.median(axis=1, skipna=True).fillna(1.0).clip(lower=0.1)
                    else: local_factor_series.fillna(1.0, inplace=True) # Default if no valid keys
                else: local_factor_series.fillna(1.0, inplace=True) # Default if no valid data
                weight_conditions = [num_valid_neighbors_series == 1, num_valid_neighbors_series == 2, num_valid_neighbors_series >= 3]
                weight_choices = [1.0/3.0, 2.0/3.0, 1.0]
                weight_series = pd.Series(np.select(weight_conditions, weight_choices, default=0.0), index=current_batch_index_final)
                f_reg_series = pd.Series(f_reg, index=current_batch_index_final)
                factor_combined = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)
                if 'Radar_Data_mm_per_min' in df_target_loop_final.columns:
                    raw_radar_in_batch = df_target_loop_final.loc[batch_mask_final, 'Radar_Data_mm_per_min'].fillna(0)
                    df_target_loop_final.loc[batch_mask_final, 'Final_Adjusted_Rainfall'] = raw_radar_in_batch * factor_combined
        # --- End Batch Loop for current window---
        
        all_data_final_processed_this_cycle = all_data_for_batches

        # ---------------------------------------------------------------------
        # Save & Plot (Outputs are for the current window)
        # ---------------------------------------------------------------------
        date_str_for_outputs = processing_end_for_this_cycle.strftime('%Y%m%d_%H%M')
        
        logger.info(f"--- Saving Detailed Output CSV Files for window ending {date_str_for_outputs} ---")
        daily_csv_output_dir = Path(CFG.RESULTS_DIR) / f"daily_outputs_{date_str_for_outputs}" 

        daily_csv_output_dir = Path(CFG.RESULTS_DIR) / f"detailed_sensor_output"  ### ATH

        daily_csv_output_dir.mkdir(parents=True, exist_ok=True)
        cols_to_save = [
            'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Alpha', 
            'Batch_Alpha_Adjusted_Radar', 'Adjusted_Diff_from_network', 
            'Adjusted_Ratio_From_Network', 'Flagged', 'Batch_Flag', 
            'Radar_Freg_Adjusted', 'Final_Adjusted_Rainfall'
        ] # Removed Rolling_Abs_Error and Rolling_Prop_Flagged as they were not calculated in your example
        for coord_str_save, df_final_to_save in tqdm(all_data_final_processed_this_cycle.items(), desc="Saving Daily CSVs"):
            df_save_copy = df_final_to_save.copy()
            save_cols_ok = [col for col in cols_to_save if col in df_save_copy.columns]
            if not save_cols_ok: logger.warning(f"No cols for {coord_str_save}. Skip CSV."); continue
            df_csv_out = df_save_copy[save_cols_ok].copy()
            channel_desc = "Unknown"
            utm_tuple_val = coordinate_locations_utm_all.get(str(coord_str_save)) 
            if utm_tuple_val and utm_to_channel_desc_map_all:
                channel_desc = utm_to_channel_desc_map_all.get(utm_tuple_val, "DescNotFound")
            df_csv_out['Channel_Description'] = str(channel_desc)
            safe_coord_fn_part = str(coord_str_save).replace('(','').replace(')','').replace(',','_').replace(' ','')
            out_csv_fn = daily_csv_output_dir / f"{safe_coord_fn_part}_data_adjustments.csv"
            try: 
                df_csv_out = df_csv_out[['Channel_Description'] + save_cols_ok] 
                # Save with UTC timezone info for clarity if index is still tz-aware
                # If index was made naive for some reason, this will save naive
                df_csv_out.to_csv(out_csv_fn, date_format='%Y-%m-%d %H:%M:%S%z') 
            except Exception as e_csv_final: logger.error(f"Fail save {out_csv_fn}: {e_csv_final}")
        logger.info(f"Saved daily CSVs to {daily_csv_output_dir}")
        
        logger.info(f"--- Generating Daily Visualizations for window ending {date_str_for_outputs} ---")
        coord_map_plot_daily = {}
        if valid_coordinate_locations_utm_window and utm_to_channel_desc_map_all:
            for c_str_plot, utm_t_plot_val in valid_coordinate_locations_utm_window.items():
                 coord_map_plot_daily[c_str_plot] = utm_to_channel_desc_map_all.get(utm_t_plot_val, "Unknown")
        
        daily_plot_output_dir_actual = Path(CFG.PLOTS_OUTPUT_DIR) / f"daily_run_{date_str_for_outputs}"
        daily_plot_output_dir_actual.mkdir(parents=True, exist_ok=True)
        
        # Store and restore CFG.PLOTS_OUTPUT_DIR if plotting function doesn't take output_dir
        original_plots_dir_cfg_val = CFG.PLOTS_OUTPUT_DIR if hasattr(CFG, 'PLOTS_OUTPUT_DIR') else None
        CFG.PLOTS_OUTPUT_DIR = daily_plot_output_dir_actual 
        try:

            # coords_file = ALL_COORDS_FILE
            pkl_dir = Path(CFG.PKL_DATA_DIR)
            pkl_pattern = CFG.PKL_FILENAME_PATTERN

            all_data_for_plotting = load_full_historical_data_from_pkl(
                list_of_coords_to_plot=valid_coordinate_locations_utm_window.keys(),
                pkl_data_directory=pkl_dir,
                pkl_filename_pattern=pkl_pattern
            )
            print("ALL DATA FOR PLOTTING")
            print(all_data_for_plotting)
            
            create_plots_with_error_markers(
                all_data_for_plotting, 
                valid_coordinate_locations_utm_window, 
                coord_map_plot_daily, 
                events_df=None, 
                all_data_iter0=None
            )
        except Exception as e_plot:
            logger.error(f"Error during create_plots_with_error_markers: {e_plot}")
            traceback.print_exc()
        finally:
            if original_plots_dir_cfg_val is not None:
                 CFG.PLOTS_OUTPUT_DIR = original_plots_dir_cfg_val 
            elif hasattr(CFG, 'PLOTS_OUTPUT_DIR'): # remove if it was added temporarily
                 del CFG.PLOTS_OUTPUT_DIR
        logger.info(f"Daily plots saved to {daily_plot_output_dir_actual}.")
        
        if gdf_wgs84_all is not None and not gdf_wgs84_all.empty:
            daily_dashboard_filename = Path(CFG.DASHBOARD_DIR) / f"dashboard_adj_{date_str_for_outputs}.html"
            try:
                generate_html_dashboard(
                    all_data_final_processed_this_cycle, 
                    valid_coordinate_locations_utm_window, 
                    gdf_wgs84_all,
                    svk_coords_utm_all if 'svk_coords_utm_all' in locals() else [],
                    output_file=str(daily_dashboard_filename)
                )
                logger.info(f"Generated daily dashboard: {daily_dashboard_filename}")
            except Exception as e_dash:
                logger.error(f"Error during generate_html_dashboard: {e_dash}")
                traceback.print_exc()


        save_last_actual_data_timestamp(processing_end_for_this_cycle) 
        return True, data_was_actually_processed_in_this_window

    except Exception as e_cycle_main_exc:
        logger.critical(f"--- CRITICAL ERROR in daily adjustment cycle (Window: {processing_start_utc} to {processing_end_for_this_cycle}) ---")
        logger.critical(traceback.format_exc())
        return False, data_was_actually_processed_in_this_window # Still return this flag

# =============================================================================
# Script Entry Point for Live Processing
# =============================================================================
# if __name__ == "__main__":
#     RUN_INTERVAL_SECONDS = 24 * 60 * 60  # Default: Once per day
#     # RUN_INTERVAL_SECONDS = 60
    
#     run_interval_from_cfg = getattr(CFG, 'LIVE_RUN_INTERVAL_SECONDS', RUN_INTERVAL_SECONDS)
#     if isinstance(run_interval_from_cfg, (int, float)) and run_interval_from_cfg > 0:
#         RUN_INTERVAL_SECONDS = int(run_interval_from_cfg)
    
#     logger.info("===== Live adjustment pipeline started =====")
#     logger.info("This script assumes external data ingestion scripts update data sources.")
#     logger.info(f"Processing will occur approx. every {RUN_INTERVAL_SECONDS / 3600:.1f} hours, analyzing data up to 1 hour before runtime.")
    
#     radar_shift_config_info = getattr(CFG, 'SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC', "NOT CONFIGURED (defaults to False)")
#     logger.info(f"Radar timestamp shift to 'local numbers as UTC' in main script is currently: {radar_shift_config_info}.")
#     if radar_shift_config_info is True:
#         logger.warning("Radar timestamp shift (SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC) is ENABLED. Ensure input radar data (from PKL) is TRUE UTC.")
#     elif radar_shift_config_info is False:
#         logger.info("Radar timestamps will be processed as TRUE UTC (SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC is False).")
#     else: # Not set or invalid
#         logger.warning("CFG.SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC flag not found or invalid in config.py. Defaulting to NO SHIFT (radar processed as TRUE UTC).")

#     while True:
#         cycle_trigger_time = datetime.datetime.now()
#         logger.info(f"===== New Live Update Cycle Triggered at {cycle_trigger_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
        
#         current_processing_start_utc = load_last_actual_data_timestamp()
#         now_for_window_calc = pd.Timestamp.now(tz='UTC')
#         current_processing_end_for_this_cycle = (now_for_window_calc - pd.Timedelta(hours=1)).floor('min')

#         cycle_status_message = ""
#         processed_new_data_in_run_flag = False 

#         if current_processing_start_utc >= current_processing_end_for_this_cycle:
#             logger.info(f"No new full time window to process. Last processed end: {current_processing_start_utc}, Current potential end: {current_processing_end_for_this_cycle}.")
#             # No actual processing work done, but the check was successful.
#             # State file is NOT updated here; it's updated inside run_daily_adjustment_cycle upon its own success for a valid window.
#         else:
#             # cycle_completed_successfully: True if the function ran without CRITICAL error
#             # processed_new_data_in_run_flag: True if all_data_current_window had data initially
#             cycle_completed_successfully, processed_new_data_in_run_flag = run_daily_adjustment_cycle(
#                 current_processing_start_utc,
#                 current_processing_end_for_this_cycle
#             )

#             if cycle_completed_successfully:
#                 if processed_new_data_in_run_flag:
#                     cycle_status_message = f"processed new data up to {current_processing_end_for_this_cycle}."
#                 else: 
#                     cycle_status_message = f"completed (window up to {current_processing_end_for_this_cycle}) but found no new data to process within this window."
#                 logger.info(f"Adjustment cycle {cycle_status_message} Handled at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
#             else:
#                 logger.error(f"Adjustment cycle for window ending {current_processing_end_for_this_cycle} FAILED at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. State NOT updated for this failed window. Check logs.")
        
#         logger.info(f"Sleeping for {RUN_INTERVAL_SECONDS / 3600:.1f} hours until the next cycle attempt...")
#         print(f"Sleeping for {RUN_INTERVAL_SECONDS / 3600:.1f} hours until the next cycle attempt...")
#         try:
#             time.sleep(RUN_INTERVAL_SECONDS)
#         except KeyboardInterrupt:
#             logger.info("KeyboardInterrupt received. Exiting live processing loop.")
#             break
#         except Exception as e_slp_main:
#             logger.error(f"An error occurred during main sleep: {e_slp_main}. Retrying after a shorter delay (1 hour).")
#             time.sleep(3600) 
    
#     logger.info("===== Live adjustment pipeline shutting down. =====")
#     # logging.shutdown() # Usually not needed explicitly at script end

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Main live adjustment pipeline script.")
    parser.add_argument('--run-once', action='store_true', help='Run the daily adjustment cycle once and exit.')
    args = parser.parse_args()

    # set args to run_once
    args.run_once = True
    target_coords_static, coordinate_locations_utm_all_static = load_target_coordinates()
    sensordata_static, gdf_wgs84_all_static = load_sensor_data()
    utm_to_channel_desc_map_all_static, svk_coords_utm_all_static = process_sensor_metadata(sensordata_static)
    master_sensor_data = load_and_initialize_master_data(
        list_of_target_coords=list(target_coords_static.keys()), # Assuming target_coords keys are the ones to init
        pkl_dir=CFG.MASTER_PKL_DIR, # e.g., Path("all_data_pkl2")
        pkl_pattern=CFG.MASTER_PKL_PATTERN, # e.g., "all_data_({x},{y}).pkl"
        combined_csv_dir=CFG.COMBINED_DATA_DIR # e.g., Path("combined_data")
    )

    if args.run_once:
        logger.info("===== Running a single live adjustment cycle (triggered by --run-once) =====")
        
        current_processing_start_utc = load_last_actual_data_timestamp()
        now_for_window_calc = pd.Timestamp.now(tz='UTC')
        current_processing_end_for_this_cycle = (now_for_window_calc - pd.Timedelta(hours=1)).floor('min')

        if current_processing_start_utc >= current_processing_end_for_this_cycle:
            logger.info(f"No new full time window to process for single run. Last processed end: {current_processing_start_utc}, Current potential end: {current_processing_end_for_this_cycle}.")
            # Exit gracefully if no new window
        else:
            # cycle_completed_successfully, processed_new_data_in_run_flag = run_daily_adjustment_cycle(
            #     current_processing_start_utc,
            #     current_processing_end_for_this_cycle
            # )
            cycle_completed_successfully, processed_any_data_flag, actual_max_ts_processed = run_daily_adjustment_cycle(
                master_sensor_data_dict_input=master_sensor_data, # Pass the master dict
                processing_start_utc=current_processing_start_utc,
                processing_end_for_this_cycle=current_processing_end_for_this_cycle,
                target_coords_static=target_coords_static,
                coordinate_locations_utm_all_static=coordinate_locations_utm_all_static,
                utm_to_channel_desc_map_all_static=utm_to_channel_desc_map_all_static,
                gdf_wgs84_all_static=gdf_wgs84_all_static,
                svk_coords_utm_all_static=svk_coords_utm_all_static
            )

            if cycle_completed_successfully:
                logger.info(f"Single daily adjustment cycle completed successfully for window ending {current_processing_end_for_this_cycle}.")
            else:
                logger.error(f"Single daily adjustment cycle FAILED for window ending {current_processing_end_for_this_cycle}. Check logs for errors.")
                sys.exit(1) # Exit with error code
        
        logger.info("===== Single live adjustment cycle finished. =====")

    else: # Original looping behavior if not --run-once
        RUN_INTERVAL_SECONDS = 24 * 60 * 60 
        RUN_INTERVAL_SECONDS = 1
        run_interval_from_cfg = getattr(CFG, 'LIVE_RUN_INTERVAL_SECONDS', RUN_INTERVAL_SECONDS)
        if isinstance(run_interval_from_cfg, (int, float)) and run_interval_from_cfg > 0:
            RUN_INTERVAL_SECONDS = int(run_interval_from_cfg)
        
        logger.info("===== Live adjustment pipeline started (looping mode) =====")
        # ... (rest of your original while True loop) ...
        while True:
            cycle_trigger_time = datetime.datetime.now()
            logger.info(f"===== New Live Update Cycle Triggered at {cycle_trigger_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            
            current_processing_start_utc = load_last_actual_data_timestamp()
            now_for_window_calc = pd.Timestamp.now(tz='UTC')
            current_processing_end_for_this_cycle = (now_for_window_calc - pd.Timedelta(hours=1)).floor('min')

            if current_processing_start_utc >= current_processing_end_for_this_cycle:
                logger.info(f"No new full time window to process. Last processed end: {current_processing_start_utc}, Current potential end: {current_processing_end_for_this_cycle}.")
            else:
                cycle_completed_successfully, processed_new_data_in_run_flag = run_daily_adjustment_cycle(
                    current_processing_start_utc,
                    current_processing_end_for_this_cycle
                )
                if cycle_completed_successfully:
                    if processed_new_data_in_run_flag:
                        cycle_status_message = f"processed new data up to {current_processing_end_for_this_cycle}."
                    else: 
                        cycle_status_message = f"completed (window up to {current_processing_end_for_this_cycle}) but found no new data to process within this window."
                    logger.info(f"Adjustment cycle {cycle_status_message} Handled at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
                else:
                    logger.error(f"Adjustment cycle for window ending {current_processing_end_for_this_cycle} FAILED at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. State NOT updated for this failed window. Check logs.")
            print(RUN_INTERVAL_SECONDS)
            logger.info(f"Sleeping for {RUN_INTERVAL_SECONDS / 3600:.1f} hours until the next cycle attempt...")
            try:
                time.sleep(RUN_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Exiting live processing loop.")
                break
            except Exception as e_slp_main:
                logger.error(f"An error occurred during main sleep: {e_slp_main}. Retrying after a shorter delay (1 hour).")
                time.sleep(3600) 
        
        logger.info("===== Live adjustment pipeline shutting down. =====")


# --- END OF FILE main_v3_batch3_live.py ---