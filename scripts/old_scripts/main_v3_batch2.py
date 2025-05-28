

import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging
import datetime
# Removed unused imports like network_iterative2, event_detection2 if not needed

from config import CFG
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
from anomaly import flag_anomalies
from results import aggregate_results
from scripts.old_scripts.plotting3_batches import (create_plots_with_error_markers, generate_html_dashboard,
                               create_flagging_plots_dashboard)
from scripts.old_scripts.plotting3_batches import debug_alpha_for_coord, debug_alpha_and_neighbors_plot
from network import get_nearest_neighbors
from batch_adjustment import compute_regional_adjustment

# --- Logging Setup ---
# ... (as before) ...
log_file_main = CFG.RESULTS_DIR / 'batch_adjustment_process.log' # Changed log file name
formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    main_file_handler = logging.FileHandler(log_file_main, mode='w')
    main_file_handler.setFormatter(formatter_main)
    logger.addHandler(main_file_handler)

# --- Helper to initialize columns ---
def initialize_adjustment_columns(df):
    # ... (as before) ...
    cols_to_init = { 'Alpha': float,'Rolling_Gauge': float, 'Rolling_Radar': float, 'Median_Neighbor_Alpha': float, 'Network_Adjusted_Radar': float, 'Adjusted_Diff_from_network': float, 'Adjusted_Ratio_From_Network': float, 'Flagged': bool, 'Batch_Flag': bool, 'Radar_Freg_Adjusted': float, 'Final_Adjusted_Rainfall': float, 'Rolling_Abs_Error': float, 'Rolling_Prop_Flagged': float }
    for col, dtype in cols_to_init.items():
        if col not in df.columns: df[col] = pd.Series(dtype=dtype, index=df.index)
    return df

# --- Main Function ---
def main_batch_adjustment():
    logger.info('Starting BATCH ADJUSTMENT process...')
    all_data = {}

    try:
        # --- 1. Load Data ---
        # ... (as before) ...
        logger.info("--- Loading Initial Data ---")
        print("--- Loading Initial Data ---")
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        if not initial_all_data or not valid_coordinate_locations_utm: return
        all_data = initial_all_data.copy(); logger.info(f"Loaded initial data for {len(all_data)} coordinates.")
        all_data = {str(k): v for k, v in all_data.items()}
        coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
        valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}


        # --- 2. Preprocessing: Rename, Timezone, Initialize, Feature Engineering ---
        print("Preprocessing...")
        processed_data_step2 = {}
        for coord, df_orig in tqdm(all_data.items(), desc="Preprocessing"):
            # ... (Preprocessing logic including Timezone Standardization as in previous correction) ...
            df = df_orig.copy()
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    if df.index.tz is None: df.index = df.index.tz_localize('UTC')
                    elif df.index.tz != datetime.timezone.utc: df.index = df.index.tz_convert('UTC')
                except Exception as e: logger.warning(f"TZ error {coord}: {e}"); processed_data_step2[coord] = df_orig; continue
            else: # Try conversion
                 try:
                      df.index = pd.to_datetime(df.index)
                      if df.index.tz is None: df.index = df.index.tz_localize('UTC')
                      elif df.index.tz != datetime.timezone.utc: df.index = df.index.tz_convert('UTC')
                 except Exception as e: logger.error(f"Index conv error {coord}: {e}"); processed_data_step2[coord] = df_orig; continue

            rename_dict = {};
            if 'Radar Data' in df.columns and 'Radar_Data_mm_per_min' not in df.columns: rename_dict['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns and 'Gauge_Data_mm_per_min' not in df.columns: rename_dict['Gauge Data'] = 'Gauge_Data_mm_per_min'
            if rename_dict: df.rename(columns=rename_dict, inplace=True)
            df = initialize_adjustment_columns(df)
            if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns: df = apply_feature_engineering({coord: df})[coord]
            processed_data_step2[coord] = df
        all_data = processed_data_step2; del processed_data_step2



        # --- 6. Batch Processing Loop ---
        print("Steps 3b, 4, 5: Processing in 24h batches...")
        # ... (Generate batch_periods as before) ...
        global_start = pd.Timestamp.max.tz_localize('UTC'); global_end = pd.Timestamp.min.tz_localize('UTC'); valid_indices_found = False
        for df in all_data.values(): # Find range
            if df is not None and isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex) and df.index.tz == datetime.timezone.utc and not df.empty:
                 current_start=df.index.min(); current_end=df.index.max()
                 global_start=min(global_start,current_start); global_end=max(global_end,current_end); valid_indices_found=True
        if not valid_indices_found or global_start>=global_end: logger.error("No valid global time range."); return
        batch_start_times = pd.date_range(start=global_start.floor(CFG.BATCH_DURATION), end=global_end, freq=CFG.BATCH_DURATION, tz='UTC')
        batch_periods = [(start, start + CFG.BATCH_DURATION) for start in batch_start_times]
        if not batch_periods: logger.error("No batch periods."); return
        logger.info(f"Generated {len(batch_periods)} batches from {batch_periods[0][0]} to {batch_periods[-1][1]}.")


        # --- Loop through batches ---
        for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):

                        # --- Step 2: Calculate "24h Adjusted Radar" (Batch Alpha Adjustment) ---
            ################################################
            ''' Workflow Step 2: Calculate Batch Alpha Adjusted Radar
                Output column: Batch_Alpha_Adjusted_Radar
                Uses: Timestep Alpha over the 24h batch for self and neighbors
            '''
            ################################################
            logger.info(f"Batch {batch_start}: Calculating Step 2 (Batch Alpha Adj)...")
            batch_avg_alphas = {} # Store avg alpha for each sensor in this batch
            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if batch_mask.any() and 'Alpha' in df.columns:
                    batch_alpha_series = df.loc[batch_mask, 'Alpha']
                    # Calculate mean, ignore NaNs, default to 1.0 if all are NaN or empty
                    avg_alpha = batch_alpha_series.mean(skipna=True)
                    batch_avg_alphas[coord] = avg_alpha if pd.notna(avg_alpha) else 1.0
                else:
                    batch_avg_alphas[coord] = 1.0 # Default if no data or no Alpha

            # Now calculate the adjusted radar for each sensor using median batch avg alpha
            for coord, df_target in all_data.items():
                batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask.any(): continue

                neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                # Include self in median calculation? Let's include it.
                relevant_coords = [coord] + [n for n in neighbors if n in batch_avg_alphas]
                batch_alpha_values = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords] # Get precalculated batch averages

                median_batch_avg_alpha = np.nanmedian(batch_alpha_values) if batch_alpha_values else 1.0
                if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0 # Handle NaN median

                # Apply this single factor to the raw radar for the batch slice
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = (df_target.loc[batch_mask, 'Radar_Data_mm_per_min'] + CFG.EPSILON) * median_batch_avg_alpha
                else:
                    df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = np.nan

             # --- Step 3: Detect Faulty Gauges (Timestep Flags -> Batch Flag) ---
            # --- 3a: Calculate intermediate diff/ratio for flagging ---
            # **DECISION:** Use the NEW Step 2 result ('Batch_Alpha_Adjusted_Radar')
            # or the OLD 'Network_Adjusted_Radar' (instantaneous median alpha) for flagging?
            # Let's use the NEW one for consistency with the redefined Step 2.
            logger.info(f"Batch {batch_start}: Calculating inputs for timestep flagging...")
            for coord, df in all_data.items():
                 batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                 if not batch_mask.any(): continue

                 req_cols = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                 if all(c in df.columns for c in req_cols) and \
                    not df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'].isnull().all() and \
                    not df.loc[batch_mask, 'Rolling_Gauge'].isnull().all():

                     # Need rolling version of Batch_Alpha_Adjusted_Radar for THIS BATCH
                     rolling_batch_alpha_adj = df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'].rolling(CFG.ROLLING_WINDOW, center=True, min_periods=1).mean().ffill().bfill()
                     rolling_gauge_batch = df.loc[batch_mask, 'Rolling_Gauge'] # Rolling gauge was calculated over whole series

                     # Align just to be safe, though indices should match
                     rolling_gauge_aligned = rolling_gauge_batch.reindex(rolling_batch_alpha_adj.index)

                     diff = rolling_batch_alpha_adj - rolling_gauge_aligned
                     ratio = (rolling_batch_alpha_adj + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                     ratio.replace([np.inf, -np.inf], 3.0, inplace=True)
                     ratio_clipped = ratio.clip(upper=3.0).fillna(1.0)

                     df.loc[batch_mask, 'Adjusted_Diff_from_network'] = diff
                     df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = ratio_clipped
                 else:
                     df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                     df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan

            # --- Pass 1: Calculate Batch_Flag for all sensors in this batch ---
            batch_flags_this_batch = {}
            for coord, df in all_data.items():
                 batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                 if not batch_mask.any():
                     batch_flags_this_batch[coord] = False; continue

                 # Apply flag_anomalies logic ONLY to the batch slice
                 # Need to pass a dict containing only the slice to flag_anomalies
                 df_slice_dict = {coord: df.loc[batch_mask].copy()}
                 flagged_slice_dict = flag_anomalies(df_slice_dict) # This adds 'Flagged' to the slice copy

                 # Assign the 'Flagged' result back to the main DataFrame
                 if coord in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord].columns:
                      df.loc[batch_mask, 'Flagged'] = flagged_slice_dict[coord]['Flagged']
                 else: # Handle cases where flagging failed for the slice
                      df.loc[batch_mask, 'Flagged'] = False # Default to False

                 # Now determine Batch_Flag based on the 'Flagged' column just assigned
                 batch_flag = False
                 df_batch_flagged = df.loc[batch_mask] # Get the slice again *with* 'Flagged'
                 if 'Flagged' in df_batch_flagged.columns and not df_batch_flagged.empty:
                     flagged_points = df_batch_flagged['Flagged'].sum()
                     total_points = len(df_batch_flagged)
                     percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                     if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT: batch_flag = True
                 batch_flags_this_batch[coord] = batch_flag
                 df.loc[batch_mask, 'Batch_Flag'] = batch_flag


            # --- Pass 2: Calculate f_reg and Adjustments ---
            f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end)
            # logger.debug(f"Batch {batch_start} to {batch_end}: f_reg={f_reg:.3f}, valid_gauges={valid_count}") # Debug log

            # Pre-fetch data *needed for Step 5b* for this batch slice
            # We need: Alpha, Flagged (timestep)
            batch_data_cache = {}
            for coord, df in all_data.items():
                 batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                 if batch_mask.any():
                      cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                      batch_data_cache[coord] = df.loc[batch_mask, [c for c in cols_to_cache if c in df.columns]].copy() # Cache only needed cols


            # Apply adjustments (Vectorized Step 5b)
            for coord, df_target in all_data.items(): # Iterate through sensors to adjust
                batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask.any(): continue

                # --- 5a: Calculate Radar_Freg_Adjusted ---
                if 'Radar_Data_mm_per_min' in df_target.columns:
                     df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'] * f_reg
                else:
                     df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = np.nan


                # --- 5b: Calculate Final_Adjusted_Rainfall (Vectorized) ---
                neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                current_batch_index = df_target.loc[batch_mask].index

                # Initialize outputs for this batch slice
                num_valid_neighbors_series = pd.Series(0, index=current_batch_index, dtype=int)
                local_factor_series = pd.Series(1.0, index=current_batch_index, dtype=float) # Default to 1.0

                valid_neighbor_data = {} # Store alpha and flag for valid neighbors

                # Build boolean masks for validity conditions across the batch index
                # Use the PRE-FETCHED cache
                neighbor_is_valid_mask = pd.DataFrame(index=current_batch_index) # Start with empty DF

                for n in neighbors:
                    if n == coord: continue # Don't use self as neighbor

                    # Check batch flag (single value for the batch)
                    is_batch_faulty = batch_flags_this_batch.get(n, True)
                    if is_batch_faulty:
                        neighbor_is_valid_mask[n] = False # Mark all timesteps invalid if batch faulty
                        continue

                    # Check if neighbor data was cached for this batch
                    neighbor_batch_df = batch_data_cache.get(n)
                    if neighbor_batch_df is None or neighbor_batch_df.empty:
                         neighbor_is_valid_mask[n] = False
                         continue

                    # Reindex cached data just in case (should match if cache was done right)
                    neighbor_alpha = neighbor_batch_df.get('Alpha').reindex(current_batch_index)
                    neighbor_flagged = neighbor_batch_df.get('Flagged').reindex(current_batch_index)

                    # Create validity mask for this neighbor
                    valid_mask_n = pd.Series(True, index=current_batch_index)
                    if neighbor_flagged is not None:
                         valid_mask_n &= ~neighbor_flagged.fillna(True) # Not flagged (NaN treated as flagged)
                    if neighbor_alpha is not None:
                         valid_mask_n &= neighbor_alpha.notna() # Has valid alpha
                         valid_mask_n &= (neighbor_alpha > 0) # Ensure alpha is positive if needed
                    else: # No alpha means invalid
                         valid_mask_n = pd.Series(False, index=current_batch_index)

                    neighbor_is_valid_mask[n] = valid_mask_n

                    # Store data only for potentially valid neighbors to calculate median later
                    if valid_mask_n.any(): # Only store if potentially useful
                        valid_neighbor_data[n] = neighbor_alpha # Store alpha series


                # Calculate number of valid neighbors at each timestep
                if not neighbor_is_valid_mask.empty:
                     num_valid_neighbors_series = neighbor_is_valid_mask.sum(axis=1)

                # Calculate local factor (median alpha of valid neighbors)
                if valid_neighbor_data: # Check if there's any neighbor data
                     df_valid_neighbor_alphas = pd.DataFrame(valid_neighbor_data)
                     # Apply the validity mask *before* calculating median
                     masked_alphas = df_valid_neighbor_alphas.where(neighbor_is_valid_mask[valid_neighbor_data.keys()]) # Align mask columns
                     local_factor_series = masked_alphas.median(axis=1, skipna=True)
                     local_factor_series = local_factor_series.fillna(1.0).clip(lower=0.1) # Fill NaNs, clip result

                # Determine weight based on count series
                conditions = [
                    num_valid_neighbors_series == 1,
                    num_valid_neighbors_series == 2,
                    num_valid_neighbors_series >= 3
                ]
                choices = [1.0/3.0, 2.0/3.0, 1.0]
                weight_series = pd.Series(np.select(conditions, choices, default=0.0), index=current_batch_index)

                # Calculate final weighted factor
                weighted_local_factor_series = local_factor_series * weight_series + 1.0 * (1.0 - weight_series)

                # Apply to the freg adjusted radar
                target_freg_radar_series = df_target.loc[batch_mask, 'Radar_Freg_Adjusted']
                final_adjusted_batch_series = target_freg_radar_series * weighted_local_factor_series

                # Assign back to the main DataFrame slice
                df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series


        # --- 7. Post-processing: Calculate Final Rolling Errors ---
        # ... (as before) ...
        print("Calculating final rolling error metrics...")
        for coord, df_orig in tqdm(all_data.items(), desc="Calculating Rolling Errors"):
            df = df_orig.copy()
            if 'Network_Adjusted_Radar' in df and 'Gauge_Data_mm_per_min' in df:
                 gauge_filled = df['Gauge_Data_mm_per_min'].ffill().bfill(); net_adj_filled = df['Network_Adjusted_Radar'].ffill().bfill()
                 if not net_adj_filled.isnull().all() and not gauge_filled.isnull().all():
                     abs_error = abs(net_adj_filled - gauge_filled); df['Rolling_Abs_Error'] = abs_error.rolling('60min', center=True, min_periods=1).mean()
                     ratio = net_adj_filled / (gauge_filled + CFG.EPSILON); flag_points = ((ratio > 1 + CFG.RATIO_THRESHOLD) | (ratio < 1 - CFG.RATIO_THRESHOLD)).astype(float)
                     df['Rolling_Prop_Flagged'] = flag_points.rolling('60min', center=True, min_periods=1).mean()
                 else: df['Rolling_Abs_Error'] = np.nan; df['Rolling_Prop_Flagged'] = np.nan
            else: df['Rolling_Abs_Error'] = np.nan; df['Rolling_Prop_Flagged'] = np.nan
            all_data[coord] = df # Update the main dictionary


        # --- 8. Final Aggregation, Saving, Plotting ---
        # ... (as before, ensure functions are called correctly) ...
        events_df = None; all_data_iter0 = None
        logger.info("Generating final visualizations...")
        print("Generating final visualizations...")
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels, events_df=events_df, all_data_iter0=all_data_iter0)
        if not gdf_wgs84.empty: generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84, svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        print('calling the create_flagging_plots_dashboard.....................................')
        create_flagging_plots_dashboard(all_data, events_df=events_df, output_dir=str(CFG.FLAGGING_PLOTS_DIR), dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))
        # ... (Debug plots) ...

    except Exception as e:
        logger.exception("--- An error occurred during batch adjustment execution ---")
        print("\n--- An error occurred during batch adjustment execution ---")
        traceback.print_exc()
    finally:
        logger.info("Batch adjustment process finished.")
        print("\nBatch adjustment process finished.")
        logging.shutdown()

if __name__ == "__main__":
    main_batch_adjustment()