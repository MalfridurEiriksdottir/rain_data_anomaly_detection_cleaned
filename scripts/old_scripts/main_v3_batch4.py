# --- START OF FILE main_v3_batch4.py ---
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging
import datetime

# Custom module imports
from config import CFG # Ensure CFG has SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
from anomaly import flag_anomalies
from plotting3_batches2 import create_plots_with_error_markers, generate_html_dashboard
from network import get_nearest_neighbors
from batch_adjustment import compute_regional_adjustment

# Basic logger setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# Helper Function Definition
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

# =============================================================================
# Main Batch Adjustment Function
# =============================================================================
def main_batch_adjustment():
    all_data_final_processed = {}
    try:
        # ---------------------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------------------
        logger.info("--- Loading Initial Data ---")
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        utm_to_channel_desc_map, svk_coords_utm = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        
        if not initial_all_data: logger.error("No initial data loaded. Exiting."); return
        
        all_data_loaded = {str(k): v for k, v in initial_all_data.items()}
        coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
        valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}

        # ---------------------------------------------------------------------
        # 2. Preprocessing
        # ---------------------------------------------------------------------
        logger.info("--- Preprocessing Data ---")
        preprocessed_data_for_batching = {}
        for coord, df_orig in tqdm(all_data_loaded.items(), desc="Preprocessing Sensors"):
            if not isinstance(df_orig, pd.DataFrame) or df_orig.empty:
                logger.warning(f"Skipping {coord}: Original data not a DataFrame or empty.")
                continue
            df = df_orig.copy()

            # --- Step 2a: Ensure DatetimeIndex and that it represents TRUE UTC ---
            # This assumes that get_gauge_dataX.py saves true UTC for gauges,
            # and get_radar_data2.py saves true UTC for radar.
            try:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df.dropna(subset=[df.index.name if df.index.name else 'index'], inplace=True)
                if df.empty: logger.warning(f"{coord} empty after to_datetime. Skipping."); continue
                
                if df.index.tz is None: # If naive, assume it's true UTC numbers
                    df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz) != 'UTC': # If other timezone, convert to true UTC
                    df.index = df.index.tz_convert('UTC')
                # Now, df.index is TRUE UTC for both gauge and radar.
            except Exception as e:
                logger.error(f"Error standardizing index for {coord} to TRUE UTC: {e}. Skipping.")
                continue
            
            # --- Step 2b: Conditionally Shift Radar Timestamps (for display preference) ---
            # This step changes the *numbers* of the UTC-labelled radar index to match CPH local time.
            # Gauge data index (true UTC) is NOT affected.
            
            # Define how to identify a DataFrame as containing radar data for this specific shift
            is_radar_df = ('Radar_Data_mm_per_min' in df.columns and 
                           'Gauge_Data_mm_per_min' not in df.columns) # Example: radar-only DataFrame
            # You might also have combined DataFrames where you still want to apply this to the radar's conceptual time
            # In that case, this simple check isn't enough, and the premise of shifting the whole index is flawed.
            # This example proceeds assuming 'is_radar_df' correctly identifies when this shift is desired.

            apply_radar_shift_to_local_numbers = getattr(CFG, 'SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC', False)

            if is_radar_df and apply_radar_shift_to_local_numbers and not df.empty:
                logger.info(f"Applying 'local numbers as UTC' shift to RADAR timestamps for {coord}...")
                original_true_utc_radar_index_min = df.index.min() # Log true UTC
                try:
                    cph_local_aware = df.index.tz_convert('Europe/Copenhagen') # True UTC -> CPH Local (e.g., 05:00Z -> 07:00+02)
                    naive_cph_numbers = cph_local_aware.tz_localize(None)    # CPH Local -> Naive (e.g., 07:00+02 -> 07:00)
                    df.index = naive_cph_numbers.tz_localize('UTC', ambiguous='raise', nonexistent='raise') # Naive -> "Fake UTC" (e.g., 07:00 -> 07:00Z)
                    logger.debug(f"Shifted RADAR index for {coord}. True UTC start: {original_true_utc_radar_index_min}, New 'LocalNumAsUTC' start: {df.index.min()}")
                except Exception as e_shift:
                    logger.error(f"Error shifting radar timestamps for {coord}: {e_shift}. Index remains TRUE UTC.")
                    # Revert to true UTC state if shift fails (it was true UTC before this block)
                    df.index = pd.to_datetime(df_orig.index, errors='coerce').dropna().tz_localize('UTC', errors='ignore')
                    if df.index.tz is None: df.index = df.index.tz_localize('UTC') # ensure
                    elif str(df.index.tz) != 'UTC': df.index = df.index.tz_convert('UTC')

            elif is_radar_df and not apply_radar_shift_to_local_numbers:
                 logger.info(f"Radar for {coord} remains TRUE UTC as per CFG.SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC=False.")


            # --- Step 2c: Rename Columns ---
            rename_map = {}
            if 'Radar Data' in df.columns: rename_map['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns: rename_map['Gauge Data'] = 'Gauge_Data_mm_per_min'
            if rename_map: df.rename(columns=rename_map, inplace=True)
            
            # --- Step 2d: Fill NA for Gauge Data ---
            if 'Gauge_Data_mm_per_min' in df.columns:
                df['Gauge_Data_mm_per_min'] = df['Gauge_Data_mm_per_min'].fillna(0.0)
            
            # --- Step 2e: Initialize Columns ---
            df = initialize_adjustment_columns(df)

            # --- Step 2f: Feature Engineering ---
            df = apply_feature_engineering({coord: df})[coord]
            
            preprocessed_data_for_batching[coord] = df
        
        all_data_for_batches = preprocessed_data_for_batching
        if not all_data_for_batches: logger.error("No data after preprocessing. Exiting."); return
        del preprocessed_data_for_batching

        # ---------------------------------------------------------------------
        # 3. Batch Processing Loop (All data now uses its final UTC-labelled index)
        # ---------------------------------------------------------------------
        logger.info("--- Starting Batch Processing Loop ---")
        global_start = pd.Timestamp.max.tz_localize('UTC')
        global_end = pd.Timestamp.min.tz_localize('UTC')
        # ... (rest of global_start, global_end, batch_periods calculation is the same) ...
        valid_indices_found = False
        for df_batch_check in all_data_for_batches.values():
            if (isinstance(df_batch_check, pd.DataFrame) and 
                isinstance(df_batch_check.index, pd.DatetimeIndex) and 
                str(df_batch_check.index.tz) == 'UTC' and not df_batch_check.empty):
                current_start = df_batch_check.index.min()
                current_end = df_batch_check.index.max()
                if pd.notna(current_start): global_start = min(global_start, current_start)
                if pd.notna(current_end): global_end = max(global_end, current_end)
                valid_indices_found = True

        if not valid_indices_found or global_start >= global_end or pd.isna(global_start) or pd.isna(global_end):
            logger.error(f"No valid data or invalid time range for batching. Global Start: {global_start}, Global End: {global_end}")
            return
        
        batch_start_times = pd.date_range(start=global_start.floor(CFG.BATCH_DURATION),
                                          end=global_end, freq=CFG.BATCH_DURATION, tz='UTC')
        batch_periods = [(start, start + CFG.BATCH_DURATION) for start in batch_start_times]
        if not batch_periods:
            logger.error("No batch periods generated.")
            return
        
        logger.info(f"Generated {len(batch_periods)} batch periods from {batch_periods[0][0]} to {batch_periods[-1][1]}")

        for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):
            logger.debug(f"Processing Batch: {batch_start} to {batch_end}")
            
            # --- Batch Alpha Adjusted Radar ---
            batch_avg_alphas = {}
            for coord, df in all_data_for_batches.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if batch_mask.any() and 'Alpha' in df.columns:
                    avg_alpha = df.loc[batch_mask, 'Alpha'].mean(skipna=True)
                    batch_avg_alphas[coord] = avg_alpha if pd.notna(avg_alpha) else 1.0
                else: batch_avg_alphas[coord] = 1.0

            for coord, df_target in all_data_for_batches.items():
                batch_mask_target = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask_target.any(): continue
                neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                relevant_coords_alpha = [coord] + [n for n in neighbors if n in batch_avg_alphas]
                batch_alpha_values_median = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords_alpha]
                median_batch_avg_alpha = np.nanmedian(batch_alpha_values_median) if batch_alpha_values_median else 1.0
                if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    df_target.loc[batch_mask_target, 'Batch_Alpha_Adjusted_Radar'] = \
                        (df_target.loc[batch_mask_target, 'Radar_Data_mm_per_min'] + CFG.EPSILON) * median_batch_avg_alpha

            # --- Prepare Flagging Metrics ---
            for coord, df in all_data_for_batches.items():
                batch_mask_flagprep = (df.index >= batch_start) & (df.index < batch_end)
                if not batch_mask_flagprep.any(): continue
                req_cols_flagprep = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                if all(c in df.columns for c in req_cols_flagprep) and \
                   not df.loc[batch_mask_flagprep, 'Rolling_Gauge'].isnull().all() and \
                   not df.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar'].isnull().all():
                    rolling_batch_alpha_adj = df.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar']\
                        .rolling(CFG.ROLLING_WINDOW, center=True, min_periods=1).mean().ffill().bfill()
                    rolling_gauge_batch = df.loc[batch_mask_flagprep, 'Rolling_Gauge']
                    rolling_gauge_aligned = rolling_gauge_batch.reindex(rolling_batch_alpha_adj.index)
                    diff = rolling_batch_alpha_adj - rolling_gauge_aligned
                    ratio_raw = (rolling_batch_alpha_adj + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                    ratio_no_inf = ratio_raw.replace([np.inf, -np.inf], 3.0)
                    ratio_clipped = ratio_no_inf.clip(lower=0.0, upper=3.0).fillna(1.0)
                    df.loc[batch_mask_flagprep, 'Adjusted_Diff_from_network'] = diff
                    df.loc[batch_mask_flagprep, 'Adjusted_Ratio_From_Network'] = ratio_clipped

            # --- Flagging Anomalies & Batch Reliability ---
            batch_flags_this_batch = {} 
            for coord, df in all_data_for_batches.items():
                batch_mask_flagging = (df.index >= batch_start) & (df.index < batch_end)
                if not batch_mask_flagging.any():
                    batch_flags_this_batch[coord] = False 
                    df.loc[batch_mask_flagging, 'Batch_Flag'] = False 
                    continue
                df_slice_for_flagging = {coord: df.loc[batch_mask_flagging].copy()}
                flagged_slice_dict = flag_anomalies(df_slice_for_flagging) 
                if coord in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord].columns:
                    df.loc[batch_mask_flagging, 'Flagged'] = flagged_slice_dict[coord]['Flagged']
                is_batch_faulty = False
                if 'Flagged' in df.columns:
                    df_batch_current_sensor = df.loc[batch_mask_flagging]
                    if not df_batch_current_sensor.empty:
                        flagged_points = df_batch_current_sensor['Flagged'].sum()
                        total_points = len(df_batch_current_sensor)
                        percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                        if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT: is_batch_faulty = True
                batch_flags_this_batch[coord] = is_batch_faulty
                df.loc[batch_mask_flagging, 'Batch_Flag'] = is_batch_faulty

            # --- Compute Regional Adjustment (f_reg) ---
            f_reg, valid_sensor_count_for_freg = compute_regional_adjustment(
                all_data_for_batches, batch_start, batch_end, batch_flags_this_batch
            )
            logger.debug(f"Batch {batch_start}-{batch_end}: f_reg = {f_reg:.4f} from {valid_sensor_count_for_freg} valid sensors.")

            # --- Compute Final Adjustments (Final_Adjusted_Rainfall) ---
            batch_data_cache_for_final = {}
            for coord_cache, df_cache in all_data_for_batches.items():
                batch_mask_cache = (df_cache.index >= batch_start) & (df_cache.index < batch_end)
                if batch_mask_cache.any():
                    cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                    batch_data_cache_for_final[coord_cache] = df_cache.loc[batch_mask_cache, [c for c in cols_to_cache if c in df_cache.columns]].copy()
            
            for coord, df_target in all_data_for_batches.items():
                batch_mask_final = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask_final.any(): continue
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    df_target.loc[batch_mask_final, 'Radar_Freg_Adjusted'] = \
                        df_target.loc[batch_mask_final, 'Radar_Data_mm_per_min'] * f_reg

                neighbors_final = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                current_batch_index_final = df_target.loc[batch_mask_final].index
                num_valid_neighbors_series = pd.Series(0, index=current_batch_index_final, dtype=int)
                local_factor_series = pd.Series(1.0, index=current_batch_index_final, dtype=float)
                valid_neighbor_alphas_for_local = {}
                neighbor_is_valid_timestep_mask = pd.DataFrame(index=current_batch_index_final)

                for n_coord in neighbors_final:
                    if n_coord == coord: continue 
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
                    # Ensure keys used for .where() exist as columns in neighbor_is_valid_timestep_mask
                    valid_keys_for_mask = [key for key in valid_neighbor_alphas_for_local.keys() if key in neighbor_is_valid_timestep_mask.columns]
                    if valid_keys_for_mask: # Proceed only if there are valid keys
                        masked_n_alphas = df_valid_n_alphas.where(neighbor_is_valid_timestep_mask[valid_keys_for_mask])
                        local_factor_series = masked_n_alphas.median(axis=1, skipna=True).fillna(1.0).clip(lower=0.1)
                    else: # No valid keys to use for masking, local_factor_series remains default 1.0
                        logger.debug(f"No valid neighbor keys for masking alphas for coord {coord} in batch {batch_start}")
                        local_factor_series.fillna(1.0, inplace=True) # Ensure it's filled
                else: # No valid neighbor alphas, local_factor_series remains default 1.0
                    local_factor_series.fillna(1.0, inplace=True)

                weight_conditions = [num_valid_neighbors_series == 1, num_valid_neighbors_series == 2, num_valid_neighbors_series >= 3]
                weight_choices = [1.0/3.0, 2.0/3.0, 1.0]
                weight_series = pd.Series(np.select(weight_conditions, weight_choices, default=0.0), index=current_batch_index_final)
                f_reg_series = pd.Series(f_reg, index=current_batch_index_final)
                factor_combined = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    raw_radar_in_batch = df_target.loc[batch_mask_final, 'Radar_Data_mm_per_min']
                    df_target.loc[batch_mask_final, 'Final_Adjusted_Rainfall'] = raw_radar_in_batch * factor_combined
        # --- End Batch Loop ---
        all_data_final_processed = all_data_for_batches

        # ... (Saving and Plotting sections remain largely the same) ...
        # ---------------------------------------------------------------------
        # 8. Save Detailed Output CSV per Coordinate
        # ---------------------------------------------------------------------
        logger.info("--- Saving Detailed Output CSV Files ---")
        output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output_batch_v4"  # Changed version for clarity
        output_csv_dir.mkdir(parents=True, exist_ok=True)

        cols_to_save_final = [
            'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Alpha', 
            'Batch_Alpha_Adjusted_Radar', 'Adjusted_Diff_from_network', 
            'Adjusted_Ratio_From_Network', 'Flagged', 'Batch_Flag', 
            'Radar_Freg_Adjusted', 'Final_Adjusted_Rainfall'
        ]
        for coord_str, df_to_save_orig in tqdm(all_data_final_processed.items(), desc="Saving Output CSVs"):
            df_to_save = df_to_save_orig.copy()
            save_cols_present_final = [col for col in cols_to_save_final if col in df_to_save.columns]
            if not save_cols_present_final:
                logger.warning(f"No specified columns to save for {coord_str}. Skipping."); continue
            
            df_for_csv = df_to_save[save_cols_present_final].copy()
            
            # Add Channel Description
            channel_desc_value = "Unknown_Channel"
            utm_tuple_lookup = coordinate_locations_utm.get(str(coord_str)) 
            if utm_tuple_lookup and utm_to_channel_desc_map: # Ensure map is not None
                channel_desc_value = utm_to_channel_desc_map.get(utm_tuple_lookup, "Desc_Not_Found")
            elif not utm_tuple_lookup: channel_desc_value = "UTM_Coord_Not_Found_In_Lookup"
            df_for_csv['Channel_Description'] = str(channel_desc_value)
            
            first_cols_csv = ['Channel_Description']
            remaining_cols_csv = [col for col in df_for_csv.columns if col not in first_cols_csv]
            df_for_csv = df_for_csv[first_cols_csv + remaining_cols_csv]

            safe_coord_filename = str(coord_str).replace('(','').replace(')','').replace(',','_').replace(' ','')
            output_filename = output_csv_dir / f"{safe_coord_filename}_data_adjustments_batch_v4.csv"
            try: 
                # When saving, if df_for_csv.index is tz-aware (UTC from our processing),
                # to_csv will write ISO8601 with offset by default.
                # If you want naive UTC strings: df_for_csv.index.tz_localize(None).to_series() or pass date_format
                df_for_csv.to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z') # Explicitly save with TZ
            except Exception as e_csv: 
                logger.error(f"Failed to save CSV for {coord_str} to {output_filename}: {e_csv}")
        logger.info(f"Saved detailed output CSV files to {output_csv_dir}")

        # ---------------------------------------------------------------------
        # 9. Final Visualization
        # ---------------------------------------------------------------------
        logger.info("--- Generating Final Visualizations ---")
        events_df_placeholder = None
        coord_str_to_channel_desc_map_plot = {}
        if valid_coordinate_locations_utm and utm_to_channel_desc_map: # Ensure map is not None
            for c_str, utm_t_plot in valid_coordinate_locations_utm.items():
                 desc_plot = utm_to_channel_desc_map.get(utm_t_plot, "Unknown Channel")
                 coord_str_to_channel_desc_map_plot[c_str] = desc_plot
        else: logger.warning("Could not create coord_str to channel_description map for plotting.")

        create_plots_with_error_markers(
            all_data_final_processed, valid_coordinate_locations_utm, 
            coord_str_to_channel_desc_map_plot, events_df=events_df_placeholder, all_data_iter0=None
        )
        if gdf_wgs84 is not None and not gdf_wgs84.empty:
            generate_html_dashboard(
                all_data_final_processed, valid_coordinate_locations_utm, gdf_wgs84,
                svk_coords_utm if 'svk_coords_utm' in locals() else [],
                output_file=str(CFG.DASHBOARD_DIR / "dashboard_batch_v4.html") # New dashboard name
            )
        else: logger.warning("GeoDataFrame gdf_wgs84 is empty or None. Skipping dashboard generation.")

    except Exception as e_main:
        logger.critical("\n--- An error occurred during main_batch_adjustment execution ---")
        logger.critical(traceback.format_exc())
    finally:
        logger.info("\nBatch adjustment process finished.")
        logging.shutdown()

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main_batch_adjustment()
# --- END OF FILE main_v3_batch4.py ---