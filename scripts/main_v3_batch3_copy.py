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
import pandas as pd

# def shift_radar_to_utc(df, radar_col='Radar_Data_mm_per_min',
#                        tz_name='Europe/Copenhagen') -> pd.DataFrame:
#     """
#     Correct a DataFrame whose index was (incorrectly) localized to UTC
#     but actually represented Europe/Copenhagen local times.
#     Shifts radar_col timestamps backward by the historic UTC offset
#     so that index becomes true UTC.

#     Returns a new DataFrame with radar_col re-indexed on the corrected times.
#     """
#     # 1. Ensure index is timezone-aware in UTC
#     idx = df.index
#     if idx.tz is None:
#         idx = idx.tz_localize('UTC')
#     else:
#         idx = idx.tz_convert('UTC')

#     print(f'[shift_radar_to_utc] Original index range: {idx.min()} to {idx.max()}')

#     # 2. Compute the actual offset for each timestamp
#     tz = zoneinfo.ZoneInfo(tz_name)
#     # utcoffsets will be Timedelta of either 1h or 2h
#     offsets = idx.map(lambda ts: ts.astimezone(tz).utcoffset())

#     # 3. Subtract those offsets to get the true UTC instants
#     corrected_idx = idx - pd.to_timedelta(offsets)

#     print(f'[shift_radar_to_utc] Corrected index range: {corrected_idx.min()} to {corrected_idx.max()}')

#     # 4. Extract, re-index, and merge back
#     radar = df[radar_col].copy()
#     radar.index = corrected_idx

#     # Drop old radar column and join the re-indexed one
#     df = df.drop(columns=[radar_col])
#     df = df.join(radar).sort_index()

#     return df

import zoneinfo
import pandas as pd

def shift_radar_to_utc(df, radar_col='Radar_Data_mm_per_min',
                       tz_name='Europe/Copenhagen') -> pd.DataFrame:
    """
    The DataFrame index is currently labeled as UTC but actually is local Copenhagen time.
    We add the historic UTC offset (1h or 2h) to recover true UTC instants.
    """
    # 1. Ensure index is treated as UTC
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')

    # Log before/after range
    # print(f"[shift] ORIGINAL UTC‐as‐local range: {idx.min()} → {idx.max()}")

    # 2. Compute the offset (Timedelta of +1h or +2h)
    tz = zoneinfo.ZoneInfo(tz_name)
    offsets = idx.map(lambda ts: ts.astimezone(tz).utcoffset())

    # 3. ADD those offsets to correct
    corrected_idx = idx + pd.to_timedelta(offsets)

    # Log corrected range
    # print(f"[shift] CORRECTED true‐UTC range: {corrected_idx.min()} → {corrected_idx.max()}")

    # 4. Sample pairs for eyeballing
    # for orig, corr in zip(idx[:3], corrected_idx[:3]):
    #     print(f"[shift] sample: {orig} -> {corr}")

    # 5. Rebuild the radar series on corrected times
    radar = df[radar_col].copy()
    radar.index = corrected_idx

    df = df.drop(columns=[radar_col])
    df = df.join(radar).sort_index()

    return df




# =============================================================================
# Main Batch Adjustment Function
# =============================================================================
def main_batch_adjustment():
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
        target_coords, coordinate_locations_utm = load_target_coordinates()  # Load target coordinates (UTM format)
        sensordata, gdf_wgs84 = load_sensor_data()  # Load sensor raw data and geographic DataFrame
        # print(sensordata.columns)
        # print(sensordata['wkt_geom'])
        utm_to_channel_desc_map, svk_coords_utm = process_sensor_metadata(sensordata)  # Process metadata related to sensors
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        # print(initial_all_data['(660016,6126660)'])
        # # print the first few timestamps
        # for coord, df in initial_all_data.items():
        #     if isinstance(df, pd.DataFrame) and not df.empty:
        #         print(f"First timestamp for {coord}: {df.index[0]}")
        #     else:
        #         print(f"No valid data for {coord}")
        
        # Check if data was loaded successfully.
        if not initial_all_data or not valid_coordinate_locations_utm:
            return
        
        # Copy and convert keys to strings for consistency.
        all_data = initial_all_data.copy()
        # print(all_data['(682988,6114834)'])
        all_data = {str(k): v for k, v in all_data.items()}
        coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
        valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}

        

        # ---------------------------------------------------------------------
        # 2. Preprocessing: Rename Columns, Standardize Timezone, Initialize Columns,
        #    and Apply Feature Engineering
        # ---------------------------------------------------------------------
        
        print("Preprocessing...")
        processed_data_step2 = {}
        for coord, df_orig in tqdm(all_data.items(), desc="Preprocessing"):
            df = df_orig.copy()
            # print(f"Processing {coord}...")
            
            # Ensure index is a DatetimeIndex with UTC timezone.
            if isinstance(df.index, pd.DatetimeIndex):
                # print(f"Index is already a DatetimeIndex for {coord}.")
                try:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz != datetime.timezone.utc:
                        df.index = df.index.tz_convert('UTC')
                except Exception as e:
                    print(f"Error standardizing index timezone for {coord} to UTC: {e}. Skipping further processing for this sensor.")
                    processed_data_step2[coord] = df_orig
                    continue
            else:
                try:
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz != datetime.timezone.utc:
                        df.index = df.index.tz_convert('UTC')
                except Exception as e:
                    print(f"Error converting index to DatetimeIndex or localizing to UTC for {coord}: {e}. Skipping.")
                    processed_data_step2[coord] = df_orig
                    continue

            rename_dict = {}
            if 'Radar Data' in df.columns and 'Radar_Data_mm_per_min' not in df.columns:
                rename_dict['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns and 'Gauge_Data_mm_per_min' not in df.columns:
                rename_dict['Gauge Data'] = 'Gauge_Data_mm_per_min'
            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)


            # Rename columns to ensure consistency. 

            # Right after your rename_dict block, e.g.:

            # print('Before shifting radar to UTC:', df.head())

            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)

            # # —> Now correct radar timestamps:
            # if 'Radar_Data_mm_per_min' in df.columns:
            #     df = shift_radar_to_utc(df)
            if 'Radar_Data_mm_per_min' in df.columns:
                # print(f"[{coord}] before shift: {df.index[:3]}")
                df = shift_radar_to_utc(df)
                # print(f"[{coord}] after shift:  {df.index[:3]}")

            # Then continue with gauge fill, initialize_adjustment_columns, etc.

            # print('After shifting radar to UTC:', df.head())

            

                        # <<<--- SET MISSING GAUGE DATA TO ZERO ---<<<
            gauge_col = 'Gauge_Data_mm_per_min'
            if gauge_col in df.columns:
                # Check for NaNs before filling
                # nan_count_before = df[gauge_col].isnull().sum()
                # if nan_count_before > 0:
                #      print(f"Found {nan_count_before} NaNs in {gauge_col} for {coord}. Filling with 0.")
                df[gauge_col] = df[gauge_col].fillna(0.0)
            else:
                 print(f"Warning: Gauge column '{gauge_col}' not found for {coord}. Cannot fill NaNs.")
                 # If gauge data is essential, maybe skip this sensor?
                 # Or create a zero column if subsequent steps require it?
                 # df[gauge_col] = 0.0 # Example: Create a zero column if needed later

            # <<<--- END FILLNA ---<<<

            

            # Initialize required adjustment columns.
            df = initialize_adjustment_columns(df)

            # Apply feature engineering
            if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
                df = apply_feature_engineering({coord: df})[coord]
            
            processed_data_step2[coord] = df
        
        # Update main data dictionary.
        all_data = processed_data_step2
        del processed_data_step2

        
        
        # ---------------------------------------------------------------------
        # 3. Batch Processing Loop: Process Data in 24-Hour Batches
        # ---------------------------------------------------------------------
        print("Steps 3b, 4, 5: Processing in 24h batches...")
        
        
        # Determine the global minimum and maximum timestamps across all sensor data.
        # ------------------------------------------------------------------
        # 3. Decide the period we really need to touch
        # ------------------------------------------------------------------
        from utils import get_last_processed_utc

        last_done = get_last_processed_utc()           # may be None
        now_utc    = pd.Timestamp.utcnow().floor("h")

        # ❶ Don’t touch the most recent hour – wait until tomorrow
        cutoff_utc = now_utc - pd.Timedelta(hours=1)
        # cutoff_utc = (now_utc - pd.Timedelta(hours=1)).tz_localize("UTC")

        # ❷ Compute global_start / global_end over all sensors as today
        global_start = pd.Timestamp.max.tz_localize("UTC")
        global_end   = pd.Timestamp.min.tz_localize("UTC")

        for df in all_data.values():
            if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                global_start = min(global_start, df.index.min())
                global_end   = max(global_end, df.index.max())

        # ❸ Decide the true processing window
        process_from = (last_done + CFG.BATCH_DURATION) if last_done is not None else global_start
        process_to   = min(global_end, cutoff_utc)

        if process_from >= process_to:
            print("Nothing new to adjust – exiting early.")
            return

        
        # Generate batch periods using 24h.
        batch_start_times = pd.date_range(start=process_from.floor(CFG.BATCH_DURATION),
                                  end  =process_to,
                                  freq =CFG.BATCH_DURATION,
                                  tz   ="UTC")

        batch_periods = [(start, start + CFG.BATCH_DURATION) for start in batch_start_times]
        if not batch_periods:
            print("No batch periods generated.")
            return
        
        # Loop through each batch period.
        for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):

            # ---------------------------------------------------------------
            # Step 2: Calculate Batch Alpha Adjusted Radar
            # ---------------------------------------------------------------
            batch_avg_alphas = {}  # Store average alpha for each sensor within the batch
            
            # Compute average Alpha for each sensor over the batch period.
            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if batch_mask.any() and 'Alpha' in df.columns:
                    batch_alpha_series = df.loc[batch_mask, 'Alpha']
                    avg_alpha = batch_alpha_series.mean(skipna=True)
                    batch_avg_alphas[coord] = avg_alpha if pd.notna(avg_alpha) else 1.0
                else:
                    batch_avg_alphas[coord] = 1.0  # Default value if data is missing

            # Adjust radar readings using the median average alpha from sensor and its neighbors.
            
            for coord, df_target in all_data.items():
                
                batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask.any():
                    continue

                # Get nearest neighbors for this sensor.
                neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                # Include the sensor itself in the calculation.
                relevant_coords = [coord] + [n for n in neighbors if n in batch_avg_alphas]
                batch_alpha_values = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords]
                median_batch_avg_alpha = np.nanmedian(batch_alpha_values) if batch_alpha_values else 1.0
                if pd.isna(median_batch_avg_alpha):
                    median_batch_avg_alpha = 1.0

                # Apply adjustment factor to the raw radar data.
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = (
                        df_target.loc[batch_mask, 'Radar_Data_mm_per_min'] + CFG.EPSILON
                    ) * median_batch_avg_alpha
                else:
                    df_target.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'] = np.nan

            # ---------------------------------------------------------------
            # Step 3: Detect Faulty Gauges & Prepare Flagging Metrics
            # ---------------------------------------------------------------
            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if not batch_mask.any():
                    continue

                req_cols = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                if (all(c in df.columns for c in req_cols) and 
                    
                    
                    not df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar'].isnull().all() and 
                    not df.loc[batch_mask, 'Rolling_Gauge'].isnull().all()):


                    neighbors_step3 = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                    relevant_coords_step3 = [coord] + [n for n in neighbors_step3 if n in batch_avg_alphas]
                    batch_alpha_values_step3 = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords_step3]


                    median_batch_avg_alpha_step3 = np.nanmedian(batch_alpha_values_step3) if batch_alpha_values_step3 else 1.0
                    if pd.isna(median_batch_avg_alpha_step3):
                       median_batch_avg_alpha_step3 = 1.0
                    
                    # Calculate rolling mean for adjusted radar data over the batch.
                    rolling_batch_alpha_adj = df.loc[batch_mask, 'Batch_Alpha_Adjusted_Radar']\
                        .rolling(CFG.ROLLING_WINDOW, center=True, min_periods=1).mean().ffill().bfill()
                    # Rolling gauge value (pre-calculated over the whole series).
                    rolling_gauge_batch = df.loc[batch_mask, 'Rolling_Gauge']
                    rolling_gauge_aligned = rolling_gauge_batch.reindex(rolling_batch_alpha_adj.index)
                    
                    # Compute difference and ratio between adjusted radar and gauge data.
                    diff = rolling_batch_alpha_adj - rolling_gauge_aligned
                    ratio = (rolling_batch_alpha_adj + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                    ratio.replace([np.inf, -np.inf], 3.0, inplace=True)
                    ratio_clipped = ratio.clip(upper=3.0).fillna(1.0)

                    df.loc[batch_mask, 'Adjusted_Diff_from_network'] = diff
                    df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = ratio_clipped
                else:
                    df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                    df.loc[batch_mask, 'Adjusted_Ratio_From_network'] = np.nan

            # ---------------------------------------------------------------
            # Pass 1: Flagging Anomalies per Batch
            # ---------------------------------------------------------------
            batch_flags_this_batch = {}

            reliable_sensors_debug = [] # List to store reliable sensors for this batch
            unreliable_sensors_debug = []

            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if not batch_mask.any():
                    batch_flags_this_batch[coord] = False
                    continue

                # Flag anomalies only for the current batch slice.
                df_slice_dict = {coord: df.loc[batch_mask].copy()}
                flagged_slice_dict = flag_anomalies(df_slice_dict)
                if coord in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord].columns:
                    df.loc[batch_mask, 'Flagged'] = flagged_slice_dict[coord]['Flagged']
                else:
                    df.loc[batch_mask, 'Flagged'] = False  # Default flag if flagging fails

                # Determine if the sensor is faulty during this batch based on the flag percentage.
                batch_flag = False
                df_batch_flagged = df.loc[batch_mask]
                if 'Flagged' in df_batch_flagged.columns and not df_batch_flagged.empty:

                    flagged_points = df_batch_flagged['Flagged'].sum()
                    total_points = len(df_batch_flagged)
                    percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0

                    if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT:
                        batch_flag = True
                batch_flags_this_batch[coord] = batch_flag
                df.loc[batch_mask, 'Batch_Flag'] = batch_flag

                



            # ---------------------------------------------------------------
            # Pass 2: Compute Regional Adjustment and Additional Corrections
            # ---------------------------------------------------------------

            f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_this_batch)



            # Pre-fetch necessary columns for later adjustments (Step 5b)
            batch_data_cache = {}
            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if batch_mask.any():
                    cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                    batch_data_cache[coord] = df.loc[batch_mask, [c for c in cols_to_cache if c in df.columns]].copy()

            # ---------------------------------------------------------------
            # Step 5: Compute Final Adjustments
            # ---------------------------------------------------------------
            for coord, df_target in all_data.items():

                batch_mask = (df_target.index >= batch_start) & (df_target.index < batch_end)
                if not batch_mask.any():
                    continue

                # 5a: Calculate the adjusted radar signal based on regional factor.
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = df_target.loc[batch_mask, 'Radar_Data_mm_per_min'] * f_reg
                else:
                    df_target.loc[batch_mask, 'Radar_Freg_Adjusted'] = np.nan

                # 5b: Calculate Final Adjusted Rainfall using neighbor-based adjustments.
                neighbors = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                current_batch_index = df_target.loc[batch_mask].index

                # Initialize series for counting valid neighbors and the local adjustment factor.
                num_valid_neighbors_series = pd.Series(0, index=current_batch_index, dtype=int)
                local_factor_series = pd.Series(1.0, index=current_batch_index, dtype=float)

                valid_neighbor_data = {}  # To store alpha values from valid neighbors
                neighbor_is_valid_mask = pd.DataFrame(index=current_batch_index)

                # Evaluate neighbor validity across each timestep.
                for n in neighbors:
                    if n == coord:
                        continue 
                    is_batch_faulty = batch_flags_this_batch.get(n, True)
                    if is_batch_faulty:
                        neighbor_is_valid_mask[n] = False
                        continue

                    neighbor_batch_df = batch_data_cache.get(n)
                    if neighbor_batch_df is None or neighbor_batch_df.empty:
                        neighbor_is_valid_mask[n] = False
                        continue

                    neighbor_alpha = neighbor_batch_df.get('Alpha').reindex(current_batch_index)
                    neighbor_flagged = neighbor_batch_df.get('Flagged').reindex(current_batch_index)
                    
                    # Create a validity mask for this neighbor.
                    valid_mask_n = pd.Series(True, index=current_batch_index)
                    if neighbor_flagged is not None:
                        valid_mask_n &= ~neighbor_flagged.fillna(True)
                    if neighbor_alpha is not None:
                        valid_mask_n &= neighbor_alpha.notna()
                        valid_mask_n &= (neighbor_alpha > 0)
                    else:
                        valid_mask_n = pd.Series(False, index=current_batch_index)

                    neighbor_is_valid_mask[n] = valid_mask_n

                    # Store neighbor alpha series if at least one valid timestep exists.
                    if valid_mask_n.any():
                        valid_neighbor_data[n] = neighbor_alpha

                # Count valid neighbor points per timestep.
                if not neighbor_is_valid_mask.empty:
                    num_valid_neighbors_series = neighbor_is_valid_mask.sum(axis=1)

                # Compute local adjustment factor as the median alpha from valid neighbors.
                if valid_neighbor_data:
                    df_valid_neighbor_alphas = pd.DataFrame(valid_neighbor_data)
                    masked_alphas = df_valid_neighbor_alphas.where(neighbor_is_valid_mask[valid_neighbor_data.keys()])
                    local_factor_series = masked_alphas.median(axis=1, skipna=True)
                    local_factor_series = local_factor_series.fillna(1.0).clip(lower=0.1)

                # Determine weight based on the number of valid neighbors.
                conditions = [
                    num_valid_neighbors_series == 1,
                    num_valid_neighbors_series == 2,
                    num_valid_neighbors_series >= 3
                ]
                choices = [1.0 / 3.0, 2.0 / 3.0, 1.0]
                weight_series = pd.Series(np.select(conditions, choices, default=0.0), index=current_batch_index)

                # Create a pandas Series of the regional factor (f_reg), aligned with the batch index
                f_reg_series = pd.Series(f_reg, index=current_batch_index)

                # Calculate the COMBINED factor using Morten's weighting:
                # Factor_Combined = Factor_Local * Weight + f_reg * (1 - Weight)
                factor_combined_series = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)

                # Apply the COMBINED factor directly to the ORIGINAL RAW radar data
                # Ensure 'Radar_Data_mm_per_min' exists before applying
                if 'Radar_Data_mm_per_min' in df_target.columns:
                    raw_radar_series = df_target.loc[batch_mask, 'Radar_Data_mm_per_min']
                    final_adjusted_batch_series = raw_radar_series * factor_combined_series
                else:
                    final_adjusted_batch_series = pd.Series(np.nan, index=current_batch_index)


                # Assign the result to the Final_Adjusted_Rainfall column
                df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series



        # ---------------------------------------------------------------------
        # 7. Post-processing: Calculate Rolling Error Metrics
        # ---------------------------------------------------------------------
        print("Calculating final rolling error metrics...")
        for coord, df_orig in tqdm(all_data.items(), desc="Calculating Rolling Errors"):
            df = df_orig.copy()
            # Ensure both gauge and network adjusted radar data are available.
            if 'Network_Adjusted_Radar' in df and 'Gauge_Data_mm_per_min' in df:
                gauge_filled = df['Gauge_Data_mm_per_min'].ffill().bfill()
                net_adj_filled = df['Network_Adjusted_Radar'].ffill().bfill()
                if not net_adj_filled.isnull().all() and not gauge_filled.isnull().all():
                    abs_error = abs(net_adj_filled - gauge_filled)
                    df['Rolling_Abs_Error'] = abs_error.rolling('60min', center=True, min_periods=1).mean()
                    ratio = net_adj_filled / (gauge_filled + CFG.EPSILON)
                    flag_points = ((ratio > 1 + CFG.RATIO_THRESHOLD) | (ratio < 1 - CFG.RATIO_THRESHOLD)).astype(float)
                    df['Rolling_Prop_Flagged'] = flag_points.rolling('60min', center=True, min_periods=1).mean()
                else:
                    df['Rolling_Abs_Error'] = np.nan
                    df['Rolling_Prop_Flagged'] = np.nan
            else:
                df['Rolling_Abs_Error'] = np.nan
                df['Rolling_Prop_Flagged'] = np.nan
            all_data[coord] = df  # Update main data dictionary

        # --- 8. Save Detailed Output CSV per Coordinate ---
        ################################################
        ''' Saving detailed output file per coordinate
            Includes Raw Data, Adjustments, and Batch Flag status and timestamp flags.
        '''
        ################################################
        print("Saving detailed output CSV files...")
        output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output" 
        output_csv_dir.mkdir(parents=True, exist_ok=True) # Create directory

        cols_to_save = [
            'Gauge_Data_mm_per_min',
            'Radar_Data_mm_per_min',         # Raw Radar
            'Batch_Alpha_Adjusted_Radar',    # Step 2 result ("24h Adjusted")
            'Radar_Freg_Adjusted',           # Debug series (Raw * f_reg)
            'Final_Adjusted_Rainfall',       # Step 5 result ("Final Adjusted")
            'Flagged',                     # Timestep flag (anomaly detection)
            'Batch_Flag'                     # Batch reliability status
        ]

        coordinate_locations_utm_lookup = valid_coordinate_locations_utm if 'valid_coordinate_locations_utm' in locals() else {}
        channel_description_lookup = utm_to_channel_desc_map if 'utm_to_channel_desc_map' in locals() else {}

        for coord, df_final in tqdm(all_data.items(), desc="Saving Output CSVs"):
            coord_str = str(coord)
            save_cols_present = [col for col in cols_to_save if col in df_final.columns]
            if not save_cols_present:
                print(f"No columns to save for {coord_str}.")
                continue

            df_to_save = df_final[save_cols_present].copy()

                # --- START: Add Channel Description Information ---
            channel_desc_value = "Unknown_Channel" # Default value
            try:
                # 1. Get UTM tuple from coordinate string key
                utm_tuple = coordinate_locations_utm_lookup.get(coord_str)
                # print('utm_tuple', utm_tuple)
                # print('coord_str', coord_str)

                if utm_tuple:
                    # 2. Get Channel Description directly from the new map using UTM tuple
                    channel_desc_value = channel_description_lookup.get(utm_tuple, "Description_Not_Found")
                else:
                    # logger.warning(f"Could not find UTM tuple for coord '{coord_str}' in coordinate_locations_utm.")
                    channel_desc_value = "Coord_Not_Found" # More specific default

            except Exception as e:
                # Log the error if logger is configured
                # logger.error(f"Error looking up channel description for {coord_str}: {e}")
                channel_desc_value = "Lookup_Error" # Indicate an error occurred
            # --- END: Add Channel Description Information ---


            # Add the channel description as a new column
            # Using "Channel" as the column name as requested
            df_to_save['Channel'] = str(channel_desc_value) # Ensure string

            # coordinate string for filename
            safe_coord = coord_str.replace('(','').replace(')','').replace(',','_').replace(' ','')
            output_filename = output_csv_dir / f"{safe_coord}_data_adjustments.csv"

            try:
                # Reorder columns to put Channel info first
                first_cols = ['Channel']
                remaining_cols = [col for col in df_to_save.columns if col not in first_cols]
                final_cols = first_cols + remaining_cols
                # df_to_save[final_cols].to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z')
                # Find the last timestamp that is already present in the on-disk file
                file_exists = output_filename.exists()
                if file_exists:
                    last_saved_idx = pd.read_csv(output_filename, usecols=[0], parse_dates=[0]).iloc[-1,0]
                    last_saved_idx = pd.to_datetime(last_saved_idx, utc=True)
                else:
                    last_saved_idx = None

                # keep only new rows
                mask_new = df_to_save.index > (last_saved_idx if last_saved_idx is not None else pd.Timestamp.min.tz_localize("UTC"))
                if not mask_new.any():
                    continue                                     # nothing to write

                df_to_save_new = df_to_save.loc[mask_new]

                df_to_save_new[final_cols].to_csv(
                        output_filename,
                        mode   ="a",           # append
                        header =not file_exists,
                        date_format='%Y-%m-%d %H:%M:%S%z'
                )

                # logger.info(f"Saved detailed data for {coord_str} to {output_filename}")
            except Exception as e:
                # logger.error(f"Failed to save CSV for {coord_str}: {e}")
                print(f"Failed to save CSV for {coord_str}: {e}")

            # try:
            #     # Save with timezone information
            #     df_to_save.to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z')
                
            # except Exception as e:
            #     print(f"Failed to save CSV for {coord_str}: {e}")
        print('Saved detailed output CSV files to ', output_csv_dir)

        # ---------------------------------------------------------------------
        # 8. Final Aggregation, Saving, and Plotting
        # ---------------------------------------------------------------------
        events_df = None
        all_data_iter0 = None
        print("Generating final visualizations...")

        coord_str_to_channel_desc = {}
        if 'valid_coordinate_locations_utm' in locals() and 'utm_to_channel_desc_map' in locals():
            for coord_str, utm_tuple in valid_coordinate_locations_utm.items():
                 desc = utm_to_channel_desc_map.get(utm_tuple, "Unknown Channel")
                 coord_str_to_channel_desc[coord_str] = desc
        else:
            print("Warning: Could not create coordinate to channel map for plotting - required data missing.")
            # coord_str_to_channel_desc remains {}

            
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, coord_str_to_channel_desc,
                                        events_df=events_df, all_data_iter0=all_data_iter0)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84,
                                    svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
            
        from utils import set_last_processed_utc
        set_last_processed_utc(process_to)

    
    except Exception as e:
        print("\n--- An error occurred during batch adjustment execution ---")
        traceback.print_exc()
    
    finally:
        print("\nBatch adjustment process finished.")
        logging.shutdown()


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main_batch_adjustment()