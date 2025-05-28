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
from results import aggregate_results
from plotting3_batches2 import (create_plots_with_error_markers, generate_html_dashboard,
                               create_flagging_plots_dashboard, debug_alpha_for_coord, debug_alpha_and_neighbors_plot)
from network import get_nearest_neighbors
from batch_adjustment import compute_regional_adjustment

# =============================================================================
# Logging Setup
# =============================================================================
# Set up logging to file with a specific formatter and file name based on configuration.
log_file_main = CFG.RESULTS_DIR / 'batch_adjustment_process.log'
formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    main_file_handler = logging.FileHandler(log_file_main, mode='w')
    main_file_handler.setFormatter(formatter_main)
    logger.addHandler(main_file_handler)


# =============================================================================
# Helper Function Definitions
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
    # Initialize missing columns in the DataFrame using the specified types.
    for col, dtype in cols_to_init.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype, index=df.index)
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
    logger.info('Starting BATCH ADJUSTMENT process...')
    all_data = {}

    try:
        # ---------------------------------------------------------------------
        # 1. Load Data
        # ---------------------------------------------------------------------
        logger.info("--- Loading Initial Data ---")
        print("--- Loading Initial Data ---")
        target_coords, coordinate_locations_utm = load_target_coordinates()  # Load target coordinates (UTM format)
        sensordata, gdf_wgs84 = load_sensor_data()  # Load sensor raw data and geographic DataFrame
        sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)  # Process metadata related to sensors
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        
        # Check if data was loaded successfully.
        if not initial_all_data or not valid_coordinate_locations_utm:
            return
        
        # Copy and convert keys to strings for consistency.
        all_data = initial_all_data.copy()
        logger.info(f"Loaded initial data for {len(all_data)} coordinates.")
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
            
            # Ensure index is a DatetimeIndex with UTC timezone.
            if isinstance(df.index, pd.DatetimeIndex):
                try:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    elif df.index.tz != datetime.timezone.utc:
                        df.index = df.index.tz_convert('UTC')
                except Exception as e:
                    logger.warning(f"Timezone error for {coord}: {e}")
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
                    logger.error(f"Index conversion error for {coord}: {e}")
                    processed_data_step2[coord] = df_orig
                    continue

            # Rename columns to ensure consistency.
            rename_dict = {}
            if 'Radar Data' in df.columns and 'Radar_Data_mm_per_min' not in df.columns:
                rename_dict['Radar Data'] = 'Radar_Data_mm_per_min'
            if 'Gauge Data' in df.columns and 'Gauge_Data_mm_per_min' not in df.columns:
                rename_dict['Gauge Data'] = 'Gauge_Data_mm_per_min'
            if rename_dict:
                df.rename(columns=rename_dict, inplace=True)

            # Initialize required adjustment columns.
            df = initialize_adjustment_columns(df)

            # Apply feature engineering if both gauge and radar data columns exist.
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
        global_start = pd.Timestamp.max.tz_localize('UTC')
        global_end = pd.Timestamp.min.tz_localize('UTC')
        valid_indices_found = False
        for df in all_data.values():
            if (df is not None and isinstance(df, pd.DataFrame) and 
                isinstance(df.index, pd.DatetimeIndex) and 
                df.index.tz == datetime.timezone.utc and not df.empty):
                current_start = df.index.min()
                current_end = df.index.max()
                global_start = min(global_start, current_start)
                global_end = max(global_end, current_end)
                valid_indices_found = True
        if not valid_indices_found or global_start >= global_end:
            logger.error("No valid global time range found.")
            return
        
        # Generate batch periods using a configured duration (e.g., 24h).
        batch_start_times = pd.date_range(start=global_start.floor(CFG.BATCH_DURATION),
                                          end=global_end, freq=CFG.BATCH_DURATION, tz='UTC')
        batch_periods = [(start, start + CFG.BATCH_DURATION) for start in batch_start_times]
        if not batch_periods:
            logger.error("No batch periods generated.")
            return
        logger.info(f"Generated {len(batch_periods)} batches from {batch_periods[0][0]} to {batch_periods[-1][1]}.")

        # Loop through each batch period.
        for batch_start, batch_end in tqdm(batch_periods, desc="Processing Batches"):

            # ---------------------------------------------------------------
            # Step 2: Calculate Batch Alpha Adjusted Radar
            # ---------------------------------------------------------------
            logger.info(f"Batch {batch_start}: Calculating Step 2 (Batch Alpha Adjustment)...")
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
            logger.info(f"Batch {batch_start}: Calculating inputs for timestep flagging...")
            for coord, df in all_data.items():
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)
                if not batch_mask.any():
                    continue

                req_cols = ['Rolling_Radar', 'Rolling_Gauge', 'Batch_Alpha_Adjusted_Radar'] # Check if Batch_Alpha_Adjusted_Radar exists for safety
                if all(c in df.columns for c in req_cols):

                    # Retrieve the median batch avg alpha calculated in Step 2 for this sensor/batch
                    # Need to re-calculate or retrieve median_batch_avg_alpha here if not stored globally
                    # Re-calculating for clarity (assuming batch_avg_alphas is available):
                    neighbors_step3 = get_nearest_neighbors(coord, valid_coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                    relevant_coords_step3 = [coord] + [n for n in neighbors_step3 if n in batch_avg_alphas]
                    batch_alpha_values_step3 = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords_step3]
                    median_batch_avg_alpha_step3 = np.nanmedian(batch_alpha_values_step3) if batch_alpha_values_step3 else 1.0
                    if pd.isna(median_batch_avg_alpha_step3):
                       median_batch_avg_alpha_step3 = 1.0

                    # Get the pre-calculated rolling values for the batch period
                    # These values implicitly used data across boundaries when first calculated
                    rolling_radar_batch = df.loc[batch_mask, 'Rolling_Radar']
                    rolling_gauge_batch = df.loc[batch_mask, 'Rolling_Gauge']

                    if not rolling_radar_batch.isnull().all() and not rolling_gauge_batch.isnull().all():

                        # Estimate the rolling adjusted radar using the batch factor
                        # Create a Series of the batch median alpha aligned with the rolling_radar index
                        median_alpha_series = pd.Series(median_batch_avg_alpha_step3, index=rolling_radar_batch.index)
                        rolling_batch_alpha_adj_estimated = rolling_radar_batch * median_alpha_series

                        # Ensure indices align before calculating diff/ratio
                        rolling_gauge_aligned = rolling_gauge_batch.reindex(rolling_batch_alpha_adj_estimated.index)

                        # Compute difference and ratio using the estimated rolling adjusted radar
                        diff = rolling_batch_alpha_adj_estimated - rolling_gauge_aligned
                        ratio = (rolling_batch_alpha_adj_estimated + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                        ratio.replace([np.inf, -np.inf], 3.0, inplace=True) # Cap infinite values
                        ratio_clipped = ratio.clip(upper=3.0).fillna(1.0) # Clip and fill NaNs

                        df.loc[batch_mask, 'Adjusted_Diff_from_network'] = diff
                        df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = ratio_clipped
                    else:
                        # Handle cases where rolling data might be missing within the batch
                        df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                        df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan # Use 1.0 or np.nan based on desired flagging behavior
                else:
                    # Handle cases where required columns are missing
                    logger.warning(f"Missing required columns for flagging input calculation in batch {batch_start} for {coord}. Skipping.")
                    df.loc[batch_mask, 'Adjusted_Diff_from_network'] = np.nan
                    df.loc[batch_mask, 'Adjusted_Ratio_From_Network'] = np.nan # Use 1.0 or np.nan

            # ---------------------------------------------------------------
            # Pass 1: Flagging Anomalies per Batch
            # ---------------------------------------------------------------
            batch_flags_this_batch = {}
            # print('Flagging anomalies per batch...')
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
                    # print(percent_flagged)
                    # print(CFG.FAULTY_GAUGE_THRESHOLD_PERCENT)
                    if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT:
                        batch_flag = True
                        # print(f"Batch {batch_start} to {batch_end}: {coord} flagged as faulty ({percent_flagged:.2f}%)")
                batch_flags_this_batch[coord] = batch_flag
                df.loc[batch_mask, 'Batch_Flag'] = batch_flag
                
                # print('Batch flag added to column')

            # ---------------------------------------------------------------
            # Pass 2: Compute Regional Adjustment and Additional Corrections
            # ---------------------------------------------------------------
            f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end)
            # Debug logging can be enabled if needed:
            # logger.debug(f"Batch {batch_start} to {batch_end}: f_reg={f_reg:.3f}, valid_gauges={valid_count}")

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
                        continue  # Exclude self
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

                # --- START OF MODIFICATION for Factor Combination & Application ---

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
                    # Handle case where raw radar is missing, result should be NaN
                    final_adjusted_batch_series = pd.Series(np.nan, index=current_batch_index)


                # Assign the result to the Final_Adjusted_Rainfall column
                df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series

                # --- END OF MODIFICATION ---

                # # Compute the final weighted local factor and apply it to the radar data.
                # weighted_local_factor_series = local_factor_series * weight_series + 1.0 * (1.0 - weight_series)
                # target_freg_radar_series = df_target.loc[batch_mask, 'Radar_Freg_Adjusted']
                # final_adjusted_batch_series = target_freg_radar_series * weighted_local_factor_series
                # df_target.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted_batch_series

                                # --- Optional: Keep Debug Prints (Update if necessary) ---
                # if coord == DEBUG_COORD and is_debug_batch:
                #     print(f"\nDEBUG ({DEBUG_COORD}) - Step 5b Final Adj. Components (Morten's Method):")
                #     print("  Sample Num Valid Neighbors (first 5):")
                #     print(num_valid_neighbors_series.head())
                #     print("  Sample Local Factor (Median Neighbor Alpha) (first 5):")
                #     print(local_factor_series.head())
                #     print("  Sample f_reg used in weighting (first 5):") # Added f_reg print
                #     print(f_reg_series.head())
                #     print("  Sample Weight Series (first 5):")
                #     print(weight_series.head())
                #     print("  Sample Combined Factor (Local*W + f_reg*(1-W)) (first 5):")
                #     print(factor_combined_series.head())
                #     print("  Sample Raw Radar used (first 5):") # Added Raw Radar print
                #     print(raw_radar_series.head() if 'Radar_Data_mm_per_min' in df_target.columns else "Raw Radar Missing")
                #     print("  Sample Final_Adjusted_Rainfall (Line 4) (first 5):")
                #     print(final_adjusted_batch_series.head())
                #     print(f"  Count of NaNs in Final_Adjusted_Rainfall for this batch: {final_adjusted_batch_series.isna().sum()}")
                #     print(f"  Count of Zeros in Final_Adjusted_Rainfall for this batch: {(final_adjusted_batch_series == 0).sum()}")

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
            Includes Raw Data, Adjustments, and Batch Flag status
        '''
        ################################################
        print("Saving detailed output CSV files...")
        output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output" # Define a specific subdir
        output_csv_dir.mkdir(parents=True, exist_ok=True) # Create directory

        cols_to_save = [
            'Gauge_Data_mm_per_min',
            'Radar_Data_mm_per_min',         # Raw Radar
            'Batch_Alpha_Adjusted_Radar',    # Step 2 result ("24h Adjusted")
            'Radar_Freg_Adjusted',           # Debug series (Raw * f_reg)
            'Final_Adjusted_Rainfall',       # Step 5 result ("Final Adjusted")
            'Batch_Flag'                     # Batch reliability status
            # Optional: 'Flagged' (timestep flag), 'Alpha', 'Rolling_Abs_Error' etc.
        ]

        for coord, df_final in tqdm(all_data.items(), desc="Saving Output CSVs"):
            coord_str = str(coord)
            # Select only existing columns from the desired list
            save_cols_present = [col for col in cols_to_save if col in df_final.columns]
            if not save_cols_present:
                logger.warning(f"No columns to save for {coord_str}. Skipping CSV.")
                continue

            df_to_save = df_final[save_cols_present].copy()

            # Sanitize coordinate string for filename
            safe_coord = coord_str.replace('(','').replace(')','').replace(',','_').replace(' ','')
            output_filename = output_csv_dir / f"{safe_coord}_data_adjustments.csv"

            try:
                # Save with timezone information
                df_to_save.to_csv(output_filename, date_format='%Y-%m-%d %H:%M:%S%z')
                logger.info(f"Saved detailed data for {coord_str} to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save CSV for {coord_str}: {e}")

        # ---------------------------------------------------------------------
        # 8. Final Aggregation, Saving, and Plotting
        # ---------------------------------------------------------------------
        events_df = None
        all_data_iter0 = None
        logger.info("Generating final visualizations...")
        print("Generating final visualizations...")
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels,
                                        events_df=events_df, all_data_iter0=all_data_iter0)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84,
                                    svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        # print('Calling the create_flagging_plots_dashboard function...')
        # create_flagging_plots_dashboard(all_data, events_df=events_df,
        #                                 output_dir=str(CFG.FLAGGING_PLOTS_DIR),
        #                                 dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))
    
    except Exception as e:
        logger.exception("--- An error occurred during batch adjustment execution ---")
        print("\n--- An error occurred during batch adjustment execution ---")
        traceback.print_exc()
    
    finally:
        logger.info("Batch adjustment process finished.")
        print("\nBatch adjustment process finished.")
        logging.shutdown()


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    main_batch_adjustment()
