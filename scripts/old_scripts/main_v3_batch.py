# main_v2.py

import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging

from config import CFG
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
from network_iterative2 import compute_network_metrics_iterative # Import the corrected function
# Make sure IterationLogFilter is imported IF you use it directly here (not needed if only used in network_iterative)
# from network_iterative import IterationLogFilter
from anomaly import flag_anomalies
from event_detection2 import identify_and_flag_rain_events
from results import aggregate_results
from scripts.old_scripts.plotting3_batches import (create_plots_with_error_markers, generate_html_dashboard,
                      create_flagging_plots_dashboard)

from scripts.old_scripts.plotting3_batches import debug_alpha_for_coord, debug_alpha_and_neighbors_plot

from network import get_nearest_neighbors

from batch_adjustment import compute_regional_adjustment

# --- Configuration for Iteration ---
MAX_ITERATIONS = 3
SAVE_EXCLUSION_LOG = True
EXCLUSION_LOG_CSV = CFG.RESULTS_DIR / 'all_iterative_exclusions.csv'
# ---

# --- Setup Logging ---
# Use a simpler format for the main log, or ensure iteration is passed/defaulted
log_file_main = CFG.RESULTS_DIR / 'iterative_process.log' # Keep separate main log
# Format without iteration for general messages
formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__) # Use main logger name or keep separate
if not logger.handlers: # Setup only once
    logger.setLevel(logging.INFO)
    # Main process file handler
    main_file_handler = logging.FileHandler(log_file_main, mode='w')
    main_file_handler.setFormatter(formatter_main) # Use simple formatter
    logger.addHandler(main_file_handler)

    # Optional Console Handler (using simple formatter)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # console_handler.setFormatter(formatter_main)
    # logger.addHandler(console_handler)
# ---

def initialize_adjustment_columns(df):
    cols_to_init = {
        'Alpha': float,'Rolling_Gauge': float, 'Rolling_Radar': float,
        'Median_Neighbor_Alpha': float, 'Network_Adjusted_Radar': float,
        'Adjusted_Diff_from_network': float, 'Adjusted_Ratio_From_Network': float,
        'Flagged': bool, 'Batch_Flag': bool,
        'Radar_Freg_Adjusted': float, 'Final_Adjusted_Rainfall': float,
        'Rolling_Abs_Error': float, 'Rolling_Prop_Flagged': float
    }
    for col, dtype in cols_to_init.items():
        if col not in df.columns:
            df[col] = pd.Series(dtype=dtype, index=df.index)
    return df



def main_iterative():
    # Log messages from main_iterative (before loop) use the simple format
    logger.info('Starting ITERATIVE anomaly detection process...')
    all_data = {}
    # previous_flagged_event_intervals = {}
    # all_exclusion_logs = []

    all_data_iter0 = {}

    try:
        logger.info("--- Loading Initial Data ---")
        print("--- Loading Initial Data ---") # Keep console feedback
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        if not initial_all_data or not valid_coordinate_locations_utm:
            logger.critical("Fatal: No initial data loaded. Exiting.")
            print("Fatal: No initial data loaded. Exiting.")
            return
        all_data = initial_all_data.copy()
        # print all_data['(660016,6126660)'] for 2024-06-23

        # save all_data['(660016,6126660)'] to csv
        # all_data['(660016,6126660)'].to_csv('all_data_660016_6126660.csv')


        
        
        logger.info(f"Loaded initial data for {len(all_data)} coordinates.")

        # Ensure coordinate keys are strings
        # all_data = {str(k): v for k, v in all_data.items()}
        # coordinate_locations_utm = {str(k): v for k, v in coordinate_locations_utm.items()}
        # valid_coordinate_locations_utm = {str(k): v for k, v in valid_coordinate_locations_utm.items()}



        events_df = pd.DataFrame()
        batch_start = all_data[list(all_data.keys())[0]].index[0]  # Initialize batch_start with the first timestamp of the first coordinate
        for coord, df in all_data.items():
            
            # print(coord)
            # print(df.columns)
            df['Radar_Data_mm_per_min'] = df['Radar Data']
            df['Gauge_Data_mm_per_min'] = df['Gauge Data']
            all_data[coord] = df.copy()  # Ensure we are working with a copy of the DataFrame
        # Generating adjusted values for the 24h batches and batch flagging.
        print("Generating adjusted values for the 24h batches and batch flagging...")
        for coord, df in tqdm(all_data.items()):
            # print(coord)
            # print(df.columns)
            # df['Radar_Data_mm_per_min'] = df['Radar Data']
            # df['Gauge_Data_mm_per_min'] = df['Gauge Data']
            # batch_start = batch_start
            batch_start = df.index.min()

            sensor_end = df.index.max()

            while batch_start < sensor_end:
                batch_end = batch_start + pd.Timedelta(hours = 24)
                batch_mask = (df.index >= batch_start) & (df.index < batch_end)

                df_batch = df.loc[batch_mask]

                median_neighbor_alpha_batch = pd.Series(index=df_batch.index, dtype=float)
                # if coord == '(661433,6131423)':
                #     print('coord:', coord)
                #     print(df.columns)

                if 'Alpha' in df.columns:
                    
                    neighbors = get_nearest_neighbors(coord, coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                    
                    valid_neighbors = [n for n in neighbors if n in all_data and 'Alpha' in all_data[n].columns]
                    if valid_neighbors:
                        neighbor_alphas = {n: all_data[n]['Alpha'].reindex(df_batch.index) for n in valid_neighbors}
                        median_neighbor_alpha_batch = pd.DataFrame(neighbor_alphas).median(axis=1, skipna=True)
                        median_neighbor_alpha_batch = median_neighbor_alpha_batch.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)
                        if 'Radar_Data_mm_per_min' in df.columns:
                            adjusted_radar = df.loc[batch_mask, 'Radar_Data_mm_per_min'] * median_neighbor_alpha_batch
                            df.loc[batch_mask, 'Network_Adjusted_Radar'] = adjusted_radar

                            # Batch flagging
                            if 'Gauge_Data_mm_per_min' in df.columns:
                                gauge_values = df_batch['Gauge_Data_mm_per_min']
                                abs_error = abs(adjusted_radar - gauge_values)
                                rolling_abs_error = abs_error.rolling('60min', closed='both').mean()
                                
                                ratio = adjusted_radar / (gauge_values + CFG.EPSILON)

                                flag_points = ((ratio > 1 + CFG.RATIO_THRESHOLD) | (ratio < 1 - CFG.RATIO_THRESHOLD)).astype(float)
                                rolling_prop_flagged = flag_points.rolling('60min', closed='both').mean()

                                cond = (rolling_abs_error > CFG.ABS_DIFF_THRESHOLD_MM_MIN) & (rolling_prop_flagged > 0.7)
                                cond = ((rolling_abs_error > CFG.ABS_DIFF_THRESHOLD_MM_MIN) & (rolling_prop_flagged > 0.7)).fillna(0)

                                # Compute the proportion (percentage) of timepoints in this batch where the condition holds:
                                if len(cond) > 0:
                                    percentage_true = cond.sum() / len(cond)
                                else:
                                    percentage_true = 0
                                    # print(f"Warning: No valid data points in batch for {coord} from {batch_start} to {batch_end}.")

                                if percentage_true >= 0.30:
                                    batch_flag = True
                                    # print(f"Batch flagged for {coord} from {batch_start} to {batch_end} ({percentage_true:.2%} of points meet conditions).")
                                else:
                                    batch_flag = False
                                df.loc[batch_mask, 'Batch_Flag'] = batch_flag
                            else:
                                df.loc[batch_mask, 'Batch_Flag'] = False
                else:
                    df.loc[batch_mask, 'Batch_Flag'] = False
                    df.loc[batch_mask, 'Network_Adjusted_Radar'] = False
                # df['Rolling_Abs_Error'] = abs_error.rolling('60min', closed='both').mean()
                # df['Rolling_Prop_Flagged'] = flag_points.rolling('60min', closed='both').mean()
                df.loc[batch_mask, 'Median_Neighbor_Alpha'] = median_neighbor_alpha_batch



                


                f_reg, valid_count = compute_regional_adjustment(all_data, batch_start, batch_end)
                # print(f"Batch {batch_start} - {batch_end}: f_reg = {f_reg:.3f} using {valid_count} valid gauges.")

                if 'Radar_Data_mm_per_min' in df.columns:
                    final_adjusted = df.loc[batch_mask, 'Radar_Data_mm_per_min'] * f_reg
                    df.loc[batch_mask, 'Final_Adjusted_Rainfall'] = final_adjusted

                batch_start = batch_end

                # print(df['Rolling_Abs_Error'])
                # print(df['Rolling_Prop_Flagged'])
            if 'Network_Adjusted_Radar' in df and 'Gauge_Data_mm_per_min' in df:
                # print(f"Calculating errors for {coord}...")
                abs_error = abs(df['Network_Adjusted_Radar'] - df['Gauge_Data_mm_per_min'])
                # Compute the 60-minute rolling average of the absolute error
                df['Rolling_Abs_Error'] = abs_error.rolling('60min', closed='both').mean()
                
                # Compute the ratio at each time point (adding a small epsilon for safety)
                ratio = df['Network_Adjusted_Radar'] / (df['Gauge_Data_mm_per_min'] + CFG.EPSILON)
                # Create a binary series: 1 if the ratio falls outside the acceptable range, 0 otherwise
                flag_points = ((ratio > 1 + CFG.RATIO_THRESHOLD) | (ratio < 1 - CFG.RATIO_THRESHOLD)).astype(float)
                # Compute the 60-minute rolling average to get the proportion flagged
                df['Rolling_Prop_Flagged'] = flag_points.rolling('60min', closed='both').mean()
                # print(f"Max Rolling Abs Error for {coord}: {df['Rolling_Abs_Error'].max()}")
                # print(f"Max Rolling Prop Flagged for {coord}: {df['Rolling_Prop_Flagged'].max()}")
            
            # print('neighbors for coord:', coord, neighbors)
            all_data[coord] = df



        from results import aggregate_results  # Ensure this function is imported

        logger.info("Aggregating final results for summary CSV...")
        print("Aggregating final results...")

        results_df = aggregate_results(all_data)
        if not results_df.empty:
            try:
                summary_csv_path = str(CFG.SUMMARY_CSV_FILE)
                results_df.to_csv(summary_csv_path, index=False)
                logger.info(f"Final summary saved to {summary_csv_path}")
                print(f"Final summary saved to {summary_csv_path}")
            except Exception as e:
                logger.error(f"Error saving summary CSV: {e}")
                print(f"Error saving summary CSV: {e}")
        else:
            logger.warning("No results to aggregate; summary CSV not created.")
            print("No results to aggregate; summary CSV not created.")

        # Generate Visualizations
        logger.info("Generating final visualizations...")
        print("Generating final visualizations...")
        # ... (visualization logic remains the same) ...
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels, events_df=events_df, all_data_iter0=all_data_iter0)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84, svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        print('calling the create_flagging_plots_dashboard.....................................')
        # create_flagging_plots_dashboard(all_data, events_df=events_df, output_dir=str(CFG.FLAGGING_PLOTS_DIR), dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))
        create_flagging_plots_dashboard(
            all_data, 
            events_df=events_df, 
            output_dir=str(CFG.FLAGGING_PLOTS_DIR), 
            dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE)
        )
        logger.info("Finished final visualizations.")
        print("Finished final visualizations.")

        # Optional Debug Plot
        # ...
        debug_coord = '(675616,6122248)'
        if debug_coord in all_data:
            fig_debug = debug_alpha_for_coord(debug_coord, all_data, valid_coordinate_locations_utm)
            fig_debug.write_html(str(CFG.DEBUG_ALPHA_PLOT_FILE), full_html=True)
            logger.info(f"Debug plot saved: {CFG.DEBUG_ALPHA_PLOT_FILE}")
            print(f"Debug plot saved: {CFG.DEBUG_ALPHA_PLOT_FILE}")
        else:
            logger.warning(f"Skipping debug plot: {debug_coord} not in loaded data.")
            print(f"Skipping debug plot: {debug_coord} not in loaded data.")

        if debug_coord in all_data:
            fig_debug_neighbors = debug_alpha_and_neighbors_plot(coord=debug_coord, all_data=all_data,
                                                                 coordinate_locations=valid_coordinate_locations_utm,
                                                                 n_neighbors=CFG.N_NEIGHBORS)
            debug_plot_path = CFG.DASHBOARD_DIR / f"debug_neighbors_{debug_coord}.html"
            fig_debug_neighbors.write_html(str(debug_plot_path), full_html=True)
            logger.info(f"Neighbor debug plot saved: {debug_plot_path}")
            print(f"Neighbor debug plot saved: {debug_plot_path}")
        else:
            logger.warning(f"Skipping neighbor debug plot: {debug_coord} not in loaded data.")
            print(f"Skipping neighbor debug plot: {debug_coord} not in loaded data.")

    except Exception as e:
        logger.exception("--- An error occurred during iterative execution ---")
        print("\n--- An error occurred during iterative execution ---")
        traceback.print_exc()
    finally:
        logger.info("Iterative anomaly detection process finished.")
        print("\nIterative anomaly detection process finished.")
        logging.shutdown()

if __name__ == "__main__":
    main_iterative()