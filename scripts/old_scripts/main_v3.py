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
from scripts.old_scripts.plotting3 import (create_plots_with_error_markers, generate_html_dashboard,
                      create_flagging_plots_dashboard)

from scripts.old_scripts.plotting3 import debug_alpha_for_coord, debug_alpha_and_neighbors_plot

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

def main_iterative():
    # Log messages from main_iterative (before loop) use the simple format
    logger.info('Starting ITERATIVE anomaly detection process...')
    all_data = {}
    previous_flagged_event_intervals = {}
    all_exclusion_logs = []

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
        logger.info(f"Loaded initial data for {len(all_data)} coordinates.")

        events_df = pd.DataFrame()

        # --- Iteration Loop ---
        for iteration in range(MAX_ITERATIONS):
            # Log iteration start using the main logger (no iteration number needed in format)
            logger.info(f"--- Starting Iteration {iteration} ---")
            print(f"\n--- Starting Iteration {iteration} ---")

            # --- Step 2: Feature Engineering ---
            if iteration == 0:
                 logger.info("Applying feature engineering...")
                 print("Applying feature engineering...")
                 all_data = apply_feature_engineering(all_data)

            # --- Step 3: Network Metrics ---
            logger.info(f"Computing network metrics (Iteration {iteration})...")
            # This function call will use its own logger setup with the iteration number
            all_data, current_exclusion_log = compute_network_metrics_iterative(
                all_data,
                valid_coordinate_locations_utm,
                previous_flagged_event_intervals,
                current_iteration=iteration
            )
            all_exclusion_logs.extend(current_exclusion_log)
            logger.info(f"Finished network metrics for iteration {iteration}.")



            # --- Step 4: Flag Anomalies ---
            logger.info(f"Flagging anomalies (Iteration {iteration})...")
            print(f"Flagging anomalies (Iteration {iteration})...")
            all_data = flag_anomalies(all_data)
            logger.info(f"Finished flagging for iteration {iteration}.")

                                    # >>>>>>>>>> Store Iteration 0 specific results <<<<<<<<<<
            if iteration == 0:
                logger.info("Storing results from Iteration 0 for later plotting...")
                for coord, df_iter0 in all_data.items():
                    # Store only the columns needed for comparison plotting
                    cols_to_store = ['Network_Adjusted_Radar', 'Flagged'] # Add others if needed, e.g., 'Median_Neighbor_Alpha'
                    present_cols = [col for col in cols_to_store if col in df_iter0.columns]
                    # print(present_cols)
                    if present_cols:
                         # Create a small DataFrame or Series with just these columns
                         # Use .copy() to ensure it's independent
                         all_data_iter0[coord] = df_iter0[present_cols].copy()
                logger.info("Finished storing Iteration 0 results.")
            # >>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # --- Step 5: Identify Events ---
            logger.info(f"Identifying rain events (Iteration {iteration})...")
            print(f"Identifying rain events (Iteration {iteration})...")
            rain_events = identify_and_flag_rain_events(all_data)
            logger.info(f"Finished event identification for iteration {iteration}.")

            current_flagged_event_intervals = {}
            events_df = pd.DataFrame()
            flagged_events_df = pd.DataFrame()

            if rain_events:
                events_df = pd.DataFrame(rain_events)
                events_df.sort_values(by=["sensor_coord", "event_start"], inplace=True)
                flagged_events_df = events_df[events_df['is_flagged_event']].copy()

                for coord, group in flagged_events_df.groupby('sensor_coord'):
                    current_flagged_event_intervals[coord] = sorted([
                        (row['event_start'], row['event_end'])
                        for _, row in group.iterrows()
                    ])

                log_msg_events = f"Iteration {iteration}: Found {len(events_df)} total events, {len(flagged_events_df)} flagged events."
                logger.info(log_msg_events)
                print(log_msg_events)
                # Optional saving per iteration
                # ...
            else:
                log_msg_no_events = f"Iteration {iteration}: No rain events identified."
                logger.info(log_msg_no_events)
                print(log_msg_no_events)

            # --- Prepare for next iteration ---
            if iteration > 0 and current_flagged_event_intervals == previous_flagged_event_intervals:
                log_msg_converge = f"Convergence reached at iteration {iteration}. Stopping."
                logger.info(log_msg_converge)
                print(log_msg_converge)
                break

            previous_flagged_event_intervals = current_flagged_event_intervals

        # --- End of Iteration Loop ---
        logger.info("--- Iterative Process Finished ---")
        print("\n--- Iterative Process Finished ---")

        # --- Final Steps ---

        # Save Exclusion Log
        if SAVE_EXCLUSION_LOG and all_exclusion_logs:
            # ... (saving logic remains the same) ...
             logger.info(f"Saving exclusion log to {EXCLUSION_LOG_CSV}...")
             print(f"Saving exclusion log to {EXCLUSION_LOG_CSV}...")
             # ... rest of saving ...

        # Print Min Neighbor Summary
        logger.info("--- Min Neighbor Count During Events (Final Iteration) ---")
        if not events_df.empty:
            # ... (printing logic remains the same) ...
            print("\n" + "="*80)
            print("Summary of Minimum Neighbors Used During Events (Final Iteration)")
            print("="*80)
            # ... rest of printing ...
        else:
            logger.info("No events identified in the final iteration to report neighbor counts for.")
            print("\nNo events identified in the final iteration to report neighbor counts for.\n")

        # Aggregate & Save Results
        logger.info("Aggregating final results...")
        print("Aggregating final results...")
        results_df = aggregate_results(all_data)
        if not results_df.empty:
            try:
                results_df.to_csv(CFG.SUMMARY_CSV_FILE, index=False)
                logger.info(f"Final summary saved to {CFG.SUMMARY_CSV_FILE}")
                print(f"Final summary saved to {CFG.SUMMARY_CSV_FILE}")
            except Exception as e:
                logger.error(f"Error saving summary CSV: {e}")
                print(f"Error saving summary CSV: {e}")
        # ... (aggregation and saving logic remains the same) ...

        # Save Final Events
        if 'events_df' in locals() and not events_df.empty:
            final_events_file = CFG.EVENTS_CSV_FILE
            logger.info(f"Saving final events to {final_events_file}...")
            print(f"Saving final events to {final_events_file}...")
            try:
                CFG.RESULTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists
                events_df.to_csv(final_events_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
                logger.info(f"Final events saved to {final_events_file}")
                print(f"Final events saved to {final_events_file}")
            except Exception as e:
                logger.error(f"Error saving events CSV: {e}")
                print(f"Error saving events CSV: {e}")
             # ... (saving logic remains the same) ...

            # logger.info(f"Final events summary saved to {CFG.EVENTS_CSV_FILE}")
            # print(f"Final events summary saved to {CFG.EVENTS_CSV_FILE}")
             # ... rest of saving ...
            final_flagged_events_df = events_df[events_df['is_flagged_event']].copy()
            if not final_flagged_events_df.empty:
                flagged_events_file = CFG.FLAGGED_EVENTS_CSV_FILE
                logger.info(f"Saving flagged events to {flagged_events_file}...")
                print(f"Saving flagged events to {flagged_events_file}...")
                try:
                    final_flagged_events_df.to_csv(flagged_events_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
                    logger.info(f"Flagged events saved to {flagged_events_file}")
                    print(f"Flagged events saved to {flagged_events_file}")
                except Exception as e:
                    logger.error(f"Error saving flagged events CSV: {e}")
                    print(f"Error saving flagged events CSV: {e}")
            else:
                logger.warning("No flagged events identified in the final iteration.")
                print("No flagged events identified in the final iteration.")
            
        else:
             logger.warning("No events identified in the final iteration. CSV files not updated.")

        # Generate Visualizations
        logger.info("Generating final visualizations...")
        print("Generating final visualizations...")
        # ... (visualization logic remains the same) ...
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels, events_df=events_df, all_data_iter0=all_data_iter0)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84, svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        print('calling the create_flagging_plots_dashboard.....................................')
        create_flagging_plots_dashboard(all_data, events_df=events_df, output_dir=str(CFG.FLAGGING_PLOTS_DIR), dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))
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