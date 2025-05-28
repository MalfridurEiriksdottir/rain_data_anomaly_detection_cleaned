# iterative_main.py
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np

from config import CFG
# Import necessary functions from your existing modules
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import apply_feature_engineering
# We'll need a modified network calculation
# from network import compute_network_metrics, get_nearest_neighbors
from scripts.old_scripts.network_iterative import compute_network_metrics_iterative # Hypothetical modified module/function
from anomaly import flag_anomalies # Or flag_anomalies_v2 if you activate it
from scripts.old_scripts.event_detection import identify_and_flag_rain_events
from results import aggregate_results
# Import plotting if needed per iteration or just at the end
from plotting import (create_plots_with_error_markers, generate_html_dashboard,
                      create_flagging_plots_dashboard)
import logging
# --- Configuration for Iteration ---
MAX_ITERATIONS = 3 # Example: Run initial + 2 refinement iterations
SAVE_EXCLUSION_LOG = True # Flag to control saving the log
EXCLUSION_LOG_CSV = CFG.RESULTS_DIR / 'all_iterative_exclusions.csv'
# ---
log_file = CFG.RESULTS_DIR / 'iterative_process.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w' # 'w' to overwrite each run
)
logger = logging.getLogger(__name__)

def main_iterative():
    print('Starting ITERATIVE anomaly detection process...')
    logger.info('Starting ITERATIVE anomaly detection process...')
    all_data = {} # Holds the data state for the current iteration
    previous_flagged_event_intervals = {} # Dict: coord -> list of (start_time, end_time) tuples
    all_exclusion_logs = [] # Store logs from all iterations

    try:
        # --- Step 1: Initial Data Load (Done ONCE) ---
        logger.info("--- Loading Initial Data ---")
        print("--- Loading Initial Data ---")
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)
        initial_all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        if not initial_all_data or not valid_coordinate_locations_utm:
            logger.critical("Fatal: No initial data loaded. Exiting.")
            print("Fatal: No initial data loaded. Exiting.")
            return

        all_data = initial_all_data.copy() # Start with fresh data for iteration 0
        logger.info(f"Loaded initial data for {len(all_data)} coordinates.")

        events_df = pd.DataFrame()

        # --- Iteration Loop ---
        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Starting Iteration {iteration} ---")
            logger.info(f"--- Starting Iteration {iteration} ---")

            # --- Step 2: Feature Engineering ---
            # Re-run if features might change based on input data,
            # otherwise could be done once if only network metrics change.
            # Assuming features depend only on raw Gauge/Radar, run once is fine.
            # If features depend on network outputs, recalculate here.
            if iteration == 0: # Run only once if features are independent of iteration
                 print("Applying feature engineering...")
                 logger.info("Applying feature engineering...")
                 all_data = apply_feature_engineering(all_data)

            # --- Step 3: Network Metrics (MODIFIED) ---
            print(f"Computing network metrics (Iteration {iteration})...")
            logger.info(f"Computing network metrics (Iteration {iteration})...")
            # Pass the flagged intervals from the *previous* iteration
            all_data = compute_network_metrics_iterative(
                all_data,
                valid_coordinate_locations_utm,
                previous_flagged_event_intervals # Key change: provide exclusion info
            )

            # --- Step 4: Flag Anomalies ---
            print(f"Flagging anomalies (Iteration {iteration})...")
            all_data = flag_anomalies(all_data) # Use your chosen flagging function

            # --- Step 5: Identify Events & Store Flagged Intervals ---
            print(f"Identifying rain events (Iteration {iteration})...")
            rain_events = identify_and_flag_rain_events(all_data)

            current_flagged_event_intervals = {}
            if rain_events:
                events_df = pd.DataFrame(rain_events)
                flagged_events_df = events_df[events_df['is_flagged_event']].copy()

                # Store intervals for the *next* iteration's exclusion
                for coord, group in flagged_events_df.groupby('sensor_coord'):
                    current_flagged_event_intervals[coord] = [
                        (row['event_start'], row['event_end'])
                        for _, row in group.iterrows()
                    ]

                print(f"Iteration {iteration}: Found {len(flagged_events_df)} flagged events.")
                # Optional: Save results per iteration
                # events_df.to_csv(CFG.RESULTS_DIR / f'events_iter_{iteration}.csv', index=False)
                # flagged_events_df.to_csv(CFG.RESULTS_DIR / f'flagged_events_iter_{iteration}.csv', index=False)
            else:
                print(f"Iteration {iteration}: No rain events identified.")

            # --- Prepare for next iteration ---
            # Check for convergence (optional)
            if iteration > 0 and current_flagged_event_intervals == previous_flagged_event_intervals:
                print(f"Convergence reached at iteration {iteration}. Stopping.")
                break

            previous_flagged_event_intervals = current_flagged_event_intervals



        print("--- Neighbor Counts Used Per Event (Final Iteration) ---")
        if 'events_df' in locals() and not events_df.empty:
            # Select relevant columns for printing
            cols_to_print = [
                'event_start', 'event_end', 'duration_minutes',
                'is_flagged_event', 'min_neighbors_used',
                'avg_neighbors_used', 'max_neighbors_used'
            ]
            # Ensure all desired columns exist before trying to select
            available_cols = [col for col in cols_to_print if col in events_df.columns]

            if not available_cols:
                 print("Could not find event columns to print neighbor counts.")
            else:
                print("\n" + "="*80)
                print("Summary of Neighbor Counts Used During Events (Final Iteration)")
                print("="*80)

                # Group by sensor coordinate
                # Use sort=False if you want to keep the order from events_df if it matters
                grouped_events = events_df.groupby('sensor_coord', sort=True)

                for coord, group in grouped_events:
                    print(f"\nSensor Coordinate: {coord}")
                    print("-" * (len(coord) + 20))
                    # Print the selected columns neatly - using to_string for alignment
                    # formatters can be used for specific float precision if needed
                    print(group[available_cols].to_string(index=False, float_format="%.2f", na_rep="N/A"))

                print("\n" + "="*80)
                print("End of Neighbor Count Summary")
                print("="*80 + "\n")

        else:
            # print(("No events identified in the final iteration to report neighbor counts for.")
            print("\nNo events identified in the final iteration to report neighbor counts for.\n")
        # >>>>>>>>>> END OF ADDED SECTION <<<<<<<<<<

        # --- End of Iteration Loop ---
        print("\n--- Iterative Process Finished ---")

        # --- Final Steps (using results from LAST iteration) ---
        print("Aggregating final results...")
        results_df = aggregate_results(all_data)
        if not results_df.empty:
            results_df.to_csv(CFG.SUMMARY_CSV_FILE, index=False) # Overwrite with final summary
            print(f"Final summary saved to {CFG.SUMMARY_CSV_FILE}")

        if rain_events: # Save final events
             events_df.to_csv(CFG.EVENTS_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
             flagged_events_df_final = events_df[events_df['is_flagged_event']]
             if not flagged_events_df_final.empty:
                 flagged_events_df_final.to_csv(CFG.FLAGGED_EVENTS_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')

        print("Generating final visualizations...")
        # Use the final 'all_data' and 'events_df'
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels, events_df=events_df)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84, svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        create_flagging_plots_dashboard(all_data, events_df=events_df, output_dir=str(CFG.FLAGGING_PLOTS_DIR), dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))

    except Exception as e:
        print("\n--- An error occurred during iterative execution ---")
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        print("\nIterative anomaly detection process finished.")

if __name__ == "__main__":
    main_iterative()