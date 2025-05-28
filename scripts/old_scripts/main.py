# main.py
import pandas as pd
from tqdm import tqdm
import traceback

from config import CFG
from utils import shift_radar_by_dst
from feature_engineering import apply_feature_engineering
from network import compute_network_metrics
from anomaly import flag_anomalies
from scripts.old_scripts.event_detection import identify_and_flag_rain_events
from results import aggregate_results
from plotting import (create_plots_with_error_markers, generate_html_dashboard,
                      create_flagging_plots_dashboard, debug_alpha_for_coord, debug_alpha_and_neighbors_plot)
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)

def main():
    print('Starting anomaly detection process...')
    events_df = pd.DataFrame()
    try:
        # Step 1: Load Data
        target_coords, coordinate_locations_utm = load_target_coordinates()
        sensordata, gdf_wgs84 = load_sensor_data()
        sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)
        all_data, valid_coordinate_locations_utm, _ = load_time_series_data(target_coords, coordinate_locations_utm)
        if not all_data or not valid_coordinate_locations_utm:
            return

        # Step 2: Feature Engineering
        all_data = apply_feature_engineering(all_data)
        # Compute adjusted radar using median neighbor alpha
        for coord in tqdm(all_data.keys()):
            df = all_data[coord]
            neighbors = []  # Will be computed inside network metrics
            median_neighbor_alpha = pd.Series(dtype=float)
            if 'Alpha' in df:
                from network import get_nearest_neighbors
                neighbors = get_nearest_neighbors(coord, coordinate_locations_utm, n_neighbors=CFG.N_NEIGHBORS)
                valid_neighbors = [n for n in neighbors if n in all_data and 'Alpha' in all_data[n].columns]
                if valid_neighbors:
                    neighbor_alphas = {n: all_data[n]['Alpha'].reindex(df.index) for n in valid_neighbors}
                    median_neighbor_alpha = pd.DataFrame(neighbor_alphas).median(axis=1)
                    # median_neighbor_alpha = median_neighbor_alpha.fillna(method=CFG.FILLNA_METHOD, limit=CFG.FILLNA_LIMIT)\
                    #                                          .fillna(method='bfill', limit=CFG.FILLNA_LIMIT)
                    median_neighbor_alpha = median_neighbor_alpha.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)

                    if not median_neighbor_alpha.isnull().all() and 'Radar_Data_mm_per_min' in df:
                        adjusted_radar = df['Radar_Data_mm_per_min'] * median_neighbor_alpha
                        df['Adjusted_Radar'] = adjusted_radar
            df['Median_Neighbor_Alpha'] = median_neighbor_alpha
            all_data[coord] = df

        # Step 3: Network Metrics
        all_data = compute_network_metrics(all_data, valid_coordinate_locations_utm)
        # Step 4: Flag Anomalies
        all_data = flag_anomalies(all_data)
        # Step 5: Identify Events
        rain_events = identify_and_flag_rain_events(all_data)
        if rain_events:
            events_df = pd.DataFrame(rain_events).sort_values(by=["sensor_coord", "event_start"])
            events_df.to_csv(CFG.EVENTS_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
            flagged_events_df = events_df[events_df['is_flagged_event']]
            if not flagged_events_df.empty:
                flagged_events_df.to_csv(CFG.FLAGGED_EVENTS_CSV_FILE, index=False, date_format='%Y-%m-%d %H:%M:%S')
            else:
                print("No events met flag threshold.")
        else:
            print("No rain events identified.")

        # Optional: Save processed data to CSV
        save_csv = False
        if save_csv:
            for coord, df in all_data.items():
                df.to_csv(CFG.BASE_DIR / CFG.SENSOR_CSV_OUTPUT_TPL.format(coord=coord))

        # Step 6: Aggregate Summary
        results_df = aggregate_results(all_data)
        if not results_df.empty:
            results_df.to_csv(CFG.SUMMARY_CSV_FILE, index=False)
            print(f"Summary saved to {CFG.SUMMARY_CSV_FILE}")

        # Step 7: Generate Visualizations
        create_plots_with_error_markers(all_data, valid_coordinate_locations_utm, sensor_channels, events_df=events_df)
        if not gdf_wgs84.empty:
            generate_html_dashboard(all_data, valid_coordinate_locations_utm, gdf_wgs84, svk_coords_utm, output_file=str(CFG.DASHBOARD_FILE))
        else:
            print("Skipping main dashboard: Missing sensor metadata.")
        create_flagging_plots_dashboard(all_data, events_df=events_df, output_dir=str(CFG.FLAGGING_PLOTS_DIR), dashboard_file=str(CFG.FLAGGING_DASHBOARD_FILE))

        # Step 8: Debugging Plots
        debug_coord = '(675616,6122248)'
        if debug_coord in all_data:
            fig_debug = debug_alpha_for_coord(debug_coord, all_data, valid_coordinate_locations_utm)
            fig_debug.write_html(str(CFG.DEBUG_ALPHA_PLOT_FILE), full_html=True)
            print(f"Debug plot saved: {CFG.DEBUG_ALPHA_PLOT_FILE}")
        if debug_coord in all_data:
            fig_debug_neighbors = debug_alpha_and_neighbors_plot(coord=debug_coord, all_data=all_data,
                                                                 coordinate_locations=valid_coordinate_locations_utm,
                                                                 n_neighbors=CFG.N_NEIGHBORS)
            debug_plot_path = CFG.DASHBOARD_DIR / f"debug_neighbors_{debug_coord}.html"
            fig_debug_neighbors.write_html(str(debug_plot_path), full_html=True)
            print(f"Neighbor debug plot saved: {debug_plot_path}")
        else:
            print(f"Skipping neighbor debug plot: {debug_coord} not in loaded data.")

    except Exception as e:
        print("\n--- An error occurred during execution ---")
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        print("\nAnomaly detection process finished.")

if __name__ == "__main__":
    main()
