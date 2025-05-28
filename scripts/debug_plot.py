# debug_plot.py
import argparse
import pandas as pd
from pathlib import Path

from config import CFG
from data_loading import load_target_coordinates, load_sensor_data, process_sensor_metadata, load_time_series_data
from feature_engineering import apply_feature_engineering
from network import compute_network_metrics, get_nearest_neighbors
from plotting import debug_alpha_for_coord, debug_alpha_and_neighbors_plot

def main():
    parser = argparse.ArgumentParser(description="Generate debug plots for a chosen coordinate.")
    parser.add_argument("--coord", required=True,
                        help="Coordinate string to debug (e.g. \"(675616,6122248)\").")
    args = parser.parse_args()
    chosen_coord = args.coord

    # Load target coordinates, sensor metadata, and time series data.
    print("Loading target coordinates...")
    target_coords, coordinate_locations_utm = load_target_coordinates()

    print("Loading sensor metadata...")
    sensordata, gdf_wgs84 = load_sensor_data()

    print("Processing sensor metadata...")
    sensor_channels, svk_coords_utm = process_sensor_metadata(sensordata)

    print("Loading time series data...")
    all_data, valid_coordinate_locations_utm, missing_files = load_time_series_data(target_coords, coordinate_locations_utm)
    if chosen_coord not in all_data:
        print(f"Coordinate {chosen_coord} not found in loaded data.")
        return

    # Apply feature engineering.
    print("Applying feature engineering...")
    all_data = apply_feature_engineering(all_data)

    # Compute network metrics.
    print("Computing network metrics...")
    all_data = compute_network_metrics(all_data, valid_coordinate_locations_utm)

    # Generate debug plots for the chosen coordinate.
    print(f"Generating debug plots for coordinate {chosen_coord}...")
    fig_debug = debug_alpha_for_coord(chosen_coord, all_data, valid_coordinate_locations_utm)
    fig_debug_neighbors = debug_alpha_and_neighbors_plot(chosen_coord, all_data, valid_coordinate_locations_utm)

    # Save debug plots to a folder named "debug_plots".
    debug_folder = Path(CFG.BASE_DIR) / "debug_plots"
    debug_folder.mkdir(parents=True, exist_ok=True)
    debug_file = debug_folder / f"debug_{chosen_coord}.html"
    debug_neighbors_file = debug_folder / f"debug_neighbors_{chosen_coord}.html"

    fig_debug.write_html(str(debug_file), full_html=True)
    fig_debug_neighbors.write_html(str(debug_neighbors_file), full_html=True)

    print(f"Debug plot saved to {debug_file}")
    print(f"Neighbor debug plot saved to {debug_neighbors_file}")

if __name__ == "__main__":
    main()
