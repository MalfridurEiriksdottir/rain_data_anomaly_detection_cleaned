# network.py
import numpy as np
import pandas as pd
from math import sqrt
from scipy.spatial import KDTree
from config import CFG
import tqdm as tqdm

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

def calculate_utm_distance(coord1, coord2):
    """Calculate Euclidean distance between two projected (UTM) coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def build_kdtree(coordinate_locations: dict):
    """Build a KDTree from coordinate_locations dict (sensor name -> (x, y))."""
    names = list(coordinate_locations.keys())
    coords = [coordinate_locations[name] for name in names]
    tree = KDTree(coords)
    return names, tree

def get_nearest_neighbors(coord_name: str, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> list:
    """Get nearest neighbors using a KDTree for fast lookup."""

    names, tree = build_kdtree(coordinate_locations)
    try:
        index = names.index(coord_name)
    except ValueError:
        return []
    # Query for n_neighbors+1 because the point itself is returned.
    distances, indices = tree.query(coordinate_locations[coord_name], k=n_neighbors+1)
    neighbors = [names[i] for i in indices if names[i] != coord_name]

    return neighbors[:n_neighbors]

def compute_network_metrics(all_data: dict, coordinate_locations: dict) -> dict:
    """
    Compute network metrics including Median Neighbor Alpha and Adjusted Radar.
    Uses vectorized operations and the optimized neighbor search.
    """
    print("Computing network metrics...")
    # Determine a common index across all DataFrames
    common_index = pd.Index([])
    for df in all_data.values():
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            common_index = common_index.union(df.index)
    common_index = common_index.sort_values()

    def safe_reindex(df, col, index):
        return df[col].reindex(index) if col in df.columns else pd.Series(np.nan, index=index)

    all_alphas = pd.DataFrame({
        coord: safe_reindex(df, 'Alpha', common_index)
        for coord, df in all_data.items() if 'Alpha' in df.columns
    })
    all_radar_data = pd.DataFrame({
        coord: safe_reindex(df, 'Radar_Data_mm_per_min', common_index)
        for coord, df in all_data.items() if 'Radar_Data_mm_per_min' in df.columns
    })
    all_rolling_diffs = pd.DataFrame({
        coord: safe_reindex(df, 'Rolling_Diff', common_index)
        for coord, df in all_data.items() if 'Rolling_Diff' in df.columns
    })

    processed_data = {}
    for coord in tqdm.tqdm(all_data.keys()):
        df = all_data[coord].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: Skipping network metrics for {coord}, invalid index type.")
            processed_data[coord] = df
            continue

        neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=CFG.N_NEIGHBORS)
        if not neighbors:
            df['Median_Neighbor_Alpha'] = np.nan
            df['Network_Adjusted_Radar'] = np.nan
            df['Network_Avg_Diff'] = np.nan
            df['Difference_From_Network'] = np.nan
            df['Ratio_From_Network'] = np.nan
            df['Median_Ratio_From_Network'] = np.nan
            df['Network_Avg_Alpha'] = np.nan
            df['Alpha_From_Network'] = np.nan
            processed_data[coord] = df
            continue

        valid_neighbors = [n for n in neighbors if n in all_alphas.columns]
        if not valid_neighbors:
            print(f"Warning: Neighbors found for {coord} but none have Alpha data.")
            df['Median_Neighbor_Alpha'] = np.nan
            df['Network_Adjusted_Radar'] = np.nan
            processed_data[coord] = df
            continue

        neighbor_alphas_subset = all_alphas[valid_neighbors]
        median_neighbor_alpha = neighbor_alphas_subset.median(axis=1)
        # median_neighbor_alpha = median_neighbor_alpha.fillna(method=CFG.FILLNA_METHOD, limit=CFG.FILLNA_LIMIT)\
        #                                              .fillna(method='bfill', limit=CFG.FILLNA_LIMIT)
        median_neighbor_alpha = median_neighbor_alpha.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)
        epsilon = 0.01
        network_adjusted_radar = (all_radar_data[coord]+epsilon) * median_neighbor_alpha if coord in all_radar_data.columns else pd.Series(np.nan, index=common_index)
        df['Median_Neighbor_Alpha'] = median_neighbor_alpha.reindex(df.index)
        df['Network_Adjusted_Radar'] = network_adjusted_radar.reindex(df.index)

        # Calculate old network metrics if data available
        if coord in all_rolling_diffs.columns and not all_rolling_diffs[valid_neighbors].empty:
            network_avg_diff = all_rolling_diffs[valid_neighbors].mean(axis=1)
            sensor_rolling_diff = all_rolling_diffs[coord]
            diff_from_network = sensor_rolling_diff - network_avg_diff
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_from_network = (sensor_rolling_diff + CFG.K_PARAM) / (network_avg_diff.abs() + CFG.K_PARAM)
                ratio_from_network = ratio_from_network.replace([np.inf, -np.inf], np.nan)
            df['Network_Avg_Diff'] = network_avg_diff.reindex(df.index)
            df['Difference_From_Network'] = diff_from_network.reindex(df.index)
            df['Ratio_From_Network'] = ratio_from_network.reindex(df.index)
        else:
            df['Network_Avg_Diff'] = np.nan
            df['Difference_From_Network'] = np.nan
            df['Ratio_From_Network'] = np.nan

        # Additional alpha network metric
        if coord in all_alphas.columns and not all_alphas[valid_neighbors].empty:
            network_avg_alpha = all_alphas[valid_neighbors].mean(axis=1)
            sensor_alpha = all_alphas[coord]
            with np.errstate(divide='ignore', invalid='ignore'):
                alpha_from_network = sensor_alpha / (network_avg_alpha + CFG.EPSILON)
                alpha_from_network = alpha_from_network.replace([np.inf, -np.inf], np.nan)
            df['Network_Avg_Alpha'] = network_avg_alpha.reindex(df.index)
            df['Alpha_From_Network'] = alpha_from_network.reindex(df.index)
        else:
            df['Network_Avg_Alpha'] = np.nan
            df['Alpha_From_Network'] = np.nan

        # Compute rolling metrics for adjusted radar and Gauge_Data_mm_per_min
        rolling_window = CFG.ROLLING_WINDOW
        # rolling_adj_radar = df['Network_Adjusted_Radar'].rolling(rolling_window, center=True, min_periods=1).mean()\
        #                     .fillna(method=CFG.FILLNA_METHOD, limit=CFG.FILLNA_LIMIT)\
        #                     .fillna(method='bfill', limit=CFG.FILLNA_LIMIT)
        # rolling_gauge = df['Gauge_Data_mm_per_min'].rolling(rolling_window, center=True, min_periods=1).mean()\
        #                 .fillna(method=CFG.FILLNA_METHOD, limit=CFG.FILLNA_LIMIT)\
        #                 .fillna(method='bfill', limit=CFG.FILLNA_LIMIT)
        rolling_adj_radar = df['Network_Adjusted_Radar'].rolling(rolling_window, center=True, min_periods=1).mean()\
                            .ffill(limit=CFG.FILLNA_LIMIT)\
                            .bfill(limit=CFG.FILLNA_LIMIT)\
                            .infer_objects(copy=False)
        rolling_gauge = df['Gauge_Data_mm_per_min'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                .ffill(limit=CFG.FILLNA_LIMIT)\
                                .bfill(limit=CFG.FILLNA_LIMIT)\
                                .infer_objects(copy=False)

        df['Rolling_Adjusted_Radar'] = rolling_adj_radar
        df['Rolling_Gauge_Data'] = rolling_gauge
        df['Adjusted_Diff_from_network'] = df['Rolling_Adjusted_Radar'] - df['Rolling_Gauge_Data']
        df['Adjusted_Ratio_From_Network'] = (df['Rolling_Adjusted_Radar'] + CFG.EPSILON) / (df['Rolling_Gauge_Data'] + CFG.EPSILON)
        df.loc[df['Adjusted_Ratio_From_Network'] > 3, 'Adjusted_Ratio_From_Network'] = 3.0

        processed_data[coord] = df

    return processed_data
