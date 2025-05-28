# network.py
import numpy as np
import pandas as pd
from math import sqrt
from scipy.spatial import KDTree
from config import CFG
import tqdm as tqdm # Used in compute_network_metrics

# import pandas as pd # Already imported
pd.set_option('future.no_silent_downcasting', True)

def calculate_utm_distance(coord1, coord2):
    """Calculate Euclidean distance between two projected (UTM) coordinates."""
    x1, y1 = coord1
    x2, y2 = coord2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def build_kdtree(coordinate_locations: dict):
    """Build a KDTree from coordinate_locations dict (sensor name -> (x, y))."""
    names = list(coordinate_locations.keys())
    
    coords_list = []
    if names: # Only proceed if there are names
        try:
            # Ensure all coordinate_locations[name] are valid pairs of numbers
            coords_list = [coordinate_locations[name] for name in names]
            # Minimal check for validity (e.g. ensure they are list/tuple of len 2)
            if not all(isinstance(c, (list, tuple)) and len(c) == 2 for c in coords_list):
                raise ValueError("Coordinates must be pairs (e.g., (x,y) tuples or lists).")
        except KeyError:
            # This implies coordinate_locations was modified unexpectedly or keys are unusual.
            raise ValueError("Inconsistent coordinate_locations dictionary or problematic keys.")
        except TypeError: # If a coordinate_locations[name] is not subscriptable or iterable
            raise ValueError("Invalid data type for coordinates in coordinate_locations.")


    if not coords_list: # If names was empty, or comprehension resulted in empty/invalid.
        raise ValueError("Cannot build KDTree from empty or invalid coordinate list.")
    
    try:
        tree = KDTree(coords_list)
    except Exception as e: # Catch generic errors from KDTree constructor
        raise ValueError(f"Error initializing KDTree: {e}") # Propagate underlying error
        
    return names, tree

def get_nearest_neighbors(coord_name: str, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> list:
    """Get nearest neighbors using a KDTree for fast lookup."""

    if not coordinate_locations or coord_name not in coordinate_locations:
        # e.g., logger.debug(f"get_nearest_neighbors: Empty locations or {coord_name} not in locations.")
        return []

    # If there are not enough points to find distinct neighbors (i.e., need at least 1 other point).
    # len(coordinate_locations) includes the coord_name itself.
    if len(coordinate_locations) <= 1:
        return []

    try:
        names, tree = build_kdtree(coordinate_locations)
    except ValueError as e: # Catches errors from build_kdtree, e.g., empty list.
        # e.g., logger.error(f"Failed to build KDTree for {coord_name} in get_nearest_neighbors: {e}")
        return []

    num_points_in_tree = len(names) # Should be same as len(coordinate_locations)

    # Determine k for the query. We want n_neighbors + 1 (to include self),
    # but k cannot exceed num_points_in_tree.
    # k must also be at least 1 for tree.query to be valid.
    # Since len(coordinate_locations) > 1, num_points_in_tree is at least 2.
    # So, k_to_use will be at least min(n_neighbors+1, 2).
    # If n_neighbors=0, k_to_use = min(1, num_points_in_tree) which is 1 (as num_points_in_tree >= 2).
    k_to_use = min(n_neighbors + 1, num_points_in_tree)
    
    # This check is defensive; k_to_use should be >= 1 given num_points_in_tree >= 2 and n_neighbors >=0.
    if k_to_use <= 0:
        return [] 

    try:
        query_point_coords = coordinate_locations[coord_name]
        distances, indices = tree.query(query_point_coords, k=k_to_use)
        
        # Ensure indices is iterable and 1D-like for a single query point.
        # KDTree.query for a single point typically returns a 1D array for indices.
        # If k_to_use is 1, indices might be np.array([idx_value]) or just idx_value (older scipy).
        if k_to_use == 1 and isinstance(indices, (int, np.integer)): # If scalar index
            indices = [indices] # Make it a list of one index
            
    except Exception as e: # Catch any query errors (e.g. ValueError if k invalid, or other)
        # e.g., logger.error(f"KDTree query failed for {coord_name} with k={k_to_use}: {e}")
        return []

    found_neighbors = []
    # `indices` should now be an iterable (list or 1D numpy array)
    for i_val in indices: 
        # CRITICAL FIX: Ensure index is within bounds of the `names` list.
        # This handles cases where KDTree might return num_points_in_tree as an index (padding).
        if 0 <= i_val < num_points_in_tree:
            neighbor_name_candidate = names[i_val]
            if neighbor_name_candidate != coord_name: # Exclude the point itself
                found_neighbors.append(neighbor_name_candidate)
        # else: KDTree returned an out-of-bounds index. Silently ignore.
        #     e.g., logger.warning(f"KDTree query for {coord_name} returned out-of-bounds index {i_val}. num_points_in_tree={num_points_in_tree}")

    return found_neighbors[:n_neighbors] # Return up to n_neighbors distinct neighbors

def compute_network_metrics(all_data: dict, coordinate_locations: dict) -> dict:
    """
    Compute network metrics including Median Neighbor Alpha and Adjusted Radar.
    Uses vectorized operations and the optimized neighbor search.
    """
    print("Computing network metrics...")
    # Determine a common index across all DataFrames
    common_index = pd.Index([])
    for df_val in all_data.values(): # Renamed df to df_val to avoid conflict
        if not df_val.empty and isinstance(df_val.index, pd.DatetimeIndex):
            common_index = common_index.union(df_val.index)
    common_index = common_index.sort_values()

    def safe_reindex(df_arg, col, index_arg): # Renamed df, index to avoid conflict
        return df_arg[col].reindex(index_arg) if col in df_arg.columns else pd.Series(np.nan, index=index_arg)

    all_alphas = pd.DataFrame({
        coord: safe_reindex(df, 'Alpha', common_index)
        for coord, df in all_data.items() if 'Alpha' in df.columns # df here is from all_data.items()
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
        df = all_data[coord].copy() # df here is the specific DataFrame for the current coord
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
            # df['Median_Ratio_From_Network'] = np.nan # This column was not in original assignment section
            df['Network_Avg_Alpha'] = np.nan
            df['Alpha_From_Network'] = np.nan
            processed_data[coord] = df
            continue

        valid_neighbors = [n for n in neighbors if n in all_alphas.columns]
        if not valid_neighbors:
            # print(f"Warning: Neighbors found for {coord} but none have Alpha data.") # Original log
            df['Median_Neighbor_Alpha'] = np.nan
            df['Network_Adjusted_Radar'] = np.nan
            # Also initialize other network-dependent columns to NaN if no valid neighbors
            df['Network_Avg_Diff'] = np.nan
            df['Difference_From_Network'] = np.nan
            df['Ratio_From_Network'] = np.nan
            df['Network_Avg_Alpha'] = np.nan
            df['Alpha_From_Network'] = np.nan
            processed_data[coord] = df
            continue

        neighbor_alphas_subset = all_alphas[valid_neighbors]
        median_neighbor_alpha = neighbor_alphas_subset.median(axis=1)
        median_neighbor_alpha = median_neighbor_alpha.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)
        
        epsilon = CFG.EPSILON # Use EPSILON from CFG if defined, otherwise 0.01
        
        # Ensure target coordinate 'coord' exists in all_radar_data before trying to access it
        if coord in all_radar_data.columns:
            network_adjusted_radar = (all_radar_data[coord] + epsilon) * median_neighbor_alpha
        else:
            network_adjusted_radar = pd.Series(np.nan, index=common_index) # Or df.index if more appropriate context

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
        rolling_window = str(CFG.ROLLING_WINDOW) # Ensure ROLLING_WINDOW is a string like "15T" for time-based rolling
        
        if 'Network_Adjusted_Radar' in df.columns and 'Gauge_Data_mm_per_min' in df.columns:
            rolling_adj_radar = df['Network_Adjusted_Radar'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                .ffill(limit=CFG.FILLNA_LIMIT)\
                                .bfill(limit=CFG.FILLNA_LIMIT)\
                                .infer_objects(copy=False)
            rolling_gauge = df['Gauge_Data_mm_per_min'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                    .ffill(limit=CFG.FILLNA_LIMIT)\
                                    .bfill(limit=CFG.FILLNA_LIMIT)\
                                    .infer_objects(copy=False)

            df['Rolling_Adjusted_Radar'] = rolling_adj_radar
            df['Rolling_Gauge_Data'] = rolling_gauge # Assuming this column name is intended
            df['Adjusted_Diff_from_network'] = df['Rolling_Adjusted_Radar'] - df['Rolling_Gauge_Data']
            df['Adjusted_Ratio_From_Network'] = (df['Rolling_Adjusted_Radar'] + CFG.EPSILON) / (df['Rolling_Gauge_Data'] + CFG.EPSILON)
            # Clip the ratio, ensuring it handles NaNs correctly (clip usually preserves NaNs)
            df.loc[df['Adjusted_Ratio_From_Network'].notna() & (df['Adjusted_Ratio_From_Network'] > 3), 'Adjusted_Ratio_From_Network'] = 3.0
            df.loc[df['Adjusted_Ratio_From_Network'].notna() & (df['Adjusted_Ratio_From_Network'] < 0), 'Adjusted_Ratio_From_Network'] = 0.0 # Optional: ensure non-negative if sensible
        else:
            df['Rolling_Adjusted_Radar'] = np.nan
            df['Rolling_Gauge_Data'] = np.nan
            df['Adjusted_Diff_from_network'] = np.nan
            df['Adjusted_Ratio_From_Network'] = np.nan


        processed_data[coord] = df

    return processed_data