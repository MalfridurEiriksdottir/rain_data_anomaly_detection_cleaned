# network_iterative.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from config import CFG
from network2 import get_nearest_neighbors

# --- Define Filter Globally ---
class IterationLogFilter(logging.Filter):
    """Adds the current iteration number to the log record."""
    def __init__(self, iteration=-1): # Use default iteration like -1 for outside loop
        super().__init__()
        self.iteration = iteration

    def filter(self, record):
        # Add iteration attribute if not present or override if needed
        if not hasattr(record, 'iteration'):
             record.iteration = self.iteration
        return True
# ---



# ---

def compute_network_metrics_iterative(
    all_data: dict,
    coordinate_locations: dict,
    previous_flagged_intervals: dict,
    current_iteration: int
) -> tuple[dict, list]:
    """
    Compute network metrics, excluding neighbor data from previously flagged events
    when calculating Median_Neighbor_Alpha. Logs exclusions and returns them.
    Calculates and stores the count of neighbors used for the median calculation.
    """
    print(f"Computing iterative network metrics (Iteration {current_iteration})...")
    # Create and add the filter for this specific call's iteration context
    iteration_filter = IterationLogFilter(current_iteration)
    # Apply the filter ONLY for the duration of this function call if possible
    # A cleaner way might be to pass the logger and filter around, or use LogAdapters,
    # but adding/removing the filter directly is simpler here.

    exclusion_log_for_this_iteration = []

    # --- Common index and data pre-fetching ---
    common_index = pd.Index([])
    for df_ in all_data.values():
        if not df_.empty and isinstance(df_.index, pd.DatetimeIndex):
            common_index = common_index.union(df_.index)
    common_index = common_index.sort_values()

    def safe_reindex(df, col, index):
        series = df[col] if col in df.columns else pd.Series(dtype=float)
        return series.reindex(index).fillna(np.nan)

    all_alphas = pd.DataFrame({
        coord: safe_reindex(df_, 'Alpha', common_index)
        for coord, df_ in all_data.items()
    })
    all_alphas.dropna(axis=1, how='all', inplace=True)

    all_radar_data = pd.DataFrame({
        coord: safe_reindex(df_, 'Radar_Data_mm_per_min', common_index)
        for coord, df_ in all_data.items()
    })
    all_radar_data.dropna(axis=1, how='all', inplace=True)

    all_rolling_diffs = pd.DataFrame({ # Keep if needed for other metrics
        coord: safe_reindex(df_, 'Rolling_Diff', common_index)
        for coord, df_ in all_data.items()
    })
    all_rolling_diffs.dropna(axis=1, how='all', inplace=True)
    # ---

    processed_data = {}
    for coord in tqdm(all_data.keys(), desc=f"Iter {current_iteration} Network Metrics"):
        df = all_data[coord].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: Skipping network metrics for {coord}, invalid index type.")
            processed_data[coord] = df
            continue

        neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=CFG.N_NEIGHBORS)
        valid_neighbor_alphas_for_median = {}

        if neighbors:
            valid_neighbors = [n for n in neighbors if n in all_alphas.columns]
            if valid_neighbors:
                for neighbor_coord in valid_neighbors:
                    neighbor_alpha_series = all_alphas[neighbor_coord].copy()
                    flagged_intervals_for_neighbor = previous_flagged_intervals.get(neighbor_coord, [])

                    if flagged_intervals_for_neighbor:
                        mask = pd.Series(True, index=neighbor_alpha_series.index)
                        num_points_excluded_this_neighbor = 0
                        for start_time, end_time in flagged_intervals_for_neighbor:
                            interval_mask = (neighbor_alpha_series.index >= start_time) & \
                                            (neighbor_alpha_series.index <= end_time)
                            num_affected_in_interval = interval_mask.sum()
                            if num_affected_in_interval > 0:
                                mask[interval_mask] = False
                                num_points_excluded_this_neighbor += num_affected_in_interval
                                # --- Logging ---
                                log_msg = (f"Target: {coord}, Neighbor: {neighbor_coord}, "
                                           f"Excluding interval: {start_time} to {end_time} "
                                           f"({num_affected_in_interval} points)")
                                # --- Store log entry ---
                                exclusion_log_for_this_iteration.append({
                                    'iteration': current_iteration, 'target_coord': coord,
                                    'neighbor_coord': neighbor_coord, 'excluded_start': start_time,
                                    'excluded_end': end_time, 'num_excluded_points': num_affected_in_interval
                                })
                        if num_points_excluded_this_neighbor > 0:
                           neighbor_alpha_series[~mask] = np.nan

                    valid_neighbor_alphas_for_median[neighbor_coord] = neighbor_alpha_series
            else:
                 print(f"Warning: Neighbors found for {coord} but none have Alpha data.")

        # --- Calculate Median AND Count ---
        median_neighbor_alpha = pd.Series(np.nan, index=common_index)
        neighbor_count_used = pd.Series(0, index=common_index)

        if valid_neighbor_alphas_for_median:
            df_neighbor_alphas = pd.DataFrame(valid_neighbor_alphas_for_median)
            neighbor_count_used = df_neighbor_alphas.count(axis=1)
            median_neighbor_alpha = df_neighbor_alphas.median(axis=1, skipna=True)
            median_neighbor_alpha = median_neighbor_alpha.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)

        # --- Store results back ---
        df['Median_Neighbor_Alpha'] = median_neighbor_alpha.reindex(df.index).fillna(np.nan)
        df['Neighbor_Count_Used'] = neighbor_count_used.reindex(df.index).fillna(0).astype(int)

        # --- Calculate Network_Adjusted_Radar ---
        epsilon = 0.01
        sensor_radar = all_radar_data.get(coord)
        if sensor_radar is not None:
            aligned_median_alpha = median_neighbor_alpha.reindex(sensor_radar.index) # Align to common index first
            network_adjusted_radar = (sensor_radar + epsilon) * aligned_median_alpha
            df['Network_Adjusted_Radar'] = network_adjusted_radar.reindex(df.index) # Align back to df index
        else:
            df['Network_Adjusted_Radar'] = np.nan

        # --- Calculate final adjusted Diff/Ratio ---
        if 'Network_Adjusted_Radar' in df.columns and 'Gauge_Data_mm_per_min' in df.columns:
             rolling_window = CFG.ROLLING_WINDOW
             rolling_adj_radar = df['Network_Adjusted_Radar'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                 .ffill(limit=CFG.FILLNA_LIMIT)\
                                 .bfill(limit=CFG.FILLNA_LIMIT)\
                                 .infer_objects(copy=False)
             if 'Rolling_Gauge_Data' not in df.columns:
                 df['Rolling_Gauge_Data'] = df['Gauge_Data_mm_per_min'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                         .ffill(limit=CFG.FILLNA_LIMIT)\
                                         .bfill(limit=CFG.FILLNA_LIMIT)\
                                         .infer_objects(copy=False)

             df['Rolling_Adjusted_Radar'] = rolling_adj_radar
             # Ensure alignment before operations
             rolling_gauge_aligned = df['Rolling_Gauge_Data'].reindex(df['Rolling_Adjusted_Radar'].index)
             df['Adjusted_Diff_from_network'] = df['Rolling_Adjusted_Radar'] - rolling_gauge_aligned

             # Safter division and handling inf/nan
             adj_ratio = (df['Rolling_Adjusted_Radar'] + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
             adj_ratio.replace([np.inf, -np.inf], 3.0, inplace=True) 
             df['Adjusted_Ratio_From_Network'] = adj_ratio.clip(upper=3.0).fillna(1.0) # Clip and fill remaining NaNs with 1

        else:
              df['Rolling_Adjusted_Radar'] = np.nan
              df['Adjusted_Diff_from_network'] = np.nan
              df['Adjusted_Ratio_From_Network'] = np.nan

        processed_data[coord] = df
    # --- End Main Sensor Loop ---


    return processed_data, exclusion_log_for_this_iteration