# Example modification within compute_network_metrics_iterative
# (in network.py or a new network_iterative.py)

import pandas as pd
import numpy as np
from tqdm import tqdm # Use tqdm.tqdm if needed
from config import CFG
from network import get_nearest_neighbors # Assuming get_nearest_neighbors is in network.py
import logging

log_file = CFG.RESULTS_DIR / 'iterative_exclusions.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Iter %(iteration)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w' # 'w' to overwrite log each run, 'a' to append
)

logger = logging.getLogger(__name__)

def compute_network_metrics_iterative(
    all_data: dict,
    coordinate_locations: dict,
    previous_flagged_intervals: dict # Dict: coord -> list of (start, end) tuples
) -> dict:
    """
    Compute network metrics, excluding neighbor data from previously flagged events
    when calculating Median_Neighbor_Alpha.
    """
    print("Computing iterative network metrics...")
    exclusion_log_for_this_iteration = [] # Initialize log for this iteration
    # Common index logic... (same as before)
    common_index = pd.Index([])
    for df in all_data.values():
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            common_index = common_index.union(df.index)
    common_index = common_index.sort_values()

    # Pre-fetch relevant data series reindexed to common_index (same as before)
    def safe_reindex(df, col, index):
        return df[col].reindex(index) if col in df.columns else pd.Series(np.nan, index=index)

    all_alphas = pd.DataFrame({
        coord: safe_reindex(df, 'Alpha', common_index)
        for coord, df in all_data.items() if 'Alpha' in df.columns
    })
    all_radar_data = pd.DataFrame({ # Needed for Network_Adjusted_Radar calc below
        coord: safe_reindex(df, 'Radar_Data_mm_per_min', common_index)
        for coord, df in all_data.items() if 'Radar_Data_mm_per_min' in df.columns
    })
    # ... fetch other needed series like Rolling_Diff ...
    all_rolling_diffs = pd.DataFrame({
        coord: safe_reindex(df, 'Rolling_Diff', common_index)
        for coord, df in all_data.items() if 'Rolling_Diff' in df.columns
    })


    processed_data = {}
    for coord in tqdm(all_data.keys()):
        df = all_data[coord].copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: Skipping network metrics for {coord}, invalid index type.")
            logger.warning(f"Skipping network metrics for {coord}, invalid index type.")
            processed_data[coord] = df
            continue

        neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=CFG.N_NEIGHBORS)

        # --- Start of Key Modification ---
        valid_neighbor_alphas_for_median = {} # Store potentially filtered alpha series

        if neighbors:
            valid_neighbors = [n for n in neighbors if n in all_alphas.columns]
            if valid_neighbors:
                for neighbor_coord in valid_neighbors:
                    neighbor_alpha_series = all_alphas[neighbor_coord].copy()

                    # Get flagged intervals for THIS neighbor from PREVIOUS iteration
                    flagged_intervals_for_neighbor = previous_flagged_intervals.get(neighbor_coord, [])

                    if flagged_intervals_for_neighbor:
                        # Apply exclusion mask
                        # Create a mask, True means KEEP data
                        mask = pd.Series(True, index=neighbor_alpha_series.index)
                        for start_time, end_time in flagged_intervals_for_neighbor:
                            # Set mask to False within the flagged interval
                            # Ensure timezone awareness matches if necessary
                            try:
                                mask.loc[start_time:end_time] = False
                            except KeyError:
                                # Handle cases where interval bounds might be outside the index
                                # This might need refinement based on exact index behavior
                                mask.loc[mask.index >= start_time & mask.index <= end_time] = False


                        # Exclude data points within flagged intervals by setting them to NaN
                        neighbor_alpha_series[~mask] = np.nan

                    # Store the (potentially modified) series
                    valid_neighbor_alphas_for_median[neighbor_coord] = neighbor_alpha_series
            else:
                 print(f"Warning: Neighbors found for {coord} but none have Alpha data.")
                 # Handle cases with no valid neighbors (same as before)

        # --- Calculate Median_Neighbor_Alpha using the filtered data ---
        if valid_neighbor_alphas_for_median:

            df_neighbor_alphas = pd.DataFrame(valid_neighbor_alphas_for_median)
             # Count non-NaN values across neighbors for each time step
            neighbor_count_used = df_neighbor_alphas.count(axis=1)




            median_neighbor_alpha = pd.DataFrame(valid_neighbor_alphas_for_median).median(axis=1)
            # Fill NaNs in the resulting median series (same as before)
            median_neighbor_alpha = median_neighbor_alpha.ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)


        else:
            # Handle case where no valid neighbors or no alpha data (assign NaN)
             median_neighbor_alpha = pd.Series(np.nan, index=common_index)

        # --- End of Key Modification ---

        # Assign Median_Neighbor_Alpha back to the dataframe's index
        df['Median_Neighbor_Alpha'] = median_neighbor_alpha.reindex(df.index)

        df['Neighbor_Count_Used'] = neighbor_count_used.reindex(df.index).fillna(0).astype(int) # Fill potential NaNs with 0 count


        # Calculate Network_Adjusted_Radar using the potentially modified Median_Neighbor_Alpha
        epsilon = 0.01
        sensor_radar = all_radar_data.get(coord, pd.Series(np.nan, index=common_index))
        aligned_median_alpha = median_neighbor_alpha.reindex(sensor_radar.index)
        network_adjusted_radar = (sensor_radar + epsilon) * median_neighbor_alpha
        df['Network_Adjusted_Radar'] = network_adjusted_radar.reindex(df.index)


        # --- Calculate ALL other network metrics as before ---
        # These metrics (like Adjusted_Diff/Ratio) will now implicitly use the
        # potentially refined Median_Neighbor_Alpha via Network_Adjusted_Radar.

        # Example: Recalculate Adjusted Diff/Ratio (assuming Rolling_Gauge is already computed)
        if 'Network_Adjusted_Radar' in df.columns and 'Gauge_Data_mm_per_min' in df.columns:
             rolling_window = CFG.ROLLING_WINDOW
             rolling_adj_radar = df['Network_Adjusted_Radar'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                 .ffill(limit=CFG.FILLNA_LIMIT)\
                                 .bfill(limit=CFG.FILLNA_LIMIT)\
                                 .infer_objects(copy=False)
             # Ensure Rolling_Gauge_Data is calculated or retrieved if not already present
             if 'Rolling_Gauge_Data' not in df.columns:
                 df['Rolling_Gauge_Data'] = df['Gauge_Data_mm_per_min'].rolling(rolling_window, center=True, min_periods=1).mean()\
                                         .ffill(limit=CFG.FILLNA_LIMIT)\
                                         .bfill(limit=CFG.FILLNA_LIMIT)\
                                         .infer_objects(copy=False)

             df['Rolling_Adjusted_Radar'] = rolling_adj_radar
             df['Adjusted_Diff_from_network'] = df['Rolling_Adjusted_Radar'] - df['Rolling_Gauge_Data']
             df['Adjusted_Ratio_From_Network'] = (df['Rolling_Adjusted_Radar'] + CFG.EPSILON) / (df['Rolling_Gauge_Data'] + CFG.EPSILON)
             df.loc[df['Adjusted_Ratio_From_Network'] > 3, 'Adjusted_Ratio_From_Network'] = 3.0 # Apply cap
        else:
             # Assign NaNs if required inputs are missing
              df['Rolling_Adjusted_Radar'] = np.nan
              df['Adjusted_Diff_from_network'] = np.nan
              df['Adjusted_Ratio_From_Network'] = np.nan

        # Calculate other network metrics like Difference_From_Network, Alpha_From_Network etc.
        # using the updated neighbor information if needed, or based on the new adjusted radar.
        # ... (rest of the calculations from original compute_network_metrics) ...
        # Ensure you handle cases where neighbors might be missing rolling_diff etc.

        processed_data[coord] = df

    return processed_data