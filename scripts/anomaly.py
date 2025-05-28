# anomaly.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import CFG

def flag_anomalies(all_data: dict) -> dict:
    """
    Flag anomalies using network comparison and other metrics.
    Modified logic: Prelim_Flag based on diff and ratio conditions.
    """
    # print("Flagging anomalies...")
    processed_data = {}
    for coord, df_orig in all_data.items():
        df = df_orig.copy()
        # required_cols = ['Alpha_From_Network', 'Difference_From_Network', 'Ratio_From_Network',
        #                  'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min']
        required_cols = ['Adjusted_Diff_from_network', 'Adjusted_Ratio_From_Network']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Skipping flagging for {coord}, missing required columns.")
            processed_data[coord] = df_orig
            continue

        # df['Adjusted_Radar'] = (df['Radar_Data_mm_per_min']*epsilon) * median_neighbor_alpha
        # df['Alpha_Dev'] = (df['Alpha_From_Network'] - CFG.ALPHA_DEV_CENTER).abs()
        cond_diff = df['Adjusted_Diff_from_network'].abs() > CFG.ABS_DIFF_THRESHOLD_MM_MIN
        cond_ratio = (df['Adjusted_Ratio_From_Network'] > 1 + CFG.RATIO_THRESHOLD) | \
                     (df['Adjusted_Ratio_From_Network'] < 1 - CFG.RATIO_THRESHOLD)

        df['Prelim_Flag'] = cond_diff & cond_ratio



        df['Flagged'] = df['Prelim_Flag']
        processed_data[coord] = df
    return processed_data

def flag_anomalies_v2(all_data: dict) -> dict:
    """
    Flag anomalies using network comparison and other metrics.
    Modified logic: Prelim_Flag based on diff and ratio conditions.
    """
    # print("Flagging anomalies...")
    processed_data = {}
    for coord, df_orig in all_data.items():
        df = df_orig.copy()
        required_cols = ['Alpha_From_Network', 'Difference_From_Network', 'Ratio_From_Network',
                         'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Skipping flagging for {coord}, missing required columns.")
            processed_data[coord] = df_orig
            continue

        median_neighbor_alpha = df['Median_Neighbor_Alpha'].fillna(CFG.ALPHA_DEV_CENTER)
        epsilon = 0.01
        df['Adjusted_Radar'] = (df['Radar_Data_mm_per_min']+epsilon) * median_neighbor_alpha
        df['Alpha_Dev'] = (df['Alpha_From_Network'] - CFG.ALPHA_DEV_CENTER).abs()
        cond_diff = df['Adjusted_Diff_from_network'].abs() > CFG.ABS_DIFF_THRESHOLD_MM_MIN
        cond_ratio = (df['Adjusted_Ratio_From_Network'] > 1 + CFG.RATIO_THRESHOLD) | \
                     (df['Adjusted_Ratio_From_Network'] < 1 - CFG.RATIO_THRESHOLD)
        # cond_alpha = (df['Alpha_From_Network'] < CFG.ALPHA_DEV_CENTER - CFG.ALPHA_DEV_MARGIN) | \
        #              (df['Alpha_From_Network'] > CFG.ALPHA_DEV_CENTER + CFG.ALPHA_DEV_MARGIN)
        # Using diff and ratio for preliminary flag
        df['Prelim_Flag'] = cond_diff & cond_ratio

        # Gauge-zero handling
        df['Gauge_Zero_Radar_Nonzero'] = (df['Gauge_Data_mm_per_min'] <= CFG.BOTH_ZERO_THRESHOLD_MM_MIN) & \
                                         (df['Radar_Data_mm_per_min'] > CFG.BOTH_ZERO_THRESHOLD_MM_MIN)
        gauge_zero_counts = df['Gauge_Zero_Radar_Nonzero'].rolling(
            window=CFG.GAUGE_ZERO_WINDOW, center=True, min_periods=1).sum()
        dt_minutes = 1
        if len(df.index) > 1:
            freq = df.index.freq or pd.infer_freq(df.index)
            if freq:
                dt_minutes = pd.to_timedelta(freq).total_seconds() / 60.0
            else:
                dt_minutes = (df.index[1] - df.index[0]).total_seconds() / 60.0
        window_duration_minutes = pd.Timedelta(CFG.GAUGE_ZERO_WINDOW).total_seconds() / 60.0
        window_size_points = max(1, round(window_duration_minutes / dt_minutes))
        threshold_count = CFG.GAUGE_ZERO_PERCENTAGE * window_size_points
        df['Gauge_Zero_Condition'] = gauge_zero_counts >= threshold_count

        # Combine preliminary flag (optionally include gauge zero condition)
        df['Combined_Flag'] = df['Prelim_Flag']  # | df['Gauge_Zero_Condition']
        # Optionally, apply continuity filtering here if needed
        df['Flagged'] = df['Combined_Flag']
        processed_data[coord] = df
    return processed_data

