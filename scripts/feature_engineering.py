# feature_engineering.py
import numpy as np
import pandas as pd
from config import CFG
import warnings

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for anomaly detection using centered rolling windows."""
    df_out = df.copy()
    # change the name of 'Radar Data' to 'Radar_Data_mm_per_min' for consistency
    if 'Radar Data' in df_out.columns:
        df_out.rename(columns={'Radar Data': 'Radar_Data_mm_per_min'}, inplace=True)
    # change the name of 'Gauge_Data_mm_per_min' to 'Gauge_Data_mm_per_min' for consistency
    if 'Gauge Data' in df_out.columns:
        df_out.rename(columns={'Gauge Data': 'Gauge_Data_mm_per_min'}, inplace=True)
    df_out['Difference'] = df_out['Radar_Data_mm_per_min'] - df_out['Gauge_Data_mm_per_min']
    df_out['Abs_Difference'] = df_out['Difference'].abs()
    df_out['Percentage_Difference'] = np.where(
        df_out['Gauge_Data_mm_per_min'] > CFG.MIN_GAUGE_THRESHOLD,
        (df_out['Difference'] / df_out['Gauge_Data_mm_per_min']) * 100, np.nan
    )
    gauge_clipped = df_out['Gauge_Data_mm_per_min'].abs().clip(lower=CFG.MIN_GAUGE_THRESHOLD)
    df_out['Relative_Error'] = df_out['Abs_Difference'] / gauge_clipped

    # Compute rolling means with a single call and reuse later.
    rolling_window_str = CFG.ROLLING_WINDOW
    rolling_diff = df_out['Difference'].rolling(window=rolling_window_str, center=True, min_periods=1).mean()
    rolling_gauge = df_out['Gauge_Data_mm_per_min'].rolling(window=rolling_window_str, center=True, min_periods=1).mean()
    rolling_radar = df_out['Radar_Data_mm_per_min'].rolling(window=rolling_window_str, center=True, min_periods=1).mean()
    alpha = (rolling_gauge + CFG.K_PARAM) / (rolling_radar + CFG.K_PARAM)
    
    df_out['Rolling_Diff'] = rolling_diff
    df_out['Rolling_Gauge'] = rolling_gauge
    df_out['Rolling_Radar'] = rolling_radar
    df_out['Alpha'] = alpha

    # Use fillna/backfill once for each rolling column.
    for col in ['Rolling_Diff', 'Rolling_Gauge', 'Rolling_Radar', 'Alpha']:
        if col in df_out:
            # df_out[col] = df_out[col].fillna(method=CFG.FILLNA_METHOD, limit=CFG.FILLNA_LIMIT)
            # df_out[col] = df_out[col].fillna(method='bfill', limit=CFG.FILLNA_LIMIT)
            df_out[col] = df_out[col].ffill(limit=CFG.FILLNA_LIMIT).bfill(limit=CFG.FILLNA_LIMIT)

    return df_out

def apply_feature_engineering(all_data: dict) -> dict:
    """Apply feature engineering to all sensor DataFrames in the dictionary."""
    processed_data = {}
    # print("Applying feature engineering...")

    for key, df in all_data.items():
        processed_data[key] = create_features(df)
    return processed_data
