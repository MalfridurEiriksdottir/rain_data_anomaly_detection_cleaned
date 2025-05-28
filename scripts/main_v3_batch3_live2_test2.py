
# --- START OF COMPLETE main_v3_batch3_live.py ---
import pandas as pd
from tqdm import tqdm
import traceback
import numpy as np
import logging
import datetime
import time
from pathlib import Path
import json
import sys
import argparse
import geopandas as gpd # For type hint

# Custom module imports
from config import CFG
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata)
from feature_engineering import apply_feature_engineering
from anomaly import flag_anomalies
from plotting3_batches2 import create_plots_with_error_markers, generate_html_dashboard
from network2 import get_nearest_neighbors # Ensure this is the robust version
from batch_adjustment import compute_regional_adjustment

# --- Logger Setup ---
log_file_path = getattr(CFG, 'LOG_FILE', Path("main_live_pipeline_v3.log"))
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MAIN_LIVE_V3")
logger.setLevel(logging.INFO)

# --- State File and Constants ---
STATE_FILE_PATH = CFG.BASE_DIR / "live_processing_state_main_v3.json"
DEFAULT_HISTORICAL_START_STR = getattr(CFG, 'DEFAULT_HISTORICAL_START_STR', "2024-01-01 00:00:00")

# =============================================================================
# Helper Function: initialize_adjustment_columns
# =============================================================================
def initialize_adjustment_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_init = {
        'Alpha': float, 'Rolling_Gauge': float, 'Rolling_Radar': float,
        'Median_Neighbor_Alpha': float, 'Network_Adjusted_Radar': float,
        'Adjusted_Diff_from_network': float, 'Adjusted_Ratio_From_Network': float,
        'Flagged': bool, 'Batch_Flag': bool, 'Radar_Freg_Adjusted': float,
        'Final_Adjusted_Rainfall': float, 'Rolling_Abs_Error': float,
        'Rolling_Prop_Flagged': float, 'Batch_Alpha_Adjusted_Radar': float
    }
    for col, dtype in cols_to_init.items():
        if col not in df.columns:
            if dtype == bool: df[col] = pd.Series(False, index=df.index, dtype=dtype)
            else: df[col] = pd.Series(np.nan, index=df.index, dtype=dtype)
    return df

# =============================================================================
# Master Data Loading and Persistence
# =============================================================================
def load_and_initialize_master_data(
    list_of_target_coords: list,
    pkl_dir: Path,
    pkl_pattern: str,
    combined_csv_dir: Path,
    combined_csv_pattern: str = "combined_data_({x},{y}).csv"
) -> dict:
    master_data_dict = {}
    logger.info(f"Initializing master data for {len(list_of_target_coords)} coordinates from PKL/CSV...")
    for coord_str in tqdm(list_of_target_coords, desc="Loading/Initializing Master Data"):
        try:
            parsed_coords = coord_str.strip("()").split(',')
            x = int(parsed_coords[0].strip())
            y = int(parsed_coords[1].strip())
        except Exception:
            logger.warning(f"Could not parse coord_str '{coord_str}' for master data. Skipping.")
            master_data_dict[coord_str] = pd.DataFrame()
            continue
        df = None
        pkl_filename = pkl_pattern.format(x=x, y=y)
        pkl_filepath = pkl_dir / pkl_filename
        if pkl_filepath.exists():
            try:
                df = pd.read_pickle(pkl_filepath)
                logger.debug(f"Loaded existing PKL for {coord_str} from {pkl_filepath}")
                time_col_name = df.index.name if df.index.name else 'time'
                if not isinstance(df.index, pd.DatetimeIndex):
                    logger.warning(f"Data for {coord_str} from PKL {pkl_filepath} has no DatetimeIndex. Trying to fix.")
                    if 'time' in df.columns: df.set_index('time', inplace=True)
                    elif 'datetime' in df.columns: df.set_index('datetime', inplace=True)
                    else: df = pd.DataFrame(); logger.error(f"Cannot fix index for {coord_str} from PKL.")
                if not df.empty: df.index = pd.to_datetime(df.index, errors='coerce')
                df.index.name = time_col_name
            except Exception as e:
                logger.error(f"Error loading PKL {pkl_filepath} for {coord_str}: {e}. Will try combined CSV.")
                df = None
        if df is None or df.empty:
            logger.info(f"PKL for {coord_str} not found or invalid. Loading from combined CSV: {combined_csv_dir}")
            combined_filename = combined_csv_pattern.format(x=x, y=y)
            combined_filepath = combined_csv_dir / combined_filename
            if combined_filepath.exists():
                try:
                    df_combined = pd.read_csv(combined_filepath)
                    time_col_in_csv = 'time' if 'time' in df_combined.columns else 'datetime' if 'datetime' in df_combined.columns else None
                    if time_col_in_csv and 'Radar Data' in df_combined.columns and 'Gauge Data' in df_combined.columns:
                        df_combined[time_col_in_csv] = pd.to_datetime(df_combined[time_col_in_csv], errors='coerce')
                        df_combined.dropna(subset=[time_col_in_csv], inplace=True)
                        df_combined = df_combined.set_index(time_col_in_csv).sort_index()
                        df = df_combined[['Radar Data', 'Gauge Data']].copy()
                    else: df = pd.DataFrame()
                except Exception as e: df = pd.DataFrame(); logger.error(f"Error loading CSV {combined_filepath}: {e}")
            else: df = pd.DataFrame(); logger.warning(f"Combined CSV {combined_filepath} not found.")
        if df.empty:
            master_data_dict[coord_str] = pd.DataFrame()
            continue
        
        # *** CRITICAL: Implement your actual unit conversion logic here ***
        if 'Radar_Data_mm_per_min' not in df.columns:
            if 'Radar Data' in df.columns: df['Radar_Data_mm_per_min'] = df['Radar Data'] # Placeholder
            else: df['Radar_Data_mm_per_min'] = pd.Series(dtype='float64', index=df.index)
        if 'Gauge_Data_mm_per_min' not in df.columns:
            if 'Gauge Data' in df.columns: df['Gauge_Data_mm_per_min'] = df['Gauge Data'] # Placeholder
            else: df['Gauge_Data_mm_per_min'] = pd.Series(dtype='float64', index=df.index)
        
        df = initialize_adjustment_columns(df)
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            elif str(df.index.tz).upper() != 'UTC': df.index = df.index.tz_convert('UTC')
        else: df = pd.DataFrame(); logger.error(f"DF for {coord_str} no DatetimeIndex after init.")
        master_data_dict[coord_str] = df
    logger.info(f"Master data initialization complete. Loaded/initialized for {len(master_data_dict)} coords.")
    return master_data_dict

def save_master_data_to_pkl(master_data_to_save: dict, pkl_dir: Path, pkl_pattern: str):
    logger.info(f"Saving updated master data to PKL files in: {pkl_dir}")
    pkl_dir.mkdir(parents=True, exist_ok=True)
    for coord_str, df_to_save in tqdm(master_data_to_save.items(), desc="Saving Master PKLs"):
        if df_to_save.empty: continue
        try:
            x, y = map(int, coord_str.strip("()").split(','))
            pkl_filename = pkl_pattern.format(x=x, y=y)
            pkl_filepath = pkl_dir / pkl_filename
            df_save_final = df_to_save.copy()
            if isinstance(df_save_final.index, pd.DatetimeIndex):
                if df_save_final.index.tz is None: df_save_final.index = df_save_final.index.tz_localize('UTC')
                elif str(df_save_final.index.tz).upper() != 'UTC': df_save_final.index = df_save_final.index.tz_convert('UTC')
            df_save_final.to_pickle(pkl_filepath)
        except Exception as e: logger.error(f"Error saving PKL for {coord_str}: {e}", exc_info=True)

# --- State Management Functions ---
def load_last_actual_data_timestamp():
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, 'r') as f: state = json.load(f)
            ts_str = state.get("last_actual_max_data_timestamp_utc")
            if ts_str: return pd.Timestamp(ts_str).tz_convert('UTC')
        except Exception as e: logger.error(f"Error loading state: {e}. Defaulting.")
    return pd.Timestamp(DEFAULT_HISTORICAL_START_STR).tz_localize('UTC')

def save_last_actual_data_timestamp(ts_utc: pd.Timestamp):
    if pd.isna(ts_utc): logger.warning("Attempt to save NaT as last timestamp. Skipping."); return
    try:
        ts_aware = ts_utc if ts_utc.tzinfo is not None else ts_utc.tz_localize('UTC')
        state = {"last_actual_max_data_timestamp_utc": ts_aware.isoformat()}
        STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE_PATH, 'w') as f: json.dump(state, f, indent=4)
        logger.info(f"Saved last actual max data timestamp: {ts_aware.isoformat()}")
    except Exception as e: logger.error(f"Error saving state: {e}")

# =============================================================================
# Core Processing Cycle Function
# =============================================================================
def run_daily_adjustment_cycle(
    master_sensor_data_dict_input: dict,
    processing_start_utc: pd.Timestamp,
    processing_end_for_this_cycle: pd.Timestamp,
    coordinate_locations_utm_all_static: dict,
    utm_to_channel_desc_map_all_static: dict,
    gdf_wgs84_all_static: gpd.GeoDataFrame,
    svk_coords_utm_all_static: list
    ):
    logger.info(f"--- Starting Adjustment Cycle for Window: {processing_start_utc} to {processing_end_for_this_cycle} ---")
    all_data_current_window_processed = {}
    actual_data_max_timestamp_in_window = processing_start_utc

    for coord_str, df_master_full in master_sensor_data_dict_input.items():
        if df_master_full.empty: continue
        df_master_utc = df_master_full # Assumed UTC indexed
        mask = (df_master_utc.index >= processing_start_utc) & (df_master_utc.index < processing_end_for_this_cycle)
        if mask.any():
            all_data_current_window_processed[coord_str] = df_master_utc.loc[mask].copy()
            if not all_data_current_window_processed[coord_str].empty:
                current_max_ts = all_data_current_window_processed[coord_str].index.max()
                if pd.notna(current_max_ts):
                    actual_data_max_timestamp_in_window = max(actual_data_max_timestamp_in_window, current_max_ts)

    if not all_data_current_window_processed:
        logger.info("No data in master store for the current processing window.")
        return True, False, actual_data_max_timestamp_in_window

    logger.info(f"Found {len(all_data_current_window_processed)} sensors with data in window.")
    valid_coords_in_window_list = list(all_data_current_window_processed.keys())
    coordinate_locations_utm_window_static = {
        k: coordinate_locations_utm_all_static[k] for k in valid_coords_in_window_list if k in coordinate_locations_utm_all_static
    }

    # --- Preprocessing on Windowed Data ---
    logger.info("--- Preprocessing Windowed Data ---")
    temp_preprocessed_data_for_window = {}
    for coord, df_window_orig in tqdm(all_data_current_window_processed.items(), desc="Preprocessing Window"):
        df = df_window_orig.copy()
        if 'Gauge_Data_mm_per_min' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
            gauge_valid_idx = df[df['Gauge_Data_mm_per_min'].notna()].index
            radar_valid_idx = df[df['Radar_Data_mm_per_min'].notna()].index
            if gauge_valid_idx.empty or radar_valid_idx.empty:
                logger.warning(f"Sensor {coord} (Window): Preprocessing - No valid gauge or radar for alignment.")
                df = pd.DataFrame(index=df.index, columns=df.columns)
            else:
                common_start = max(gauge_valid_idx.min(), radar_valid_idx.min())
                common_end = min(gauge_valid_idx.max(), radar_valid_idx.max())
                if common_start < common_end: df = df.loc[common_start:common_end].copy()
                else: logger.warning(f"Sensor {coord} (Window): Preprocessing - No overlapping valid data."); df = pd.DataFrame(index=df.index, columns=df.columns)
        if df.empty: temp_preprocessed_data_for_window[coord] = df; continue
        if 'Gauge_Data_mm_per_min' in df.columns: df['Gauge_Data_mm_per_min'].fillna(0.0, inplace=True)
        df = initialize_adjustment_columns(df)
        try: df = apply_feature_engineering({coord: df})[coord]
        except Exception as e: logger.error(f"FE Error {coord}: {e}"); df = pd.DataFrame(columns=df.columns, index=df.index) # Make empty on FE error
        temp_preprocessed_data_for_window[coord] = df
    all_data_current_window_processed = temp_preprocessed_data_for_window
    if not any(not df.empty for df in all_data_current_window_processed.values()):
        logger.info("No data survived preprocessing. Updating master and returning."); return True, True, actual_data_max_timestamp_in_window

    # --- Batch Processing Loop ---
    logger.info("--- Starting Batch Processing on Windowed Data ---")
    global_start_win, global_end_win, valid_indices_found_win = pd.Timestamp.max.tz_localize('UTC'), pd.Timestamp.min.tz_localize('UTC'), False
    for df_win in all_data_current_window_processed.values():
        if not df_win.empty and isinstance(df_win.index, pd.DatetimeIndex) and df_win.index.tz is not None:
            cs, ce = df_win.index.min(), df_win.index.max()
            if pd.notna(cs) and pd.notna(ce): global_start_win, global_end_win, valid_indices_found_win = min(global_start_win, cs), max(global_end_win, ce), True
    if not valid_indices_found_win or global_start_win >= global_end_win: logger.warning("Batch Loop: No valid data range in window.")
    else:
        batch_start_times = pd.date_range(start=global_start_win.floor(CFG.BATCH_DURATION), end=global_end_win, freq=CFG.BATCH_DURATION, tz='UTC')
        batch_periods = []
        for start_p in batch_start_times:
            end_p = min(start_p + pd.Timedelta(CFG.BATCH_DURATION), processing_end_for_this_cycle, global_end_win + pd.Timedelta(microseconds=1))
            if start_p < end_p: batch_periods.append((start_p, end_p))
        if batch_periods:
            logger.info(f"Generated {len(batch_periods)} batch periods for window.")
            # PASTE YOUR ORIGINAL BATCH PROCESSING LOOP HERE
            # Ensure it uses `all_data_current_window_processed` and `coordinate_locations_utm_window_static`
            # Example structure:
            valid_coordinate_locations_utm_window = coordinate_locations_utm_window_static
            for batch_start, batch_end in tqdm(batch_periods, desc="Processing Window Batches"):
                logger.debug(f"Batch: {batch_start} to {batch_end}")
                batch_avg_alphas = {}
                for coord_loop, df_loop in all_data_current_window_processed.items():
                    batch_mask_loop = (df_loop.index >= batch_start) & (df_loop.index < batch_end)
                    if batch_mask_loop.any() and 'Alpha' in df_loop.columns:
                        avg_alpha = df_loop.loc[batch_mask_loop, 'Alpha'].mean(skipna=True)
                        batch_avg_alphas[coord_loop] = avg_alpha if pd.notna(avg_alpha) else 1.0
                    else: batch_avg_alphas[coord_loop] = 1.0
                for coord_loop_target, df_target_loop in all_data_current_window_processed.items(): 
                    batch_mask_target = (df_target_loop.index >= batch_start) & (df_target_loop.index < batch_end)
                    if not batch_mask_target.any(): continue
                    neighbors = get_nearest_neighbors(coord_loop_target, valid_coordinate_locations_utm_window, n_neighbors=CFG.N_NEIGHBORS)
                    relevant_coords_alpha = [coord_loop_target] + [n for n in neighbors if n in batch_avg_alphas]
                    batch_alpha_values_median = [batch_avg_alphas.get(c, 1.0) for c in relevant_coords_alpha]
                    median_batch_avg_alpha = np.nanmedian(batch_alpha_values_median) if batch_alpha_values_median else 1.0
                    if pd.isna(median_batch_avg_alpha): median_batch_avg_alpha = 1.0
                    if 'Radar_Data_mm_per_min' in df_target_loop.columns:
                        df_target_loop.loc[batch_mask_target, 'Batch_Alpha_Adjusted_Radar'] = \
                            (df_target_loop.loc[batch_mask_target, 'Radar_Data_mm_per_min'].fillna(0) + CFG.EPSILON) * median_batch_avg_alpha
                for coord_loop_flagprep, df_loop_flagprep in all_data_current_window_processed.items():
                    batch_mask_flagprep = (df_loop_flagprep.index >= batch_start) & (df_loop_flagprep.index < batch_end)
                    if not batch_mask_flagprep.any(): continue
                    req_cols_flagprep = ['Batch_Alpha_Adjusted_Radar', 'Gauge_Data_mm_per_min', 'Rolling_Gauge']
                    if all(c in df_loop_flagprep.columns for c in req_cols_flagprep) and \
                    not df_loop_flagprep.loc[batch_mask_flagprep, 'Rolling_Gauge'].isnull().all() and \
                    not df_loop_flagprep.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar'].isnull().all():
                        rolling_batch_alpha_adj = df_loop_flagprep.loc[batch_mask_flagprep, 'Batch_Alpha_Adjusted_Radar']\
                            .rolling(str(CFG.ROLLING_WINDOW), center=True, min_periods=1).mean().ffill().bfill()
                        rolling_gauge_batch = df_loop_flagprep.loc[batch_mask_flagprep, 'Rolling_Gauge']
                        rolling_gauge_aligned, rolling_batch_alpha_adj_aligned = rolling_gauge_batch.align(rolling_batch_alpha_adj, join='inner')
                        if not rolling_gauge_aligned.empty and not rolling_batch_alpha_adj_aligned.empty:
                            diff = rolling_batch_alpha_adj_aligned - rolling_gauge_aligned
                            ratio_raw = (rolling_batch_alpha_adj_aligned + CFG.EPSILON).div(rolling_gauge_aligned + CFG.EPSILON)
                            ratio_no_inf = ratio_raw.replace([np.inf, -np.inf], 3.0)
                            ratio_clipped = ratio_no_inf.clip(lower=0.0, upper=3.0).fillna(1.0)
                            df_loop_flagprep.loc[diff.index, 'Adjusted_Diff_from_network'] = diff
                            df_loop_flagprep.loc[ratio_clipped.index, 'Adjusted_Ratio_From_Network'] = ratio_clipped
                        else:
                            original_indices_in_batch_fgp = df_loop_flagprep.loc[batch_mask_flagprep].index
                            df_loop_flagprep.loc[original_indices_in_batch_fgp, 'Adjusted_Diff_from_network'] = np.nan
                            df_loop_flagprep.loc[original_indices_in_batch_fgp, 'Adjusted_Ratio_From_Network'] = np.nan
                    else:
                        df_loop_flagprep.loc[batch_mask_flagprep, 'Adjusted_Diff_from_network'] = np.nan
                        df_loop_flagprep.loc[batch_mask_flagprep, 'Adjusted_Ratio_From_Network'] = np.nan
                batch_flags_this_batch = {} 
                for coord_loop_flag, df_loop_flag in all_data_current_window_processed.items():
                    batch_mask_flagging = (df_loop_flag.index >= batch_start) & (df_loop_flag.index < batch_end)
                    if not batch_mask_flagging.any():
                        batch_flags_this_batch[coord_loop_flag] = False 
                        df_loop_flag.loc[batch_mask_flagging, 'Batch_Flag'] = False; continue
                    df_slice_for_flagging = {coord_loop_flag: df_loop_flag.loc[batch_mask_flagging].copy()}
                    if df_slice_for_flagging[coord_loop_flag].empty:
                        df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = False
                        batch_flags_this_batch[coord_loop_flag] = False; continue
                    flagged_slice_dict = flag_anomalies(df_slice_for_flagging) 
                    if coord_loop_flag in flagged_slice_dict and 'Flagged' in flagged_slice_dict[coord_loop_flag].columns:
                        df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = flagged_slice_dict[coord_loop_flag]['Flagged'].reindex(df_loop_flag.loc[batch_mask_flagging].index).fillna(False)
                    else: df_loop_flag.loc[batch_mask_flagging, 'Flagged'] = False
                    is_batch_faulty = False
                    if 'Flagged' in df_loop_flag.columns:
                        df_batch_current_sensor = df_loop_flag.loc[batch_mask_flagging]
                        if not df_batch_current_sensor.empty:
                            flagged_points = df_batch_current_sensor['Flagged'].sum()
                            total_points = len(df_batch_current_sensor)
                            percent_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                            if percent_flagged > CFG.FAULTY_GAUGE_THRESHOLD_PERCENT: is_batch_faulty = True
                    batch_flags_this_batch[coord_loop_flag] = is_batch_faulty
                    df_loop_flag.loc[batch_mask_flagging, 'Batch_Flag'] = is_batch_faulty
                f_reg, valid_sensor_count_for_freg = compute_regional_adjustment(
                    all_data_current_window_processed, batch_start, batch_end, batch_flags_this_batch)
                logger.debug(f"Batch {batch_start}-{batch_end}: f_reg = {f_reg:.4f} from {valid_sensor_count_for_freg} valid sensors.")
                batch_data_cache_for_final = {}
                for coord_cache, df_cache in all_data_current_window_processed.items():
                    batch_mask_cache = (df_cache.index >= batch_start) & (df_cache.index < batch_end)
                    if batch_mask_cache.any():
                        cols_to_cache = ['Alpha', 'Flagged', 'Radar_Data_mm_per_min']
                        batch_data_cache_for_final[coord_cache] = df_cache.loc[batch_mask_cache, [c for c in cols_to_cache if c in df_cache.columns]].copy()
                for coord_loop_final_target, df_target_loop_final in all_data_current_window_processed.items(): 
                    batch_mask_final = (df_target_loop_final.index >= batch_start) & (df_target_loop_final.index < batch_end)
                    if not batch_mask_final.any(): continue
                    if 'Radar_Data_mm_per_min' in df_target_loop_final.columns:
                        df_target_loop_final.loc[batch_mask_final, 'Radar_Freg_Adjusted'] = \
                            df_target_loop_final.loc[batch_mask_final, 'Radar_Data_mm_per_min'].fillna(0) * f_reg
                    neighbors_final = get_nearest_neighbors(coord_loop_final_target, valid_coordinate_locations_utm_window, n_neighbors=CFG.N_NEIGHBORS)
                    current_batch_index_final = df_target_loop_final.loc[batch_mask_final].index
                    if current_batch_index_final.empty: continue
                    num_valid_neighbors_series = pd.Series(0, index=current_batch_index_final, dtype=int)
                    local_factor_series = pd.Series(1.0, index=current_batch_index_final, dtype=float)
                    valid_neighbor_alphas_for_local = {}
                    neighbor_is_valid_timestep_mask = pd.DataFrame(index=current_batch_index_final)
                    for n_coord in neighbors_final:
                        if n_coord == coord_loop_final_target: continue 
                        if batch_flags_this_batch.get(n_coord, True):
                            neighbor_is_valid_timestep_mask[n_coord] = False; continue
                        neighbor_df_from_cache = batch_data_cache_for_final.get(n_coord)
                        if neighbor_df_from_cache is None or neighbor_df_from_cache.empty:
                            neighbor_is_valid_timestep_mask[n_coord] = False; continue
                        n_alpha = neighbor_df_from_cache.get('Alpha').reindex(current_batch_index_final)
                        n_flagged = neighbor_df_from_cache.get('Flagged').reindex(current_batch_index_final)
                        valid_mask_for_n_timestep = pd.Series(True, index=current_batch_index_final)
                        if n_flagged is not None: valid_mask_for_n_timestep &= ~n_flagged.fillna(True)
                        if n_alpha is not None:
                            valid_mask_for_n_timestep &= n_alpha.notna(); valid_mask_for_n_timestep &= (n_alpha > 0)
                        else: valid_mask_for_n_timestep = pd.Series(False, index=current_batch_index_final)
                        neighbor_is_valid_timestep_mask[n_coord] = valid_mask_for_n_timestep
                        if n_alpha is not None and valid_mask_for_n_timestep.any():
                            valid_neighbor_alphas_for_local[n_coord] = n_alpha
                    if not neighbor_is_valid_timestep_mask.empty:
                        num_valid_neighbors_series = neighbor_is_valid_timestep_mask.sum(axis=1).astype(int)
                    if valid_neighbor_alphas_for_local:
                        df_valid_n_alphas = pd.DataFrame(valid_neighbor_alphas_for_local)
                        valid_keys_for_mask = [key for key in valid_neighbor_alphas_for_local.keys() if key in neighbor_is_valid_timestep_mask.columns]
                        if valid_keys_for_mask:
                            masked_n_alphas = df_valid_n_alphas.where(neighbor_is_valid_timestep_mask[valid_keys_for_mask])
                            local_factor_series = masked_n_alphas.median(axis=1, skipna=True).fillna(1.0).clip(lower=0.1)
                        else: local_factor_series.fillna(1.0, inplace=True) # Default if no valid keys
                    else: local_factor_series.fillna(1.0, inplace=True) # Default if no valid data
                    weight_conditions = [num_valid_neighbors_series == 1, num_valid_neighbors_series == 2, num_valid_neighbors_series >= 3]
                    weight_choices = [1.0/3.0, 2.0/3.0, 1.0]
                    weight_series = pd.Series(np.select(weight_conditions, weight_choices, default=0.0), index=current_batch_index_final)
                    f_reg_series = pd.Series(f_reg, index=current_batch_index_final)
                    factor_combined = local_factor_series * weight_series + f_reg_series * (1.0 - weight_series)
                    if 'Radar_Data_mm_per_min' in df_target_loop_final.columns:
                        raw_radar_in_batch = df_target_loop_final.loc[batch_mask_final, 'Radar_Data_mm_per_min'].fillna(0)
                        df_target_loop_final.loc[batch_mask_final, 'Final_Adjusted_Rainfall'] = raw_radar_in_batch * factor_combined
            # --- End Batch Loop for current window---
                # pass # Replace this pass with your actual batch processing logic
            logger.info("--- Finished Batch Processing Loop for Window ---")
        else: logger.warning("Batch Loop: No batch periods generated for window.")

    # --- Calculate Final Rolling Error Metrics on Windowed Data ---
    logger.info("--- Calculating Final Rolling Error Metrics on Windowed Data ---")
    # ... (Your Step 7 from original script, operating on `all_data_current_window_processed`) ...
    logger.info("--- Calculating Final Rolling Error Metrics on Windowed Data ---")
    # This loop now operates on the data processed for the current window
    for coord, df_orig_win_err in tqdm(all_data_current_window_processed.items(), desc="Calculating Rolling Errors Window"):
        df_err = df_orig_win_err.copy() # Work on a copy of the windowed data for this sensor
        if df_err.empty:
            logger.warning(f"DataFrame for {coord} (Window) is empty. Skipping rolling error calculation.")
            all_data_current_window_processed[coord] = df_err # Ensure empty df is stored back if it was empty
            continue

        # Check if necessary columns exist and have data
        # 'Network_Adjusted_Radar' is a key output of your batch processing steps
        if 'Network_Adjusted_Radar' in df_err.columns and 'Gauge_Data_mm_per_min' in df_err.columns:
            gauge_filled = df_err['Gauge_Data_mm_per_min'].ffill().bfill()
            net_adj_filled = df_err['Network_Adjusted_Radar'].ffill().bfill()

            # Proceed only if both series have some non-NaN data after filling
            if not net_adj_filled.isnull().all() and not gauge_filled.isnull().all():
                abs_error = abs(net_adj_filled - gauge_filled)
                df_err['Rolling_Abs_Error'] = abs_error.rolling(
                    window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1
                ).mean()
                
                if 'Flagged' in df_err.columns:
                    flag_points_numeric = df_err['Flagged'].astype(float)
                    df_err['Rolling_Prop_Flagged'] = flag_points_numeric.rolling(
                        window=CFG.ROLLING_WINDOW_ERROR_METRICS, center=True, min_periods=1
                    ).mean()
                else:
                    df_err['Rolling_Prop_Flagged'] = np.nan # Initialize column if 'Flagged' is missing
                    logger.warning(f"'Flagged' column missing for {coord} (Window) when calculating Rolling_Prop_Flagged.")
            else:
                # If either key series is all NaN, rolling metrics will be NaN
                df_err['Rolling_Abs_Error'] = np.nan
                df_err['Rolling_Prop_Flagged'] = np.nan
                logger.info(f"Insufficient data (all NaNs in key series) for rolling error metrics for {coord} (Window).")
        else:
            # If essential input columns are missing, initialize metrics to NaN
            df_err['Rolling_Abs_Error'] = np.nan
            df_err['Rolling_Prop_Flagged'] = np.nan
            missing_cols_msg = []
            if 'Network_Adjusted_Radar' not in df_err.columns: missing_cols_msg.append('Network_Adjusted_Radar')
            if 'Gauge_Data_mm_per_min' not in df_err.columns: missing_cols_msg.append('Gauge_Data_mm_per_min')
            logger.warning(f"Missing column(s): {', '.join(missing_cols_msg)} for {coord} (Window). Cannot calculate rolling error metrics.")
        
        all_data_current_window_processed[coord] = df_err # Store the windowed DataFrame with (potentially new) rolling error columns back
    logger.info("--- Finished Calculating Final Rolling Error Metrics for Window ---")
    # --- Update Master Data Dictionary ---
    logger.info("--- Merging window processing results back into master data ---")
    for coord_str, df_window_result in all_data_current_window_processed.items():
        if coord_str in master_sensor_data_dict_input and isinstance(df_window_result, pd.DataFrame) and not df_window_result.empty:
            master_df = master_sensor_data_dict_input[coord_str]
            for col in df_window_result.columns:
                if col not in master_df.columns:
                    master_df[col] = pd.Series(index=master_df.index, dtype=df_window_result[col].dtype).fillna(False if df_window_result[col].dtype == bool else np.nan)
            master_df.update(df_window_result.reindex(master_df.index)) # Align and update
        elif isinstance(df_window_result, pd.DataFrame) and df_window_result.empty:
            logger.debug(f"Window result for {coord_str} was empty. No update to master.")

    # --- Save WINDOWED Detailed Output CSVs ---
    # This uses `all_data_current_window_processed` which contains the data for THE CURRENT WINDOW ONLY.
    # The `actual_data_max_timestamp_in_window` is the max timestamp of data within this window.
    date_str_for_window_outputs = actual_data_max_timestamp_in_window.strftime('%Y%m%d_%H%M')
    logger.info(f"--- Saving Detailed Output CSV Files for WINDOW ending {date_str_for_window_outputs} ---")
    
    # In your original script, this was: output_csv_dir = CFG.RESULTS_DIR / "detailed_sensor_output"
    # If you want window-specific folders for these CSVs:
    # window_csv_output_dir = Path(CFG.RESULTS_DIR) / "detailed_sensor_output_windowed" / f"window_{date_str_for_window_outputs}"
    # For now, let's stick to your original single output directory for these detailed CSVs.
    # Ensure CFG.RESULTS_DIR is a Path object.
    detailed_csv_output_dir = Path(CFG.RESULTS_DIR) / "detailed_sensor_output"
    detailed_csv_output_dir.mkdir(parents=True, exist_ok=True)

    # These are the columns your original script saved for detailed output
    cols_to_save_detailed_csv = [
        'Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Alpha',
        'Batch_Alpha_Adjusted_Radar', 'Adjusted_Diff_from_network',
        'Adjusted_Ratio_From_Network', 'Flagged', 'Batch_Flag',
        'Radar_Freg_Adjusted', 'Final_Adjusted_Rainfall'
        # Your original list also had: 'Rolling_Abs_Error', 'Rolling_Prop_Flagged'
        # Add them back if they are calculated and you want them in these CSVs.
        # 'Rolling_Abs_Error', 'Rolling_Prop_Flagged' # If calculated
    ]
    
    for coord_str_csv, df_window_for_csv in tqdm(all_data_current_window_processed.items(), desc="Saving Windowed CSVs"):
        if df_window_for_csv.empty:
            logger.debug(f"DataFrame for {coord_str_csv} (Window) is empty. Skipping detailed CSV save.")
            continue

        save_cols_present_in_df = [col for col in cols_to_save_detailed_csv if col in df_window_for_csv.columns]
        if not save_cols_present_in_df:
            logger.warning(f"No columns to save for detailed CSV for {coord_str_csv} (Window).")
            continue

        df_to_save_csv = df_window_for_csv[save_cols_present_in_df].copy()
        
        # Get channel description using the static, full maps passed into this function
        channel_desc_value = "Unknown_Channel"
        utm_tuple = coordinate_locations_utm_all_static.get(coord_str_csv) # coord_str_csv is already string
        if utm_tuple: # utm_tuple is (int, int)
            channel_desc_value = utm_to_channel_desc_map_all_static.get(utm_tuple, "Description_Not_Found")
        else:
            logger.warning(f"Could not find UTM tuple for coord '{coord_str_csv}' in static coordinate_locations map for CSV.")
            channel_desc_value = "UTM_Coord_Not_Found"
        
        df_to_save_csv['Channel_Description'] = str(channel_desc_value)
        
        safe_coord_fname_part = coord_str_csv.replace('(','').replace(')','').replace(',','_').replace(' ','')
        # You might want to make these filenames window-specific too, or they will overwrite.
        # For now, using original naming.
        output_csv_filename = detailed_csv_output_dir / f"{safe_coord_fname_part}_data_adjustments_window_{date_str_for_window_outputs}.csv" # Added window date

        try:
            first_cols_csv = ['Channel_Description']
            remaining_cols_csv = [col for col in df_to_save_csv.columns if col not in first_cols_csv]
            final_cols_order_csv = first_cols_csv + remaining_cols_csv
            df_to_save_csv[final_cols_order_csv].to_csv(output_csv_filename, date_format='%Y-%m-%d %H:%M:%S%z') # Save with TZ
            logger.debug(f"Saved detailed window data for {coord_str_csv} to {output_csv_filename}")
        except Exception as e_csv_final_save:
            logger.error(f"Failed to save detailed CSV for {coord_str_csv} (Window): {e_csv_final_save}", exc_info=True)
    
    logger.info(f"--- Finished Saving Detailed Output CSV Files for Window to {detailed_csv_output_dir} ---")
    run_end_date_str_for_naming = actual_data_max_timestamp_in_window.strftime('%Y%m%d_%H%M')
    logger.info(f"--- Generating Visualizations using full master data (reflecting updates from window ending {run_end_date_str_for_naming}) ---")
    
    # Create channel map for all sensors in master_sensor_data_dict_input
    # This map will be used by the plotting functions.
    coord_map_for_full_plots = {}
    if utm_to_channel_desc_map_all_static and coordinate_locations_utm_all_static:
        for c_str_plot in master_sensor_data_dict_input.keys(): # Iterate over keys in master data
            utm_tuple_val = coordinate_locations_utm_all_static.get(c_str_plot)
            if utm_tuple_val:
                 coord_map_for_full_plots[c_str_plot] = utm_to_channel_desc_map_all_static.get(utm_tuple_val, "Unknown Channel")
            else:
                 coord_map_for_full_plots[c_str_plot] = "UTM_Coord_Not_Found" # Should not happen if coord_str comes from valid_coords
    else:
        logger.warning("Static UTM or Channel Description maps missing for plotting.")

    # Define output directory for plots generated in this run
    # Ensure CFG.PLOTS_OUTPUT_DIR is a Path object
    plots_output_dir_for_this_run = Path(CFG.PLOTS_OUTPUT_DIR) / f"run_{run_end_date_str_for_naming}"
    plots_output_dir_for_this_run.mkdir(parents=True, exist_ok=True)
    
    # Temporarily set/override CFG.ADJUSTMENT_PLOTS_DIR if create_plots_with_error_markers uses it directly
    # Otherwise, if create_plots_with_error_markers takes an output_dir argument, pass plots_output_dir_for_this_run
    original_cfg_adj_plots_dir = getattr(CFG, 'ADJUSTMENT_PLOTS_DIR', None) # Save original if exists
    CFG.ADJUSTMENT_PLOTS_DIR = plots_output_dir_for_this_run # Set for the duration of this plotting

    try:
        # Prepare data for plotting: use copies of the master data
        data_for_plotting_final = {
            k: v.copy() for k, v in master_sensor_data_dict_input.items() 
            if isinstance(v, pd.DataFrame) and not v.empty
        }
        
        if data_for_plotting_final:
            logger.info(f"Calling create_plots_with_error_markers with {len(data_for_plotting_final)} dataframes (full history from master).")
            # Pass the full static coordinate_locations_utm_all_static map
            create_plots_with_error_markers(
                all_data=data_for_plotting_final,
                coordinate_locations_utm=coordinate_locations_utm_all_static, # Full static map
                coord_to_channel_map=coord_map_for_full_plots,         # Full map derived above
                events_df=None, # Pass your events_df if you have one
                all_data_iter0=None # Pass if you have it
            )
            # The create_plots_with_error_markers function should use CFG.ADJUSTMENT_PLOTS_DIR (now set to plots_output_dir_for_this_run)

            logger.info(f"Full period plots saved to {plots_output_dir_for_this_run}.")

            if gdf_wgs84_all_static is not None and not gdf_wgs84_all_static.empty:
                # Ensure CFG.DASHBOARD_DIR is a Path object
                dashboard_filename_for_this_run = Path(CFG.DASHBOARD_DIR) / f"dashboard_adj_{run_end_date_str_for_naming}.html"
                dashboard_filename_for_this_run.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                
                generate_html_dashboard(
                    all_data=data_for_plotting_final, # Use the full master data for the dashboard content
                    coordinate_locations_utm=coordinate_locations_utm_all_static, 
                    gdf_wgs84=gdf_wgs84_all_static, # Pass the full static GeoDataFrame
                    svk_coords_utm=svk_coords_utm_all_static, # Pass the full static list
                    output_file=str(dashboard_filename_for_this_run)
                )
                logger.info(f"Generated dashboard: {dashboard_filename_for_this_run}")
        else:
            logger.warning("No data available in master_sensor_data_dict for plotting.")

    except Exception as e_plot_main_phase:
        logger.error(f"Error during main plotting/dashboard generation phase: {e_plot_main_phase}", exc_info=True)
    finally:
        # Restore original CFG.ADJUSTMENT_PLOTS_DIR if it was changed
        if original_cfg_adj_plots_dir is not None:
             CFG.ADJUSTMENT_PLOTS_DIR = original_cfg_adj_plots_dir
        elif hasattr(CFG, 'ADJUSTMENT_PLOTS_DIR') and CFG.ADJUSTMENT_PLOTS_DIR == plots_output_dir_for_this_run :
             # If it was only set for this run and didn't exist before
             delattr(CFG, 'ADJUSTMENT_PLOTS_DIR') 
    return True, True, actual_data_max_timestamp_in_window

# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main live adjustment pipeline script.")
    parser.add_argument('--run-once', action='store_true', help='Run the adjustment cycle once and exit.')
    args = parser.parse_args()
    args.run_once = True

    logger.info("===== Pipeline Start =====")
    logger.info("Loading static metadata...")
    target_coords_static_raw, coord_loc_utm_all_static_raw = load_target_coordinates()
    sensordata_static, gdf_wgs84_all_static = load_sensor_data()
    utm_to_chan_desc_all_static_raw, svk_coords_utm_all_static = process_sensor_metadata(sensordata_static)
    
    target_coords_static = {str(k): v for k,v in target_coords_static_raw.items()}
    coordinate_locations_utm_all_static = {str(k): v for k,v in coord_loc_utm_all_static_raw.items()}
    utm_to_channel_desc_map_all_static = utm_to_chan_desc_all_static_raw
    logger.info("Static metadata loaded.")

    master_sensor_data = load_and_initialize_master_data(
        list_of_target_coords=list(target_coords_static.keys()),
        pkl_dir=CFG.MASTER_PKL_DIR,
        pkl_pattern=CFG.MASTER_PKL_PATTERN,
        combined_csv_dir=CFG.COMBINED_DATA_DIR
    )

    if args.run_once:
        logger.info("===== Running a single live adjustment cycle =====")
        current_processing_start_utc = load_last_actual_data_timestamp()
        now_calc = pd.Timestamp.now(tz='UTC')
        latency_buffer = pd.Timedelta(hours=getattr(CFG, 'PROCESSING_WINDOW_END_BUFFER_HOURS', 1))
        time_res = getattr(CFG, 'TIME_RESOLUTION', '1min')
        current_processing_end = (now_calc - latency_buffer).floor(time_res)

        if current_processing_start_utc >= current_processing_end:
            logger.info(f"No new window. Last processed: {current_processing_start_utc}. Window end: {current_processing_end}.")
        else:
            completed, processed_flag, actual_max_ts = run_daily_adjustment_cycle(
                master_sensor_data_dict_input=master_sensor_data,
                processing_start_utc=current_processing_start_utc,
                processing_end_for_this_cycle=current_processing_end,
                coordinate_locations_utm_all_static=coordinate_locations_utm_all_static,
                utm_to_channel_desc_map_all_static=utm_to_channel_desc_map_all_static,
                gdf_wgs84_all_static=gdf_wgs84_all_static,
                svk_coords_utm_all_static=svk_coords_utm_all_static
            )
            if completed:
                logger.info(f"Cycle completed. Data processed: {processed_flag}. Actual max TS: {actual_max_ts}.")
                if processed_flag and pd.notna(actual_max_ts) and actual_max_ts > current_processing_start_utc:
                    save_last_actual_data_timestamp(actual_max_ts)
                    save_master_data_to_pkl(
                        master_data_to_save=master_sensor_data,
                        pkl_dir=CFG.MASTER_PKL_DIR,
                        pkl_pattern=CFG.MASTER_PKL_PATTERN
                    )
                else: logger.info("No new data effectively processed or max TS didn't advance. State/PKLs not updated.")
            else: logger.error(f"Cycle FAILED for window ending {current_processing_end}."); sys.exit(1)
        logger.info("===== Single live adjustment cycle finished. =====")
    else:
        logger.warning("Looping mode placeholder. Implement full loop if needed.")
    logger.info("===== Pipeline Shutdown =====")

# --- END OF COMPLETE main_v3_batch3_live.py ---