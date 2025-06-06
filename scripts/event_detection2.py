# event_detection.py
import pandas as pd
from tqdm import tqdm
from config import CFG
import numpy as np

def identify_and_flag_rain_events(all_data: dict) -> list:
    """
    Identifies rain events by merging core rain periods separated by short dry spells.
    Flags events with high anomaly percentages.
    Calculates the minimum number of valid neighbors used during each event.
    """
    print("Identifying rain events...")
    all_events = []
    pd_dry_duration = pd.Timedelta(CFG.EVENT_DRY_PERIOD_DURATION)
    pd_min_rain_duration = pd.Timedelta(CFG.EVENT_MIN_RAIN_DURATION)

    required_cols_base = ['Radar_Data_mm_per_min', 'Flagged']
    neighbor_count_col = 'Neighbor_Count_Used' # Column calculated in network_iterative

    for coord, df in tqdm(all_data.items()):
        # Basic checks
        if not all(col in df.columns for col in required_cols_base) or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            print(f"Warning: Skipping event detection for {coord}, missing base columns, empty data, or invalid index.")
            continue
        df = df.sort_index()

        # Calculate smoothed radar for detection
        smoothed_radar = df['Radar_Data_mm_per_min'].rolling(
            window=CFG.EVENT_DETECT_SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        ).mean().fillna(0)

        is_raining = smoothed_radar >= CFG.EVENT_RAIN_THRESHOLD_MM_MIN
        block_ids = (is_raining.diff().fillna(False) != 0).cumsum()
        blocks_info = []
        for block_id in block_ids.unique():
            block_slice = df[block_ids == block_id]
            if block_slice.empty: continue
            start_time, end_time = block_slice.index[0], block_slice.index[-1]
            duration = end_time - start_time
            block_type = 'rain' if is_raining.loc[start_time] else 'dry'
            peak_radar = block_slice['Radar_Data_mm_per_min'].max() if block_type == 'rain' else 0
            blocks_info.append({
                "id": block_id, "start": start_time, "end": end_time,
                "duration": duration, "type": block_type, "peak": peak_radar
            })
        if not blocks_info: continue

        # Identify qualifying rain blocks and significant dry blocks
        qualifying_rain_ids = {
            b["id"] for b in blocks_info
            if b["type"] == 'rain' and b["duration"] >= pd_min_rain_duration and b["peak"] >= CFG.EVENT_MIN_PEAK_RATE_MM_MIN
        }
        significant_dry_ids = {
            b["id"] for b in blocks_info
            if b["type"] == 'dry' and b["duration"] >= pd_dry_duration
        }

        current_event_blocks = []
        event_start_time = None

        # --- Loop through blocks to form events ---
        for i, block in enumerate(blocks_info):
            is_qualifying_rain = block["id"] in qualifying_rain_ids
            is_significant_dry = block["id"] in significant_dry_ids

            process_event = False # Flag to process the collected blocks as an event

            if is_qualifying_rain:
                if not current_event_blocks: # Start of a potential new event
                    event_start_time = block["start"]
                current_event_blocks.append(block)
            elif not is_significant_dry and current_event_blocks: # Merge short dry spell
                current_event_blocks.append(block)
            elif is_significant_dry and current_event_blocks: # End event due to long dry spell
                 process_event = True
            # Also process if it's the last block and part of an event
            elif i == len(blocks_info) - 1 and current_event_blocks:
                 process_event = True


            if process_event and current_event_blocks: # Make sure there are blocks to process
                 # Finalize previous event
                 if event_start_time is None: # Should be set, but double check
                     event_start_time = current_event_blocks[0]["start"]
                 event_end_time = current_event_blocks[-1]["end"]

                 # Slice the DataFrame for the event period
                 event_slice = df.loc[event_start_time : event_end_time]

                 if not event_slice.empty:
                     # Calculate basic event stats
                     total_points = len(event_slice)
                     flagged_points = int(event_slice['Flagged'].sum())
                     percentage_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                     is_flagged_event = percentage_flagged >= CFG.EVENT_FLAG_PERCENTAGE_THRESHOLD

                     # --- Corrected: Calculate Min Neighbor Count ---
                     min_neighbors_event = np.nan # Default to NaN
                     if neighbor_count_col in event_slice.columns:
                         neighbor_counts_in_event = event_slice[neighbor_count_col].dropna()
                         # This check needs to be properly indented:
                         if not neighbor_counts_in_event.empty:
                             min_neighbors_event = int(neighbor_counts_in_event.min())
                     # --- End Correction ---

                     # Append event details to the list
                     all_events.append({
                         "sensor_coord": coord,
                         "event_start": event_start_time,
                         "event_end": event_end_time,
                         "peak_radar": event_slice['Radar_Data_mm_per_min'].max(),
                         "duration_minutes": (event_end_time - event_start_time).total_seconds() / 60.0,
                         "total_points": total_points,
                         "flagged_points": flagged_points,
                         "percentage_flagged": round(percentage_flagged, 2),
                         "is_flagged_event": is_flagged_event,
                         "min_neighbors_during_event": min_neighbors_event # Add the calculated value
                     })

                 # Reset for the next potential event
                 current_event_blocks, event_start_time = [], None

        # --- End Block Loop ---


    print(f"Identified {len(all_events)} rain events across all sensors.")
    return all_events
