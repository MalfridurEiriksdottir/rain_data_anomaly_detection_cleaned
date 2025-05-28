# event_detection.py
import pandas as pd
from tqdm import tqdm
from config import CFG
import numpy as np

def identify_and_flag_rain_events(all_data: dict) -> list:
    """
    Identifies rain events by merging core rain periods separated by short dry spells.
    Flags events with high anomaly percentages.
    """
    print("Identifying rain events...")
    all_events = []
    pd_dry_duration = pd.Timedelta(CFG.EVENT_DRY_PERIOD_DURATION)
    pd_min_rain_duration = pd.Timedelta(CFG.EVENT_MIN_RAIN_DURATION)

    for coord, df in tqdm(all_data.items()):
        if 'Radar_Data_mm_per_min' not in df.columns or 'Flagged' not in df.columns or df.empty:
            print(f"Warning: Skipping event detection for {coord}, missing columns or empty data.")
            continue
        df = df.sort_index()

        # Calculate smoothed radar for detection using a centered rolling mean.
        smoothed_radar = df['Radar_Data_mm_per_min'].rolling(
            window=CFG.EVENT_DETECT_SMOOTHING_WINDOW,
            center=True,
            min_periods=1
        ).mean().fillna(0)
        
        # Determine if it's raining using the smoothed Radar_Data_mm_per_min.
        is_raining = smoothed_radar >= CFG.EVENT_RAIN_THRESHOLD_MM_MIN

        # Identify blocks (rain or dry) using cumulative sum.
        block_ids = (is_raining.diff().fillna(False) != 0).cumsum()
        blocks_info = []
        for block_id in block_ids.unique():
            block_slice = df[block_ids == block_id]
            if block_slice.empty:
                continue
            start_time, end_time = block_slice.index[0], block_slice.index[-1]
            duration = end_time - start_time
            block_type = 'rain' if is_raining.loc[start_time] else 'dry'
            peak_radar = block_slice['Radar_Data_mm_per_min'].max() if block_type == 'rain' else 0
            blocks_info.append({
                "id": block_id,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "type": block_type,
                "peak": peak_radar
            })
        if not blocks_info:
            continue

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
        for i, block in enumerate(blocks_info):
            is_qualifying_rain = block["id"] in qualifying_rain_ids
            is_significant_dry = block["id"] in significant_dry_ids

            if is_qualifying_rain:
                if not current_event_blocks:
                    # Start an event regardless of additional conditions.
                    event_start_time = block["start"]
                    current_event_blocks.append(block)
                else:
                    current_event_blocks.append(block)
            elif not is_significant_dry and current_event_blocks:
                current_event_blocks.append(block)  # Merge short dry spell
            elif is_significant_dry and current_event_blocks:
                # Finalize previous event
                # Ensure event_start_time is set.
                if event_start_time is None and current_event_blocks:
                    event_start_time = current_event_blocks[0]["start"]
                event_end_time = current_event_blocks[-1]["end"]
                event_slice = df.loc[event_start_time : event_end_time]
                if not event_slice.empty:
                    total_points = len(event_slice)
                    flagged_points = int(event_slice['Flagged'].sum())
                    percentage_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                    is_flagged_event = percentage_flagged >= CFG.EVENT_FLAG_PERCENTAGE_THRESHOLD

                    # >>>>>>>>>> GET MINIMUM NEIGHBOR COUNT USED <<<<<<<<<<
                    min_neighbors_event = np.nan # Default to NaN
                    if 'Neighbor_Count_Used' in event_slice.columns:
                            neighbor_counts_in_event = event_slice['Neighbor_Count_Used'].dropna()
                    if not neighbor_counts_in_event.empty:
                            # Calculate only the minimum
                            min_neighbors_event = int(neighbor_counts_in_event.min())
                         # else: neighbor count column existed but was all NaN for this slice
                     # else: neighbor count column missing entirely
                     # >>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
                        # >>>>>>>>>> ADD MIN COUNT TO EVENT DICT <<<<<<<<<<
                        "min_neighbors_during_event": min_neighbors_event # Use a clear name
                         # >>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<
                    })
                current_event_blocks, event_start_time = [], None

        # Finalize any remaining event
        if current_event_blocks:
            if event_start_time is None and current_event_blocks:
                event_start_time = current_event_blocks[0]["start"]
            event_end_time = current_event_blocks[-1]["end"]
            event_slice = df.loc[event_start_time : event_end_time]
            if not event_slice.empty:
                total_points = len(event_slice)
                flagged_points = int(event_slice['Flagged'].sum())
                percentage_flagged = (flagged_points / total_points) * 100 if total_points > 0 else 0
                is_flagged_event = percentage_flagged >= CFG.EVENT_FLAG_PERCENTAGE_THRESHOLD
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
                })

    print(f"Identified {len(all_events)} rain events across all sensors.")
    return all_events
