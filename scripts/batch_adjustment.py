
import pandas as pd

def compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags):
    total_gauge = 0.0
    total_radar = 0.0
    valid_sensor_count = 0

    for coord, df in all_data.items():
        is_unreliable = batch_flags.get(coord, True)
        if is_unreliable:
            continue

        df_batch = df.loc[(df.index >= batch_start) & (df.index < batch_end)]
        if df_batch.empty:
            continue

        if 'Gauge_Data_mm_per_min' in df_batch.columns and 'Radar_Data_mm_per_min' in df_batch.columns:
            current_gauge_sum = df_batch['Gauge_Data_mm_per_min'].sum()
            current_radar_sum = df_batch['Radar_Data_mm_per_min'].sum()

            if pd.notna(current_gauge_sum) and pd.notna(current_radar_sum):
                 total_gauge += current_gauge_sum
                 total_radar += current_radar_sum
                 valid_sensor_count += 1

    # --- Calculation based on filtered sums ---
    if total_radar == 0:
        f_reg = 1.0 # Default to 1.0 if no reliable radar sum
    else:
        f_reg = total_gauge / total_radar

    # Apply weighting ONLY if count is exactly 1 or 2.
    # If count is 0, f_reg should remain 1.0 (from default above).
    # If count is >= 3, f_reg should remain as calculated from the ratio.
    if valid_sensor_count == 1:
        f_reg *= (1.0 / 3.0)
    elif valid_sensor_count == 2:
        f_reg *= (2.0 / 3.0)

    return f_reg, valid_sensor_count