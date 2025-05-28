# results.py
import numpy as np
import pandas as pd
from tqdm import tqdm

def aggregate_results(all_data: dict) -> pd.DataFrame:
    """Aggregate anomaly detection results from all sensors into a summary DataFrame."""
    results = []
    print("Aggregating results...")
    for coord, df in tqdm(all_data.items()):
        if 'Flagged' not in df.columns or df.empty:
            continue
        total_time = len(df)
        flagged_time = df['Flagged'].sum()
        percent_flagged = (flagged_time / total_time) * 100 if total_time > 0 else 0
        avg_diff = df.loc[df['Flagged'], 'Difference'].mean() if flagged_time > 0 else np.nan
        max_diff_network = df['Difference_From_Network'].abs().max() if 'Difference_From_Network' in df.columns else np.nan
        results.append({
            'Coordinate': coord,
            'Total_Timepoints': total_time,
            'Flagged_Timepoints': flagged_time,
            'Percent_Flagged': percent_flagged,
            'Average_Difference_When_Flagged': avg_diff,
            'Max_Difference_From_Network': max_diff_network
        })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(by='Percent_Flagged', ascending=False)
