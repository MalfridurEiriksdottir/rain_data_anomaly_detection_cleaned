# utils.py
import pandas as pd
from config import CFG

def shift_radar_by_dst(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Adjust timestamps based on DST (CEST adds 2 hours, others add 1 hour)."""
    if index.tz is None:
        try:
            index = index.tz_localize('UTC').tz_convert('Europe/Copenhagen')
        except Exception as e:
            print(f"Warning: Could not localize index for DST shift: {e}")
    shift_hours = index.map(lambda ts: 2 if pd.notna(ts.tzname()) and ts.tzname() == 'CEST' else 1)
    return index + pd.to_timedelta(shift_hours, unit='h')


# utils/state.py
"""
Persist – and later recall – the timestamp of the last 24 h batch that the
pipeline successfully processed.

The information is stored as JSON next to your other result artefacts.
"""

from pathlib import Path
import json
import pandas as pd
from config import CFG            # uses CFG.RESULTS_DIR

# Where the tiny JSON file lives
STATE_FILE = Path("results_files") / "last_processed.json"

# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------
def get_last_processed_utc() -> pd.Timestamp | None:
    """
    Returns the UTC timestamp of the last processed batch, or None
    if this is the first ever run (i.e. the file does not exist).
    """
    if STATE_FILE.exists():
        with STATE_FILE.open() as fh:
            ts_str = json.load(fh).get("last_processed_utc")
        try:
            return pd.to_datetime(ts_str, utc=True)
        except Exception:         # malformed file → treat as “no state”
            return None
    return None


def set_last_processed_utc(ts: pd.Timestamp) -> None:
    """
    Overwrite the marker file with `ts` (must be timezone-aware UTC).
    Only called after the pipeline finishes successfully.
    """
    ts = ts.tz_convert("UTC")     # make absolutely sure it’s UTC
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as fh:
        json.dump({"last_processed_utc": ts.isoformat()}, fh)

