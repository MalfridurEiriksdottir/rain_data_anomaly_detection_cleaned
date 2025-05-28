# config.py
from pathlib import Path
import pandas as pd

class Config:
    """Holds configuration parameters for the anomaly detection process."""
    # --- File Paths ---
    BASE_DIR = Path("")
    RESULTS_DIR = BASE_DIR / 'results_files'
    DASHBOARD_DIR = BASE_DIR / 'dashboard_files'
    PKL_DATA_DIR: Path = BASE_DIR / 'all_data_pkl2'
    MASTER_PKL_DIR = PKL_DATA_DIR
    PKL_FILENAME_PATTERN: str = "all_data_({x},{y}).pkl" 
    COMBINED_DATA_DIR: Path = BASE_DIR / 'combined_data'
    COMBINED_FILENAME_PATTERN: str = r"combined_data_\((\d+),(\d+)\)\.csv"
    MASTER_PKL_PATTERN = PKL_FILENAME_PATTERN
    EXCEL_FILE: Path = BASE_DIR / 'SensorOversigt.xlsx'
    ALL_COORDS_FILE: Path = BASE_DIR / 'all_coords.txt'
    PLOTS_OUTPUT_DIR: Path = BASE_DIR / 'plots_output'
    FLAGGING_PLOTS_DIR: Path = BASE_DIR / 'flagging_plots'
    DASHBOARD_FILE: Path = DASHBOARD_DIR / 'rainfall_anomaly_dashboard.html'
    FLAGGING_DASHBOARD_FILE: Path = DASHBOARD_DIR / 'flagging_dashboard.html'
    SUMMARY_CSV_FILE: Path = RESULTS_DIR / 'summary_df.csv'
    DEBUG_ALPHA_PLOT_FILE: Path = DASHBOARD_DIR / 'alpha_debug_plot.html'
    SENSOR_CSV_OUTPUT_TPL: str = '{coord}_data.csv'

    LOG_FILE: Path = BASE_DIR / 'logs' / 'anomaly_detection.log'

    RADAR_TIMESTAMPS_ARE_TRUE_UTC: bool = True  # Confirms assumption about input radar data
    SHIFT_RADAR_TO_CPH_LOCAL_NUMBERS_AS_UTC: bool = True

    # --- Feature Engineering ---
    rolling_window_double = 60.0
    rolling_window_int = int(rolling_window_double)
    ROLLING_WINDOW: str = f'{rolling_window_int}min'
    K_PARAM: float = 3.33/rolling_window_double
    MIN_GAUGE_THRESHOLD: float = 0.1
    FILLNA_METHOD: str = 'ffill'
    FILLNA_LIMIT: int = None

    # --- Network Comparison ---
    N_NEIGHBORS: int = 5
    DISTANCE_METRIC: str = 'utm'

    # --- Anomaly Flagging ---
    ABS_DIFF_THRESHOLD_MM_MIN: float = 0.5
    RATIO_THRESHOLD: float = 0.5
    ALPHA_DEV_CENTER: float = 0.56
    ALPHA_DEV_MARGIN: float = 0.3
    CONTINUITY_WINDOW: str = '15min'
    CONTINUITY_THRESHOLD_RATIO: float = 0.5
    GAUGE_ZERO_WINDOW: str = '60min'
    GAUGE_ZERO_PERCENTAGE: float = 0.6
    BOTH_ZERO_THRESHOLD_MM_MIN: float = 0.001
    EPSILON: float = 1e-6

    # --- Plotting ---
    MAP_DEFAULT_ZOOM: int = 10
    PLOT_TEMPLATE: str = "plotly_white"

    # --- Data Loading ---
    SENSOR_COORD_REGEX: str = r'\((\d+),(\d+)\)'
    SENSOR_CRS_INITIAL: str = "EPSG:4326"
    SENSOR_CRS_PROJECTED: str = "EPSG:25832"  # UTM32N

    # --- Event Detection ---
    EVENT_RAIN_THRESHOLD_MM_MIN: float = 0.1
    EVENT_DRY_PERIOD_DURATION: str = '45min'
    EVENT_MIN_RAIN_DURATION: str = '15min'
    EVENT_MIN_PEAK_RATE_MM_MIN: float = 0.5
    EVENT_FLAG_PERCENTAGE_THRESHOLD: float = 50.0

    # --- Event Visualization ---
    EVENT_NORMAL_COLOR: str = "rgba(180, 180, 180, 0.2)"
    EVENT_FLAGGED_COLOR: str = "rgba(255, 150, 150, 0.25)"

    EVENTS_CSV_FILE: Path = RESULTS_DIR / 'rain_events_summary.csv'
    FLAGGED_EVENTS_CSV_FILE: Path = RESULTS_DIR / 'flagged_rain_events_summary.csv'
    FLAGGED_EVENTS_DIR: Path = RESULTS_DIR / 'flagged_events_per_coordinate'
    FLAGGED_EVENTS_COORD_TPL: str = 'flagged_events_{coord}.csv'
    EVENT_DETECT_SMOOTHING_WINDOW: str = '10min'

    # --- Anomaly Flagging ---
    ADJ_RADAR_DIFF_THRESHOLD_MM_MIN: float = 3.0

    BATCH_DURATION = pd.Timedelta(hours=24)
    BATCH_OVERLAP = pd.Timedelta(hours=1)
    FAULTY_GAUGE_THRESHOLD_PERCENT = 8.0
    # FAULTY_GAUGE_THRESHOLD_PERCENT = 20.0


    BATCH_ADJUSTMENT_DIR = BASE_DIR / 'batch_adjustment_output'
    ADJUSTMENT_PLOTS_DIR = BASE_DIR / 'adjustment_plots'

CFG = Config()
