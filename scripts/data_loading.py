# data_loading.py
import re
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from config import CFG
from pathlib import Path
import logging
import traceback

# set up logging in a seperate text file
logging.basicConfig(
    filename=CFG.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a logger
logger = logging.getLogger(__name__)
# Set the logging level to INFO
logger.setLevel(logging.INFO)


def load_sensor_data(excel_file: str = str(CFG.EXCEL_FILE)):
    """Load sensor metadata from Excel and return both a DataFrame and a GeoDataFrame."""
    print(f"Loading sensor metadata from: {excel_file}")

    try:
        sensordata = pd.read_excel(excel_file)
    except Exception as e:
        print(f"Error reading Excel file {excel_file}: {e}")
        raise
    if 'wkt_geom' not in sensordata.columns:
        raise ValueError("Missing 'wkt_geom' column")
    try:
        gdf_initial = gpd.GeoDataFrame(sensordata, geometry=gpd.GeoSeries.from_wkt(sensordata['wkt_geom']),
                                       crs=CFG.SENSOR_CRS_INITIAL)
    except Exception:
        print("WKT parsing failed, attempting regex fallback...")
        coords = sensordata['wkt_geom'].str.extract(r'Point\s*\(([\d\.\-]+)\s+([\d\.\-]+)\)')
        if coords.isnull().any().any():
            raise ValueError("Coordinate extraction failed")
        gdf_initial = gpd.GeoDataFrame(
            sensordata,
            geometry=[Point(lon, lat) for lon, lat in zip(coords[0].astype(float), coords[1].astype(float))],
            crs=CFG.SENSOR_CRS_INITIAL
        )
    gdf_utm = gdf_initial.to_crs(crs=CFG.SENSOR_CRS_PROJECTED)
    sensordata['x'], sensordata['y'] = gdf_utm.geometry.x.astype(int), gdf_utm.geometry.y.astype(int)
    sensordata['(x,y)_tuple'] = list(zip(sensordata['x'], sensordata['y']))
    gdf_wgs84 = gdf_initial.to_crs(epsg=4326)
    if 'Name' in sensordata.columns:
        sensordata['Gauge_ID'] = sensordata['Name'].str.extract(r'(\d+)', expand=False).astype(int)
    else:
        print("Warning: Cannot extract 'Gauge_ID'.")
        sensordata['Gauge_ID'] = None
    # print(sensordata.head())
    return sensordata, gdf_wgs84

def load_target_coordinates():
    """Load target coordinates from a text file."""
    target_coordinate_strings = []
    coordinate_locations_utm = {}
    coord_pattern_re = re.compile(CFG.SENSOR_COORD_REGEX)
    try:
        print(f"Loading target coordinates from: {CFG.ALL_COORDS_FILE}")
        with open(CFG.ALL_COORDS_FILE, 'r') as f:
            for line in f:
                coord_str = line.strip()
                if not coord_str:
                    continue
                match = coord_pattern_re.match(coord_str)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    target_coordinate_strings.append(coord_str)
                    coordinate_locations_utm[coord_str] = (x, y)
                else:
                    print(f"Warning: Could not parse coordinate: '{coord_str}'")
    except FileNotFoundError:
        print(f"Fatal: File not found: {CFG.ALL_COORDS_FILE}")
        raise
    if not target_coordinate_strings:
        raise ValueError("No valid coordinates loaded.")
    print(f"Loaded {len(target_coordinate_strings)} target coordinates.")
    return target_coordinate_strings, coordinate_locations_utm


# def process_sensor_metadata(sensordata: pd.DataFrame):
#     """Map sensor channels and return sensor_channels and list of SVK coordinates."""
#     utm_to_channel_description = {} # Dictionary to map (x, y) to channel description
#     sensor_channels = {}
#     svk_coords_utm = []

#     if 'Channel' in sensordata.columns:
#         print("Warning: 'Channel' column found in metadata, but not used for mapping.")
#         return {}, []
    
#     svk_ids_numeric = {11130, 11131, 11132, 11133, 11134, 11135, 11136} # Assuming these are the relevant numeric IDs
#     numeric_channel_col = None
#     if 'Gauge_ID' in sensordata.columns: # Check if numeric ID was extracted from Name
#         numeric_channel_col = 'Gauge_ID'
#     if not sensordata.empty and all(c in sensordata.columns for c in ['x', 'y', 'Channel']):
#         for _, r in sensordata.iterrows():
#             # --- Create Mapping: (x, y) -> Channel Description ---
#             if pd.notna(r['x']) and pd.notna(r['y']) and pd.notna(r['Channel']):
#                 utm_tuple = (int(r['x']), int(r['y']))
#                 channel_description = str(r['Channel']) # Get the value from 'Channel' column as string

#                 # Store the mapping (overwrite if duplicate UTM tuple exists, use last value)
#                 utm_to_channel_description[utm_tuple] = channel_description

#                 if numeric_channel_col and pd.notna(r[numeric_channel_col]):
#                      if int(r[numeric_channel_col]) in svk_ids_numeric:
#                          if utm_tuple not in svk_coords_utm: # Avoid duplicates
#                             svk_coords_utm.append(utm_tuple)

#         # sensor_channels = {(r['x'], r['y']): r['Channel'] for _, r in sensordata.iterrows() if pd.notna(r['x']) and pd.notna(r['y']) and pd.notna(r['Channel'])}
#         # svk_ids = [11130, 11131, 11132, 11133, 11134, 11135, 11136]
#         # svk_coords_utm = [(r['x'], r['y']) for _, r in sensordata.iterrows() if r['Channel'] in svk_ids]
#     else:
#         print("Warning: Cannot map sensor channels/SVK from metadata.")
#     return utm_to_channel_description, svk_coords_utm

# data_loading.py

import re
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from config import CFG

# ...(load_sensor_data remains the same)...
# ...(load_target_coordinates remains the same)...

# --- CORRECTED process_sensor_metadata ---
def process_sensor_metadata(sensordata: pd.DataFrame):
    """
    Identify SVK coords and create a mapping from UTM tuple to the descriptive Channel name.

    Args:
        sensordata (pd.DataFrame): DataFrame loaded by load_sensor_data (must include 'x', 'y', 'Channel' columns).

    Returns:
        tuple: (utm_to_channel_description, svk_coords_utm)
            - utm_to_channel_description (dict): Maps (x, y) UTM tuple to the descriptive string from the 'Channel' column.
            - svk_coords_utm (list): List of (x, y) UTM tuples for SVK sensors (assuming SVK IDs are still relevant).
    """
    utm_to_channel_description = {} # Initialize the dictionary
    svk_coords_utm = []

    # --- Check required columns for mapping ---
    required_cols = ['x', 'y', 'Channel']
    if not all(c in sensordata.columns for c in required_cols):
        missing = [c for c in required_cols if c not in sensordata.columns]
        print(f"Warning: Cannot create channel description map. Missing columns in sensordata: {missing}")
        return {}, [] # Return empty results if essential columns missing

    # --- Optional: SVK Check Setup (Adapt if needed) ---
    svk_ids_numeric = {11130, 11131, 11132, 11133, 11134, 11135, 11136}
    numeric_channel_col = None
    if 'Gauge_ID' in sensordata.columns: # Check if numeric ID was extracted from Name
        numeric_channel_col = 'Gauge_ID'
        print("Using 'Gauge_ID' column for SVK check.")
    else:
        print("Warning: 'Gauge_ID' column not found for SVK check.")
    # --- End SVK Check Setup ---

    # --- Iterate and build the map ---
    print("Building UTM tuple to Channel description map...")
    for _, r in sensordata.iterrows():
        # --- Create Mapping: (x, y) -> Channel Description ---
        # Check for NaN in coordinates and Channel value
        if pd.notna(r['x']) and pd.notna(r['y']) and pd.notna(r['Channel']):
            try:
                utm_tuple = (int(r['x']), int(r['y']))
                channel_description = str(r['Channel']) # Get the value from 'Channel' column

                # Store the mapping
                utm_to_channel_description[utm_tuple] = channel_description

                # --- SVK Check (Optional, based on setup above) ---
                if numeric_channel_col and pd.notna(r[numeric_channel_col]):
                     try:
                         if int(r[numeric_channel_col]) in svk_ids_numeric:
                             if utm_tuple not in svk_coords_utm: # Avoid duplicates
                                svk_coords_utm.append(utm_tuple)
                     except ValueError:
                         pass # Ignore if Gauge_ID cannot be converted to int
                # --- End SVK Check ---

            except (ValueError, TypeError) as e:
                 print(f"Warning: Skipping row due to data conversion error: {e} - Row data: x={r.get('x')}, y={r.get('y')}, Channel={r.get('Channel')}")
        # else: # Optional: Print rows skipped due to NaNs
            # print(f"Skipping row due to NaN values: x={r.get('x')}, y={r.get('y')}, Channel={r.get('Channel')}")


    print(f"Finished building map. Found {len(utm_to_channel_description)} coordinate mappings.")
    if not utm_to_channel_description:
        print("Warning: The UTM tuple to Channel description map is empty. Check input data and column names ('x', 'y', 'Channel').")

    return utm_to_channel_description, svk_coords_utm # Return the populated map

# ...(load_time_series_data remains the same)...


def load_time_series_data(target_coords: list, locations_utm: dict):
    """
    Load time series data for each sensor from pickle files.
    Ensures the loaded DataFrame has a UTC-aware DatetimeIndex.
    """
    all_data = {}
    missing_files = []
    logger.info(f"Attempting to load time series from: {CFG.PKL_DATA_DIR}")

    if not isinstance(CFG.PKL_DATA_DIR, Path):
        pkl_dir = Path(CFG.PKL_DATA_DIR)
    else:
        pkl_dir = CFG.PKL_DATA_DIR

    if not pkl_dir.exists():
        logger.warning(f"PKL directory not found: {pkl_dir}")
        return {}, {}, [] 

    for coord_str in target_coords:
        pkl_path = pkl_dir / f'all_data_{coord_str}.pkl'
        try:
            if pkl_path.is_file():
                df = pd.read_pickle(pkl_path)
                if df.empty:
                    logger.info(f"Pickle file for {coord_str} at {pkl_path} is empty. Skipping.")
                    all_data[coord_str] = df # Store empty df so it's not 'missing'
                    continue

                logger.debug(f"Loaded {coord_str}. Initial index type: {type(df.index)}, name: {df.index.name}, columns: {df.columns.tolist()}")

                # --- Ensure DatetimeIndex and set to 'time' if not already ---
                if not isinstance(df.index, pd.DatetimeIndex):
                    time_col_found = False
                    for col_name in ['time', 'datetime']: # Common names for time column
                        if col_name in df.columns:
                            logger.debug(f"Found '{col_name}' column in {coord_str}. Setting as index.")
                            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                            df.set_index(col_name, inplace=True)
                            df.dropna(subset=[df.index.name], inplace=True) # Drop NaT from conversion
                            time_col_found = True
                            break
                    if not time_col_found:
                        logger.warning(f"No 'time' or 'datetime' column found and index is not DatetimeIndex for {coord_str}. Trying to convert existing index.")
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df.dropna(subset=[df.index.name or 'index'], inplace=True)
                
                if df.empty:
                    logger.warning(f"DataFrame for {coord_str} became empty after index processing. Storing empty.")
                    all_data[coord_str] = df
                    continue

                # --- Ensure Index is UTC ---
                if isinstance(df.index, pd.DatetimeIndex): # Double check it's now a DatetimeIndex
                    if df.index.tz is None:
                        logger.debug(f"Localizing naive index of {coord_str} to UTC.")
                        df.index = df.index.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
                    elif str(df.index.tz).upper() != 'UTC':
                        logger.debug(f"Converting index of {coord_str} from {df.index.tz} to UTC.")
                        df.index = df.index.tz_convert('UTC')
                    
                    # Drop any NaTs created during timezone operations on the index itself
                    if df.index.hasnans:
                        logger.debug(f"Index of {coord_str} has NaNs after tz ops. Removing them.")
                        df = df[df.index.notna()]
                else:
                    logger.error(f"Failed to establish a DatetimeIndex for {coord_str}. Skipping.")
                    continue # Skip this DataFrame if index is still not datetime

                if df.empty:
                    logger.warning(f"DataFrame for {coord_str} empty after all index and TZ processing. Storing empty.")

                all_data[coord_str] = df
                logger.debug(f"Finalized index for {coord_str}: Type: {type(df.index)}, Name: {df.index.name}, TZ: {df.index.tz}, Min: {df.index.min() if not df.empty else 'N/A'}")

            else:
                missing_files.append(coord_str)
                logger.warning(f"Pickle file not found for {coord_str} at {pkl_path}")
        except Exception as e:
            logger.error(f"Error loading or processing PKL file {pkl_path}: {e}")
            logger.error(traceback.format_exc())
            # all_data[coord_str] = pd.DataFrame() # Optionally add empty DF on error
    
    logger.info(f"Finished loading time series. Successfully processed {len(all_data)} DataFrames out of {len(target_coords)} targets.")
    if missing_files:
        logger.warning(f"Missing PKL files for coordinates: {missing_files}")
        
    valid_locations_utm = {k: v for k, v in locations_utm.items() if k in all_data}
    
    return all_data, valid_locations_utm, missing_files

