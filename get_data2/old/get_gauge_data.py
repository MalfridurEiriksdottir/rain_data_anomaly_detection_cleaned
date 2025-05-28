# Read the file SensorOversigt.xlsx using pandas

import pandas as pd
import os

# Read the file SensorOversigt.xlsx
df = pd.read_excel('SensorOversigt.xlsx')
from datetime import datetime, timedelta
GAUGE_FOLDER = "../gauge_data2"
DEFAULT_START_DATE = datetime(2024, 6, 21, 0, 0, 0) # Default if file is new/empty
DATE_FORMAT_API = "%Y-%m-%d %H:%M:%S"
DATE_FORMAT_CSV = "%Y-%m-%d %H:%M:%S"
API_TIMEOUT_SECONDS = 60

import os
import ssl
from urllib.request import urlopen
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyproj import Transformer

from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import quote
# from datetime import datetime, timedelta


ssl._create_default_https_context = ssl._create_unverified_context

# --- Helper Functions ---

def parse_coordinates(coord_string):
    """Extracts lon, lat from 'Point (lon lat)' string."""
    try:
        coord_string = str(coord_string).replace("Point (", "").replace(")", "")
        lon, lat = map(float, coord_string.split())
        return lon, lat
    except Exception as e:
        print(f"Error parsing coordinates '{coord_string}': {e}")
        return None, None

def transform_coordinates(lon, lat):
    """Transforms from WGS84 (EPSG:4326) to ETRS89 / UTM zone 32N (EPSG:25832)."""
    if lon is None or lat is None: return None, None
    try:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return int(x), int(y)
    except Exception as e:
        print(f"Error transforming coordinates ({lon}, {lat}): {e}")
        return None, None

def get_last_timestamp_from_csv(filepath):
    """Reads the last timestamp from the 'datetime' column of a CSV."""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath, usecols=['datetime'], parse_dates=['datetime'])
        if not df.empty:
            last_valid_time = df['datetime'].dropna().iloc[-1]
            if pd.notna(last_valid_time):
                return last_valid_time.tz_localize(None)
    except pd.errors.EmptyDataError:
        print(f"Info: CSV file {filepath} is empty.")
        return None
    except KeyError:
        print(f"Warning: 'datetime' column not found in {filepath}.")
        return None
    except Exception as e:
        print(f"Error reading last timestamp from {filepath}: {e}")
        return None
    return None


# def parse_coordinates(coord_string):
#     """Extracts lon, lat from 'Point (lon lat)' string."""
#     coord_string = coord_string.replace("Point (", "").replace(")", "")
#     lon, lat = map(float, coord_string.split())
#     return lon, lat


# def transform_coordinates(lon, lat):
#     """Transforms from WGS84 (EPSG:4326) to ETRS89 / UTM zone 32N (EPSG:25832)."""
#     transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     return x, y

# --- Main Function to Append New Data ---

def append_new_gauge_data(sensor_name, channel_no, coord_string):
    """
    Checks last timestamp, fetches new data from API, filters, and appends to CSV.
    """
    print(f"\n--- Processing Sensor: {sensor_name} (Channel: {channel_no}) ---")
    os.makedirs(GAUGE_FOLDER, exist_ok=True)

    lon, lat = parse_coordinates(coord_string)
    x, y = transform_coordinates(lon, lat)

    if x is None or y is None:
        print(f"ERROR: Skipping sensor '{sensor_name}' due to coordinate processing error.")
        return # Stop processing this sensor

    csv_filename = os.path.join(GAUGE_FOLDER, f"gaugedata_{x}_{y}_.csv")

    # --- Determine Start Date based on last record ---
    last_timestamp = get_last_timestamp_from_csv(csv_filename)
    if last_timestamp:
        start_date = last_timestamp + timedelta(seconds=1)
        print(f"Last record found: {last_timestamp}. Fetching from: {start_date}")
    else:
        start_date = DEFAULT_START_DATE
        print(f"No existing data found (or error reading timestamp). Fetching from default start: {start_date}")

    # --- Set End Date ---
    end_date = datetime.now()

    if start_date >= end_date:
        print(f"Start date ({start_date}) is not before end date ({end_date}). No new data to fetch.")
        return # Nothing to do

    # --- Prepare and Execute API Request (NO Reduction=day) ---
    service_url = "https://nkaflob.watermanager.dk/"
    date_from_str = start_date.strftime(DATE_FORMAT_API)
    date_to_str = end_date.strftime(DATE_FORMAT_API)
    encoded_date_from = quote(date_from_str)
    encoded_date_to = quote(date_to_str)
    # REMOVED &Reduction=day
    request_url = (
        f"{service_url}/Services/DataService.ashx?type=graph&channel={channel_no}"
        f"&DateFrom={encoded_date_from}"
        f"&DateTo={encoded_date_to}"
    )

    print(f"Fetching API data [{date_from_str} to {date_to_str}]...")

    try:
        req = Request(request_url)
        with urlopen(req, timeout=API_TIMEOUT_SECONDS) as f:
            if f.getcode() != 200:
                 print(f"ERROR: API request failed with status code {f.getcode()}.")
                 try: raw_response = f.read()
                 except Exception: raw_response = b"(Could not read error body)"
                 print(f"Response snippet: {raw_response[:500].decode('utf-8', errors='ignore')}")
                 return # Stop processing this sensor
            raw_response = f.read()
        response_text = raw_response.decode('utf-8', errors='ignore').strip()

        if not response_text:
            print("API returned an empty response. Assuming no new data.")
            return # Nothing to process

        # --- Parse Main JSON ---
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as json_e:
            print(f"ERROR: Failed to decode main JSON response. Error: {json_e}")
            print("-------------------- RAW RESPONSE START --------------------")
            print(response_text[:1000])
            if len(response_text) > 1000: print("...")
            print("-------------------- RAW RESPONSE END ----------------------")
            return # Stop processing this sensor

        # --- Extract and Check StringifiedData ---
        temp = data.get("Channels", [])
        if not temp:
            print(f"Warning: No 'Channels' key found in API response. Response: {str(data)[:500]}...")
            return # Nothing to process

        data_string = temp[0].get("StringifiedData") if len(temp) > 0 else None

        no_data_strings = ["no data period", "no data"]
        if data_string is None or data_string in no_data_strings:
            print(f"API reported 'no data' value ('{data_string}').")
            return # Nothing to process

        # --- Parse Nested JSON (StringifiedData) ---
        try:
            datax = json.loads(data_string)
        except json.JSONDecodeError as inner_json_e:
             print(f"ERROR: Failed to decode nested 'StringifiedData' JSON. Error: {inner_json_e}")
             print(f"StringifiedData content (start): {data_string[:500]}...")
             return # Stop processing this sensor

        # --- Process and Filter Time Series Data ---
        time_series_to_append = []
        processed_count = 0
        skipped_old_count = 0
        skipped_error_count = 0

        for item in datax:
            processed_count += 1
            if not isinstance(item, list) or len(item) < 2:
                skipped_error_count += 1
                continue # Skip malformed

            try:
                timestamp_str = str(item[0])
                value = item[1]
                datetime_obj = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                # *** Filter: Only add if strictly newer than last saved timestamp ***
                if last_timestamp is None or datetime_obj > last_timestamp:
                    time_series_to_append.append((datetime_obj, value))
                else:
                    skipped_old_count += 1

            except (ValueError, TypeError):
                skipped_error_count += 1
                continue # Skip records with bad timestamps

        print(f"Processed {processed_count} items from API. Skipped {skipped_old_count} old/overlapping. Skipped {skipped_error_count} with errors.")

        # --- Append to CSV if New Data Exists ---
        if not time_series_to_append:
            print("No *new* data points found to append.")
            return # Nothing to write

        new_df = pd.DataFrame(time_series_to_append, columns=["datetime", "value"])
        new_df['datetime'] = pd.to_datetime(new_df['datetime'])

        # Determine if header should be written (only if file didn't exist before / was empty)
        write_header = (last_timestamp is None)

        try:
            new_df.to_csv(
                csv_filename,
                mode='a', # Append mode
                header=write_header, # Write header only if new file
                index=False,
                date_format=DATE_FORMAT_CSV
            )
            print(f"Successfully appended {len(new_df)} new records to {csv_filename}")

        except Exception as csv_e:
             print(f"ERROR writing to CSV file {csv_filename}: {csv_e}")
             # Failure during write is critical

    # --- Handle Web and Other Errors ---
    except HTTPError as http_e:
        print(f"ERROR: HTTP Error: {http_e.code} {http_e.reason}")
        # (Error body printing omitted for brevity, add back if needed)
    except URLError as url_e:
        print(f"ERROR: URL Error: {url_e.reason}")
    except TimeoutError:
        print(f"ERROR: API request timed out ({API_TIMEOUT_SECONDS}s).")
    except Exception as e: # General catch-all
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

# --- Main Loop ---
print("Reading sensor list from SensorOversigt.xlsx...")
try:
    df_sensors = pd.read_excel('SensorOversigt.xlsx')
    print(f"Found {len(df_sensors)} sensors.")
except FileNotFoundError:
    print("ERROR: SensorOversigt.xlsx not found. Cannot proceed.")
    exit(1)
except Exception as e:
    print(f"ERROR reading SensorOversigt.xlsx: {e}")
    exit(1)

print("\nStarting incremental data fetching cycle...")
for _, row in df_sensors.iterrows():
    # Use .get for safer access
    sensor_name = row.get('Name', 'Unnamed Sensor')
    channel_no_raw = row.get('Channel')
    coordinates = row.get('wkt_geom')

    # Validate Channel
    try:
        channel_no = int(channel_no_raw)
    except (ValueError, TypeError):
        print(f"\nWarning: Invalid or missing 'Channel' ({channel_no_raw}) for sensor '{sensor_name}'. Skipping.")
        continue

    # Validate Coordinates
    if pd.isna(coordinates):
         print(f"\nWarning: Missing 'wkt_geom' for sensor '{sensor_name}'. Skipping.")
         continue

    # Call the function to fetch and append data for this sensor
    append_new_gauge_data(sensor_name, channel_no, str(coordinates))

print("\nIncremental data fetching cycle finished.")



# def get_data(sensor_name, channel_no, coord_string, save_csv=True, plot_data=False, start_date=None, end_date=None):
#     # start_date = datetime(2024, 6, 21, 0, 0, 0)
#     # end_date = datetime(2025, 1, 1, 0, 0, 0)
#     service_url = "https://nkaflob.watermanager.dk/"
#     request_url = (
#         f"{service_url}/Services/DataService.ashx?type=graph&channel={channel_no}"
#         f"&DateFrom={start_date.strftime('%Y-%m-%d%%20%H:%M:%S')}"
#         f"&DateTo={end_date.strftime('%Y-%m-%d%%20%H:%M:%S')}&Reduction=day"
#     )

#     print(f"Fetching data for channel {channel_no}...")

#     try:
#         f = urlopen(request_url)
#         data = json.loads(f.read())

#         temp = data.get("Channels", [])
#         if not temp:
#             print(f"No channel data found for channel {channel_no}.")
#             return

#         hdef = temp[0].get("HistDef", {}).get("Description", "No description")
#         data_string = temp[0].get("StringifiedData", "no data")

#         if data_string != "no data":
#             datax = json.loads(data_string)

#             # Prepare time series data
#             time_series = []
#             for item in datax:
#                 timestamp = item[0]
#                 value = item[1]
#                 datetime_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
#                 time_series.append((datetime_obj, value))

#             # Save to CSV
#             if save_csv:
#                 df = pd.DataFrame(time_series, columns=["datetime", "value"])
#                 folder_name = "../gauge_data"
#                 os.makedirs(folder_name, exist_ok=True)

#                 # Extract and transform coordinates
#                 lon, lat = parse_coordinates(coord_string)
#                 x, y = transform_coordinates(lon, lat)

#                 csv_filename = os.path.join(folder_name, f"gaugedata_{int(x)}_{int(y)}_.csv")
#                 df.to_csv(csv_filename, index=False)
#                 # print(df)
#                 print(f"Data for channel {channel_no} saved to {csv_filename}")

#             # Plot the data (Optional)
#             if plot_data:
#                 dates = [item[0] for item in time_series]
#                 values = [item[1] for item in time_series]

#                 locator = mdates.AutoDateLocator()
#                 formatter = mdates.ConciseDateFormatter(locator)

#                 plt.figure(figsize=(10, 4))
#                 ax = plt.gca()
#                 ax.xaxis.set_major_locator(locator)
#                 ax.xaxis.set_major_formatter(formatter)
#                 plt.title(f"{sensor_name} ({hdef}) - Channel {channel_no}")
#                 plt.ylabel("Accumulated Rain [mm]")
#                 ax.plot(dates, values, label=f"Channel {channel_no}")
#                 plt.legend()
#                 plt.show()

#         else:
#             print(f"No data available for channel {channel_no}")

#     except Exception as e:
#         print(f"Error fetching data for channel {channel_no}: {e}")


# # Example DataFrame with Name, Channel, and Coordinates in 'Point (lon lat)' format
# # df = pd.DataFrame({
# #     'Name': ['Sensor A', 'Sensor B'],
# #     'Channel': [11364, 99999],
# #     'Coordinates': ['Point (11.95974 55.20117)', 'Point (12.34567 56.78901)']
# # })

# GAUGE_FOLDER = "../gauge_data"
# DEFAULT_START_DATE = datetime(2024, 6, 21, 0, 0, 0) # Default if file is new/empty
# DATE_FORMAT_API = "%Y-%m-%d %H:%M:%S"
# DATE_FORMAT_CSV = "%Y-%m-%d %H:%M:%S"
# API_TIMEOUT_SECONDS = 60

# for _, row in df.iterrows():
#     sensor_name = row['Name']
#     channel_no = row['Channel']
#     coordinates = row['wkt_geom']


#     start_date = datetime(2025, 4, 21, 0, 0, 0)
#     end_date = datetime(2025, 4, 24, 0, 0, 0)

#     get_data(sensor_name, channel_no, coordinates, save_csv=True, plot_data=False, start_date=start_date, end_date=end_date)
