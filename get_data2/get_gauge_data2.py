# Read the file SensorOversigt.xlsx using pandas

import os
import ssl
from urllib.request import urlopen
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyproj import Transformer
import re
import numpy as np
import logging
import tqdm as tqdm

print('Start of the gauge script')
# #logger = logging.getLogger(__name__)

# Read the file SensorOversigt.xlsx
df_sensors = pd.read_excel('SensorOversigt.xlsx')

# print('Read the SensorOversigt.xlsx file')
# #logger.info('Read the SensorOversigt.xlsx file')




ssl._create_default_https_context = ssl._create_unverified_context



def parse_wkt(wkt):
    match = re.match(r"Point \(([-\d\.]+) ([-\d\.]+)\)", wkt)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return np.nan, np.nan
    
df_sensors['longitude'], df_sensors['latitude'] = zip(*df_sensors['wkt_geom'].map(parse_wkt))
# print('Extracted coordinates from the file')
# #logger.info('Extracted coordinates from the file')

transformer = Transformer.from_crs("epsg:25832", "epsg:4326", always_xy=True)

print('Transformer created')


# read all the files in the folder gauge_data
gauge_folder = "../gauge_data"
gauge_files = os.listdir(gauge_folder)
# print('Read the gauge_data folder')

# print('gauge_files:', gauge_files)

def parse_coordinates(coord_string):
    """Extracts lon, lat from 'Point (lon lat)' string."""
    coord_string = coord_string.replace("Point (", "").replace(")", "")
    lon, lat = map(float, coord_string.split())
    return lon, lat
def transform_coordinates(lon, lat):
    """Transforms from WGS84 (EPSG:4326) to ETRS89 / UTM zone 32N (EPSG:25832)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y
def get_channel(df_sensors, file, lon, lat):

        # print('Read the gauge_data file:', file)

        # Match
        tolerance = 1e-4  # very small tolerance
        match = df_sensors[
            (np.isclose(df_sensors['longitude'], lon, atol=tolerance)) &
            (np.isclose(df_sensors['latitude'], lat, atol=tolerance))
        ]

        if match.empty:
            print(f"No match found for file {file} with lon {lon}, lat {lat}")
            return
        elif len(match) > 1:
            print(f"Multiple matches found for file {file} with lon {lon}, lat {lat}")
            return

        channel_no = match.iloc[0]['Channel']
        
        return channel_no


for file in gauge_files:
    print(file)
    if file.endswith('.csv'):
        filename_noext = file.replace('.csv', '')
        parts = filename_noext.split('_')
        x_filename = float(parts[1])
        y_filename = float(parts[2])

        # print('Original UTM coords:', x, y)

        lon, lat = transformer.transform(x_filename, y_filename)
        # print('Converted lon/lat:', lon, lat)

        channel_no = get_channel(df_sensors, file, lon, lat)
        # print(f"Matched channel {channel_no} for file {file}")

        # Read the CSV file
        gauge_data = pd.read_csv(os.path.join(gauge_folder, file))

        # Convert time column to datetime
        gauge_data['datetime'] = pd.to_datetime(gauge_data['datetime'], errors='coerce')
        gauge_data.set_index('datetime', inplace=True)
        gauge_data.sort_index(inplace=True)
    last_datetime = gauge_data.index.max()
    now = pd.Timestamp.now()

    

    start_date = last_datetime + pd.Timedelta(days=1)
    end_date = now

    if now.tzinfo is None and start_date.tzinfo is not None:
        print(f"INFO: 'now' ({now}) is tz-naive. Localizing to match 'start_date' timezone ({start_date.tzinfo}).")
        now = now.tz_localize(start_date.tzinfo)
    # --- Add these lines for debugging, right before line 117 ---
    print(f"DEBUG: type(start_date) = {type(start_date)}, start_date = {start_date}")
    if hasattr(start_date, 'tzinfo'):
        print(f"DEBUG: start_date.tzinfo = {start_date.tzinfo}")

    print(f"DEBUG: type(now) = {type(now)}, now = {now}")
    if hasattr(now, 'tzinfo'):
        print(f"DEBUG: now.tzinfo = {now.tzinfo}")
    # --- End of debug lines ---
    if start_date > now:
        # print(f"Start date {start_date} is in the future for channel {channel_no}. Skipping fetch.")
        #logger.warning(f"Start date {start_date} is in the future for channel {channel_no}. Skipping fetch.")
        continue
    # Smart skip: only fetch if at least 1 day difference
    if (now - start_date) < pd.Timedelta(days=1):
        # print(f"Less than 1 day of missing data for channel {channel_no}. Skipping fetch.")
        #logger.warning(f"Less than 1 day of missing data for channel {channel_no}. Skipping fetch.")
        continue

    if start_date == now:
        # print(f"Start date {start_date} is equal to now for channel {channel_no}. Skipping fetch.")
        #logger.warning(f"Start date {start_date} is equal to now for channel {channel_no}. Skipping fetch.")
        continue

    # print('start_date:', start_date)
    # print('end_date:', end_date)



    service_url = "https://nkaflob.watermanager.dk/"
    request_url = (
        f"{service_url}/Services/DataService.ashx?type=graph&channel={channel_no}"
        f"&DateFrom={start_date.strftime('%Y-%m-%d%%20%H:%M:%S')}"
        f"&DateTo={end_date.strftime('%Y-%m-%d%%20%H:%M:%S')}&Reduction=day"
    )

    print(f"Fetching data for channel {channel_no}...")

    save_csv = True
    plot_data = False
    coord_string = f"Point ({lon} {lat})"

    try:
        print(f"Request URL: {request_url}")
        f = urlopen(request_url)
        response_text = f.read().decode('utf-8').strip()

        if not response_text:
            # print(f"No response from server for channel {channel_no}. Skipping...")
            #logger.info(f"No response from server for channel {channel_no}. Skipping...")
            continue  # Skip to next gauge

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response for channel {channel_no}: {e}")
            continue

        # print(data)

        temp = data.get("Channels", [])
        if not temp:
            print(f"No channel data found for channel {channel_no}.")
            continue

        hdef = temp[0].get("HistDef", {}).get("Description", "No description")
        data_string = temp[0].get("StringifiedData", "no data")
        

        if data_string != "no data":
            datax = json.loads(data_string)
            

            # Prepare time series data
            print('Preparing time series data...')
            time_series = []
            for item in datax:
                timestamp = item[0]
                value = item[1]
                datetime_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                time_series.append((datetime_obj, value))

            print('Time series data prepared.')

            # # change datetime to UTC
            # print('before changing to UTC:', time_series)
            # time_series = [(dt.replace(tzinfo=None), value) for dt, value in time_series]
            # print('after changing to UTC:', time_series)
            # print('Time series data:', time_series)

            # Save to CSV
            if save_csv:
                print('Saving to CSV...')
                df = pd.DataFrame(time_series, columns=["datetime", "value"])
                folder_name = "../gauge_data"
                os.makedirs(folder_name, exist_ok=True)
                print('Folder created:', folder_name)

                # Extract and transform coordinates
                lon, lat = parse_coordinates(coord_string)
                x, y = transform_coordinates(lon, lat)

                csv_filename = os.path.join(folder_name, f"gaugedata_{int(x_filename)}_{int(y_filename)}_.csv")
                # df.to_csv(csv_filename, index=False)
                # print(f"Data for channel {channel_no} saved to {csv_filename}")
                # print('CSV filename:', csv_filename)
                # print('gauge_files:', gauge_files)

                if os.path.exists(csv_filename):
                    print(f"File {csv_filename} already exists. Appending new data...")
                    existing_df = pd.read_csv(csv_filename, parse_dates=['datetime'])
                    combined_df = pd.concat([existing_df, df])
                        # Drop duplicates based on datetime
                    combined_df = combined_df.drop_duplicates(subset='datetime')

                    # Sort by datetime
                    combined_df = combined_df.sort_values('datetime')

                    # Save back
                    combined_df.to_csv(csv_filename, index=False)
                    # print(f"Appended new data and updated {csv_filename}")
                    # print('Path for the new file:', csv_filename)
                    # print('first datetime:', combined_df['datetime'].min())
                    # print('last datetime:', combined_df['datetime'].max())

                    data = pd.read_csv(csv_filename, parse_dates=['datetime'])
                    # print('Data from the file:', data)


            # Plot the data (Optional)
            if plot_data:
                dates = [item[0] for item in time_series]
                values = [item[1] for item in time_series]

                locator = mdates.AutoDateLocator()
                formatter = mdates.ConciseDateFormatter(locator)

                plt.figure(figsize=(10, 4))
                ax = plt.gca()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.title(f"({hdef}) - Channel {channel_no}")
                plt.ylabel("Accumulated Rain [mm]")
                ax.plot(dates, values, label=f"Channel {channel_no}")
                plt.legend()
                plt.show()

        else:
            print(f"No data available for channel {channel_no}")

    except Exception as e:
        print(f"Error fetching data for channel {channel_no}: {e}")

    

    
    


