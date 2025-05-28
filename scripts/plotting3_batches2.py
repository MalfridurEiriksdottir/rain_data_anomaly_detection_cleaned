# plotting.py
import json
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from shapely.geometry import Point
from tqdm import tqdm
from config import CFG
from network import get_nearest_neighbors
from matplotlib import cm as colormap
import numpy as np
import warnings
import datetime

def create_plots_with_error_markers(
    all_data: dict,
    coordinate_locations_utm: dict,
    coord_to_channel_map: dict,
    events_df: pd.DataFrame = None, 
    all_data_iter0: dict = None    
    ):
    """
    Create interactive Plotly time series plots for each sensor and save as HTML.
    Plots the 4 key radar series + gauge and highlights faulty batch intervals.
    """
    warnings.filterwarnings("ignore")
    output_dir = CFG.ADJUSTMENT_PLOTS_DIR if hasattr(CFG, 'ADJUSTMENT_PLOTS_DIR') else CFG.PLOTS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(CFG, 'ADJUSTMENT_PLOTS_DIR'): CFG.PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Creating adjustment comparison plots in: {output_dir}")

    for coord in tqdm(all_data.keys()):
        coord_str = str(coord)
        if coord_str not in all_data: continue

        df_final = all_data[coord_str]
        if not isinstance(df_final, pd.DataFrame) or df_final.empty:
            #  logger.warning(f"Skipping plot for {coord_str}: Invalid or empty DataFrame.")
             continue

        fig = go.Figure()

        channel_name = coord_to_channel_map.get(coord_str, "Unknown Channel")

        # --- Plot Gauge ---
        if 'Gauge_Data_mm_per_min' in df_final:
            gauge_y = df_final['Gauge_Data_mm_per_min']
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Gauge_Data_mm_per_min'], mode='lines',
                                     name='Gauge', line=dict(color='blue', width=1)))

        # --- Plot Raw Radar ---
        if 'Radar_Data_mm_per_min' in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Radar_Data_mm_per_min'], mode='lines',
                                     name='Raw Radar', line=dict(color='orange', width=1)))

        # --- Plot Raw Radar * f_reg ---
        if 'Radar_Freg_Adjusted' in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Radar_Freg_Adjusted'], mode='lines',
                                     name='Raw Radar * f_reg',
                                     line=dict(color='green', dash='dash', width=1.5)))
        else: print(f"'Radar_Freg_Adjusted' missing for {coord_str}")

        # --- Plot "24h Adjusted Radar" (Step 2: Batch Alpha Adj) ---
        step2_col_name = 'Batch_Alpha_Adjusted_Radar'
        if step2_col_name in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final[step2_col_name], mode='lines',
                                     name='Batch Alpha Adj. Radar', # Accurate name
                                     line=dict(color='purple', dash='dash', width=1.5)))
        else: print(f"'{step2_col_name}' missing for {coord_str}")

        # --- Plot Final Adjusted Rainfall (Step 5) ---
        if 'Final_Adjusted_Rainfall' in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Final_Adjusted_Rainfall'], mode='lines',
                                     name='Final Adjusted',
                                     line=dict(color='red', width=2)))
        else: print(f"'Final_Adjusted_Rainfall' missing for {coord_str}")

        # --- ADD THIS BLOCK: Plot Timestep Anomaly Flags ---
        if 'Flagged' in df_final.columns and gauge_y is not None: # Check if Flagged exists AND gauge data exists
            flag_series = df_final['Flagged'].fillna(False).astype(bool)
            if flag_series.any(): # Check if there are any True flags to plot
                flagged_timestamps = df_final.index[flag_series]
                # Get the corresponding gauge values at the flagged times
                gauge_values_at_flags = gauge_y.loc[flagged_timestamps]

                # Add markers on the gauge line where Flagged is True
                fig.add_trace(go.Scatter(
                    x=flagged_timestamps,
                    y=gauge_values_at_flags,
                    mode='markers',
                    name='Timestep Anomaly',
                    marker=dict(
                        color='red',
                        size=5,
                        symbol='x' # 'cross' or 'x' are good symbols
                    ),
                    showlegend=True # Add to legend
                ))
# --- START: Add Shading for Batch_Flag ---
        if 'Batch_Flag' in df_final.columns:
            flag_series = df_final['Batch_Flag'].fillna(False).astype(bool)
            if flag_series.any():
                # Find changes in the boolean series to identify blocks
                change_indices = flag_series.ne(flag_series.shift())
                block_ids = change_indices.cumsum()
                true_blocks = block_ids[flag_series] # Filter for blocks where flag is True

                if not true_blocks.empty:
                    # try:
                    #     # Group the index by the true block IDs and find min/max timestamp for each block
                    #     intervals = df_final.index[flag_series].groupby(true_blocks).agg(['min', 'max'])

                    #     for _, row in intervals.iterrows():
                    #          start_time = row['min']
                    #          end_time = row['max']
                    #          # Add a vertical rectangle for the flagged batch interval
                    #          if pd.notna(start_time) and pd.notna(end_time) and start_time <= end_time:
                    #              fig.add_vrect(
                    #                  x0=start_time, x1=end_time,
                    #                  fillcolor="rgba(211, 211, 211, 0.3)", # Light gray with transparency
                    #                  layer="below", # Ensure shading is behind data lines
                    #                  line_width=0, # No border line for the rectangle
                    #                  # Optional: Add annotation to the shaded area
                    #                  # annotation_text="Faulty Batch",
                    #                  # annotation_position="top left"
                    #              )
                    # except Exception as e_vrect:
                    #      print(f"Error adding batch flag shading for {coord_str}: {e_vrect}")
                    #      # logger.error(f"Error adding batch flag shading for {coord_str}: {e_vrect}") # Uncomment if logger is configured
                        try:
                            # --- Alternative method to find start/end of each true block ---
                            flagged_indices = df_final.index[flag_series] # Get timestamps where flag is True

                            # Iterate through the unique block IDs where the flag is true
                            for block_id in true_blocks.unique():
                                # Get all timestamps belonging to this specific block
                                block_timestamps = flagged_indices[true_blocks == block_id]

                                if not block_timestamps.empty:
                                    # Find the minimum and maximum timestamp within this block
                                    start_time = block_timestamps.min()
                                    end_time = block_timestamps.max()

                                    # Add a vertical rectangle for the flagged batch interval
                                    if pd.notna(start_time) and pd.notna(end_time) and start_time <= end_time:
                                        fig.add_vrect(
                                            x0=start_time, x1=end_time,
                                            fillcolor="rgba(211, 211, 211, 0.3)", # Light gray with transparency
                                            layer="below", # Ensure shading is behind data lines
                                            line_width=0, # No border line for the rectangle
                                    )
                        except Exception as e_vrect:
                         # It's helpful to see the specific error type
                            print(f"Error adding batch flag shading for {coord_str}: {type(e_vrect).__name__} - {e_vrect}")
                         # logger.error(f"Error adding batch flag shading for {coord_str}: {e_vrect}", exc_info=True) # Log traceback
        # --- END: Add Shading for Batch_Flag ---


        # --- Layout ---
        layout_update = dict(
            title=f"Radar Adjustment Comparison for coord {coord_str} and channel {channel_name}",
            xaxis_title="Time",
            yaxis_title="Rainfall Rate (mm/min)",
            template=CFG.PLOT_TEMPLATE,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_layout(**layout_update)

        # --- Save Plot ---
        safe_coord = coord_str#.replace('(','').replace(')','').replace(',','_').replace(' ','')
        output_path = output_dir / f"{safe_coord}.html"
        try:
            fig.write_html(str(output_path), full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            logger.error(f"Error writing plot {output_path}: {e}")

    print(f"Adjustment comparison plots saved to {output_dir}")


# =============================================================================
# Flagging Metrics Plot
# =============================================================================
def plot_flagging_metrics(df: pd.DataFrame, coord_name: str = "", events_df: pd.DataFrame = None) -> go.Figure:
    """
    Create a multi-panel Plotly figure showing key metrics related to the
    batch adjustment and flagging process.
    Includes batch boundary lines and faulty batch shading.
    """
    logger.debug(f"Plotting flagging metrics for {coord_name}")
    subplot_titles = (
        "1. Data & Final Adjustments",       # Raw + Final Adj + Batch Flags
        "2. Rolling Abs Error (Flagging Input)", # Error used to determine Flagged
        "3. Rolling Prop Flagged (Flagging Input)",# Propensity to be flagged
        "4. Adj. Diff From Net (Flagging Input)", # Direct input to flag_anomalies
        "5. Adj. Ratio From Net (Flagging Input)" # Direct input to flag_anomalies
        # Removed Neighbor Count and Gauge=0 Condition
    )
    # Create 5 rows
    fig = sp.make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04 
    )

    # --- Add vertical lines for 24h batch boundaries to all subplots ---
    if not df.index.empty:
        logger.debug(f"Adding batch boundaries for {coord_name}")
        try:
            idx = df.index
            if not isinstance(idx, pd.DatetimeIndex): idx = pd.to_datetime(idx)
            if idx.tz is None: idx = idx.tz_localize('UTC')
            elif idx.tz != datetime.timezone.utc: idx = idx.tz_convert('UTC')
            sensor_start = idx.min(); sensor_end = idx.max()
            batch_boundaries = pd.date_range(start=sensor_start.floor(CFG.BATCH_DURATION), end=sensor_end, freq=CFG.BATCH_DURATION, tz='UTC')
            for boundary in batch_boundaries:
                fig.add_vline(x=boundary, line=dict(color='rgba(0,0,0,0.3)', dash='dash', width=1), row='all', col=1)
        except Exception as e_vline: logger.error(f"Error adding vlines for {coord_name}: {e_vline}")


    
    # --- Row 1: Data, Flags & Adjustments ---
    plot_row = 1
    if 'Radar_Data_mm_per_min' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Data_mm_per_min'], mode='lines', name='Raw Radar', line=dict(color='grey', dash='dot', width=1)), row=plot_row, col=1)
    if 'Gauge_Data_mm_per_min' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Gauge_Data_mm_per_min'], mode='lines', name='Gauge', line=dict(color='blue', width=1)), row=plot_row, col=1)
    # Plot the adjusted radar used for flagging (Step 2 result)
    step2_col_name = 'Batch_Alpha_Adjusted_Radar'
    if step2_col_name in df:
        fig.add_trace(go.Scatter(x=df.index, y=df[step2_col_name], mode='lines', name='Batch Alpha Adj.', line=dict(color='purple', dash='dashdot')), row=plot_row, col=1)
    # Plot the final adjusted radar (Step 5 result)
    if 'Final_Adjusted_Rainfall' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Final_Adjusted_Rainfall'], mode='lines', name='Final Adjusted', line=dict(color='red')), row=plot_row, col=1)


    # --- Row 2: Rolling Absolute Error (Used for Flagging) ---
    plot_row = 2
    if 'Rolling_Abs_Error' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Abs_Error'], mode='lines', name='Rolling Abs Error', line=dict(color='magenta')), row=plot_row, col=1)
        fig.add_hline(y=CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1)
    else: logger.warning(f"'Rolling_Abs_Error' missing for {coord_name}")


    # --- Row 3: Rolling Proportion Flagged (Used for Flagging) ---
    plot_row = 3
    if 'Rolling_Prop_Flagged' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Rolling_Prop_Flagged'], mode='lines', name='Rolling Prop Flagged', line=dict(color='darkgreen')), row=plot_row, col=1)

        fig.add_hline(y=CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1, annotation_text="Flag Condition Thresh")
    else: logger.warning(f"'Rolling_Prop_Flagged' missing for {coord_name}")


    # --- Row 4: Adjusted Diff from Net (Input to Flagging) ---
    plot_row = 4
    if 'Adjusted_Diff_from_network' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Diff_from_network'], mode='lines', name='Adj. Diff (Flag Input)', line=dict(color='#ff7f0e')), row=plot_row, col=1) # Orange color
    fig.add_hline(y=CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1)
    fig.add_hline(y=-CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1)


    # --- Row 5: Adjusted Ratio from Net (Input to Flagging) ---
    plot_row = 5
    if 'Adjusted_Ratio_From_Network' in df:
        fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Ratio_From_Network'], mode='lines', name='Adj. Ratio (Flag Input)', line=dict(color='green')), row=plot_row, col=1)
    fig.add_hline(y=1 + CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1)
    fig.add_hline(y=1 - CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=plot_row, col=1)
    fig.add_hline(y=1, line=dict(color='grey', dash='dot', width=1), row=plot_row, col=1)
    fig.update_yaxes(range=[0, 3.5], row=plot_row, col=1) # Set range for ratio


    # --- Final Layout ---
    fig.update_layout(
        height=1600, # Adjusted height for 5 rows
        title_text=f"Flagging & Adjustment Metrics for {coord_name}",
        template=CFG.PLOT_TEMPLATE,
        showlegend=True,
        hovermode="x unified"
    )


    return fig

# =============================================================================
# Dashboard Creation Functions
# =============================================================================
# def create_flagging_plots_dashboard(all_data: dict, events_df: pd.DataFrame = None,
#                                     output_dir: str = str(CFG.FLAGGING_PLOTS_DIR),
#                                     dashboard_file: str = str(CFG.FLAGGING_DASHBOARD_FILE)):
#     """Create individual HTML plots for flagging metrics and a dashboard."""
#     output_path = CFG.FLAGGING_PLOTS_DIR
#     output_path.mkdir(parents=True, exist_ok=True)
#     print(f"Creating flagging plots in: {output_path}")



#     sensor_list = sorted(all_data.keys())
#     for coord in tqdm(sensor_list, desc="Generating Flagging Plots"):
#         coord_str = str(coord)
#         if coord_str not in all_data: continue

#         df_sensor = all_data[coord_str]
#         sensor_events = None # Pass None if events not relevant

#         if not isinstance(df_sensor, pd.DataFrame) or df_sensor.empty:
#              logger.warning(f"Skipping flagging plot for {coord_str}: Invalid or empty DataFrame.")
#              continue

#         fig = plot_flagging_metrics(df_sensor, coord_name=coord_str, events_df=sensor_events)

#         # filename
#         safe_coord = coord_str.replace('(','').replace(')','').replace(',','_').replace(' ','')
#         plot_file = output_path / f"{safe_coord}_flagging_metrics.html" # Specific filename
#         try:
#             fig.write_html(str(plot_file), full_html=False, include_plotlyjs='cdn')
#         except Exception as e:
#             logger.error(f"Error writing flagging plot {plot_file}: {e}")

#     # --- Create Dashboard HTML ---
#     plot_dir_relative = f"../{CFG.FLAGGING_PLOTS_DIR.name}"
#     iframe_container_height = 1650 # Adjusted height
#     html_header = f"""<!DOCTYPE html><html><head><title>Flagging Plots Dashboard</title>
#     <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
#     <style> /* ... CSS ... */ </style></head><body>
#     <h1>Flagging Plots Dashboard</h1><div id="controls"><label for="sensor-dropdown">Select Sensor:</label>
#     <select id="sensor-dropdown" onchange="updatePlot(this.value)">
#     <option value="">-- Select a Coordinate --</option>"""
#     html_options = ""
#     for coord in sensor_list:
#          safe_coord = str(coord).replace('(','').replace(')','').replace(',','_').replace(' ','')
#          html_options += f'\n<option value="{safe_coord}_flagging_metrics.html">{coord}</option>'

#     html_middle = f"""</select></div><div id="plot-container">
#     <iframe id="plot-frame" src="" frameborder="0"></iframe></div><script>
#     function updatePlot(f) {{ document.getElementById('plot-frame').src = f ? '{plot_dir_relative}/' + f : ''; }}
#     </script>"""
#     html_footer = """</body></html>"""
#     final_html = html_header + html_options + html_middle + html_footer

#     try:
#         with open(dashboard_file, 'w', encoding='utf-8') as f: f.write(final_html)
#         print(f"Flagging dashboard created: {dashboard_file}")
#     except Exception as e: print(f"Error writing flagging dashboard file {dashboard_file}: {e}")#logger.error(f"Error writing flagging dashboard file {dashboard_file}: {e}")


def generate_html_dashboard(all_data: dict, coordinate_locations_utm: dict, gdf_wgs84: gpd.GeoDataFrame,
                            svk_coords_utm: list, output_file: str = str(CFG.DASHBOARD_FILE)):
    """Generate an HTML dashboard with a Leaflet map and embedded Plotly time series."""
    print(f"Generating HTML dashboard: {output_file}...")
    map_data = []
    max_flagged_count = 0
    svk_coords_set = set(svk_coords_utm)
    if not coordinate_locations_utm:
        print("Warning: No coordinate locations for dashboard.")
        return

    gdf_items = [{'name': name, 'geometry': Point(coord)}
                 for name, coord in coordinate_locations_utm.items() if name in all_data]
    if not gdf_items:
        print("Warning: No valid coordinates with data to plot on map.")
        return

    temp_gdf_utm = gpd.GeoDataFrame(gdf_items, crs=CFG.SENSOR_CRS_PROJECTED)
    temp_gdf_wgs84 = temp_gdf_utm.to_crs(epsg=4326)
    map_center_lat = temp_gdf_wgs84.geometry.y.mean()
    map_center_lon = temp_gdf_wgs84.geometry.x.mean()

    for _, row in temp_gdf_wgs84.iterrows():
        fname = row['name']
        if 'Flagged' in all_data[fname].columns:
            flagged_count = int(all_data[fname]['Flagged'].sum())
            max_flagged_count = max(max_flagged_count, flagged_count)
            map_data.append({'coordinates': [row.geometry.y, row.geometry.x],
                             'flagged_count': flagged_count,
                             'coordinate_name': fname,
                             'is_svk': coordinate_locations_utm[fname] in svk_coords_set})
    plot_dir_relative = f"../{CFG.ADJUSTMENT_PLOTS_DIR.name}"
    html_content = f"""
    <!DOCTYPE html><html><head><title>Radar vs Gauge Anomaly Detection</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.4.2/chroma.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #controls {{ margin-bottom: 20px; display: flex; align-items: center; gap: 20px; }}
        #map {{ height: 500px; width: 40%; float: left; margin-right: 2%; border: 1px solid #ccc;}}
        #timeseries-plot-container {{ height: 500px; width: 58%; float: left; border: 1px solid #ccc; }}
        #plot-frame {{ width: 100%; height: 100%; border: none; }}
        .info {{ padding: 6px 8px; font: 14px/16px Arial, Helvetica, sans-serif; background: rgba(255,255,255,0.8); box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px; }}
        .info h4 {{ margin: 0 0 5px; color: #777; }}
        .clearfix::after {{ content: ""; clear: both; display: table; }}
    </style></head>
    <body><h1>Radar vs Gauge Anomaly Detection</h1>
    <div id="controls">
        <label for="coordinate-dropdown">Select Coordinate:</label>
        <select id="coordinate-dropdown" onchange="updatePlot(this.value)">
            <option value="">-- Select a Coordinate --</option>
            {''.join([f'<option value="{coord}">{coord}</option>' for coord in sorted(all_data.keys())])}
        </select>
    </div>
    <div id="map-plot-wrapper" class="clearfix">
        <div id="map"></div>
        <div id="timeseries-plot-container">
            <iframe id="plot-frame" src=""></iframe>
        </div>
    </div>
    <script>
        var mapPoints = {json.dumps(map_data)};
        var plotFrame = document.getElementById('plot-frame');
        var map = L.map('map').setView([{map_center_lat}, {map_center_lon}], {CFG.MAP_DEFAULT_ZOOM});
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);
        var info = L.control();
        info.onAdd = function (map) {{ this._div = L.DomUtil.create('div', 'info'); this.update(); return this._div; }};
        info.update = function (props) {{
            this._div.innerHTML = '<h4>Flagged Points</h4>' + (props ?
                '<b>Coordinate: ' + props.coordinate_name + '</b><br />Flagged Count: ' + props.flagged_count :
                'Hover over a coordinate');
        }};
        info.addTo(map);
        var maxFlagged = Math.max(1, {max_flagged_count});
        var colorScale = chroma.scale(['blue', 'yellow', 'red']).domain([0, maxFlagged * 0.5, maxFlagged]).mode('lab');
        mapPoints.forEach(pointData => {{
            var marker = L.circleMarker(pointData.coordinates, {{
                radius: 6,
                fillColor: colorScale(pointData.flagged_count).hex(),
                color: pointData.is_svk ? 'black' : 'white',
                weight: pointData.is_svk ? 2 : 1,
                opacity: 1,
                fillOpacity: 0.8
            }});
            marker.bindTooltip(`Coord: ${{pointData.coordinate_name}}<br>Flags: ${{pointData.flagged_count}}`);
            marker.on('mouseover', function (e) {{ info.update(pointData); }});
            marker.on('mouseout', function (e) {{ info.update(); }});
            marker.on('click', function (e) {{
                updatePlot(pointData.coordinate_name);
                document.getElementById('coordinate-dropdown').value = pointData.coordinate_name;
            }});
            marker.addTo(map);
        }});
        function updatePlot(selectedCoord) {{
            if (selectedCoord) {{
                plotFrame.src = '{plot_dir_relative}/' + selectedCoord + '.html';
            }} else {{
                plotFrame.src = '';
            }}
        }}
    </script></body></html>
    """
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard generated: {output_file}")
    except Exception as e:
        print(f"Error writing dashboard file {output_file}: {e}")





def create_flagging_plots_dashboard(all_data: dict, events_df: pd.DataFrame = None,
                                    output_dir: str = str(CFG.FLAGGING_PLOTS_DIR),
                                    dashboard_file: str = str(CFG.FLAGGING_DASHBOARD_FILE)):
    """Create individual HTML plots for flagging metrics and a dashboard."""
    output_path = CFG.FLAGGING_PLOTS_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating flagging plots in: {output_path}")
    events_by_coord = {}
    if events_df is not None and not events_df.empty and 'sensor_coord' in events_df.columns:
        for coord, group in events_df.groupby('sensor_coord'):
            events_by_coord[coord] = group

    sensor_list = sorted(all_data.keys())
    for coord in tqdm(sensor_list):
        df_sensor = all_data[coord]
        sensor_events = events_by_coord.get(coord, None)
        fig = plot_flagging_metrics(df_sensor, coord_name=coord, events_df=sensor_events)
        plot_file = output_path / f"{coord}.html"
        try:
            fig.write_html(str(plot_file), full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing flagging plot {plot_file}: {e}")

    plot_dir_relative = f"../{CFG.FLAGGING_PLOTS_DIR.name}"
    iframe_container_height = 1450
    html_header = f"""<!DOCTYPE html><html><head><title>Flagging Plots Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #controls {{ margin-bottom: 20px; }}
        #plot-container {{ width: 100%; height: {iframe_container_height}px; border: 1px solid #ccc; overflow: auto;}}
        #plot-frame {{ width: 100%; height: 100%; border: none; }}
    </style></head><body>
    <h1>Flagging Plots Dashboard</h1><div id="controls"><label for="sensor-dropdown">Select Sensor:</label>
    <select id="sensor-dropdown" onchange="updatePlot(this.value)">
    <option value="">-- Select a Coordinate --</option>"""
    html_options = "\n".join([f'<option value="{coord}.html">{coord}</option>' for coord in sensor_list])
    html_middle = f"""</select></div><div id="plot-container">
    <iframe id="plot-frame" src="" frameborder="0"></iframe></div><script>
    function updatePlot(f) {{ document.getElementById('plot-frame').src = f ? '{plot_dir_relative}/' + f : ''; }}
    </script>"""
    html_footer = """</body></html>"""
    final_html = html_header + html_options + html_middle + html_footer

    try:
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"Flagging dashboard created: {dashboard_file}")
    except Exception as e:
        print(f"Error writing flagging dashboard file {dashboard_file}: {e}")

def debug_alpha_for_coord(coord: str, all_data: dict, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> go.Figure:
    """Plot gauge/radar and alpha values for the main sensor and its neighbors."""
    if coord not in all_data:
        raise ValueError(f"No data for {coord}")
    df_main = all_data[coord]
    neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=n_neighbors)
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=(f"1. Data {coord}", "2. Neighbors", "3. Alpha"),
                           vertical_spacing=0.05)
    if 'Gauge_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Gauge_Data_mm_per_min'], mode='lines',
                                 name=f"G ({coord})", line=dict(color='blue')), row=1, col=1)
    if 'Radar_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Radar_Data_mm_per_min'], mode='lines',
                                 name=f"R ({coord})", line=dict(color='orange')), row=1, col=1)
    colors = colormap.viridis(np.linspace(0, 1, len(neighbors)))
    for i, neighbor in enumerate(neighbors):
        if neighbor in all_data:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            if 'Gauge_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Gauge_Data_mm_per_min'], mode='lines',
                                         name=f"G ({neighbor[:6]}..)", line=dict(color=color_hex, width=1)), row=2, col=1)
            if 'Radar_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Radar_Data_mm_per_min'], mode='lines',
                                         name=f"R ({neighbor[:6]}..)", line=dict(color=color_hex, dash='dot', width=1)), row=2, col=1)
    if 'Alpha' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Alpha'], mode='lines',
                                 name=f"Alpha ({coord})", line=dict(color='red', width=2)), row=3, col=1)
    for i, neighbor in enumerate(neighbors):
        if neighbor in all_data and 'Alpha' in all_data[neighbor]:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Alpha'], mode='lines',
                                     name=f"Alpha ({neighbor[:6]}..)", line=dict(color=color_hex, width=1)), row=3, col=1)
    fig.update_layout(height=1000, title_text=f"Alpha Debug: {coord}", template=CFG.PLOT_TEMPLATE, hovermode="x unified")
    return fig

def debug_alpha_and_neighbors_plot(coord: str, all_data: dict, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> go.Figure:
    """Plots gauge/radar for the main sensor and each neighbor in separate subplots, plus a final subplot comparing Alpha values."""
    print('Creating debug plot for:', coord)
    if coord not in all_data:
        raise ValueError(f"No data for {coord}")
    df_main = all_data[coord]
    neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=n_neighbors)
    if not neighbors:
        raise ValueError(f"No neighbors found for {coord}")
    num_rows = 2 + len(neighbors)
    subplot_titles = [f"1. Main Sensor: {coord}"]
    subplot_titles.extend([f"{i+2}. Neighbor: {n}" for i, n in enumerate(neighbors)])
    subplot_titles.append(f"{num_rows}. Alpha Comparison")
    fig = sp.make_subplots(rows=num_rows, cols=1, shared_xaxes=True, subplot_titles=subplot_titles, vertical_spacing=0.02)
    current_row = 1
    if 'Gauge_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Gauge_Data_mm_per_min'], mode='lines',
                                 name=f"Gauge ({coord})", legendgroup="main", line=dict(color='blue')), row=current_row, col=1)
    if 'Radar_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Radar_Data_mm_per_min'], mode='lines',
                                 name=f"Radar ({coord})", legendgroup="main", line=dict(color='orange')), row=current_row, col=1)
    if 'Alpha' in df_main and 'Radar_Data_mm_per_min' in df_main:
        #  sensor_alpha = df_main['Alpha'].fillna(method='ffill').fillna(method='bfill')
        sensor_alpha = df_main['Alpha'].ffill().bfill().infer_objects(copy=False)
        if not sensor_alpha.isnull().all():
            adj_radar = df_main['Radar_Data_mm_per_min'] * sensor_alpha
            fig.add_trace(go.Scatter(x=adj_radar.index, y=adj_radar, mode='lines',
                                     name='Adj. Radar (Self)', legendgroup='main', line=dict(color='purple', dash='dot')), row=current_row, col=1)
    for i, neighbor in enumerate(neighbors):
        current_row += 1
        legend_group = f"neighbor_{i}"
        if neighbor in all_data:
            df_n = all_data[neighbor]
            if 'Gauge_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Gauge_Data_mm_per_min'], mode='lines',
                                         name=f"Gauge ({neighbor[:8]}..)", legendgroup=legend_group, line=dict(color='blue', width=1)), row=current_row, col=1)
            if 'Radar_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Radar_Data_mm_per_min'], mode='lines',
                                         name=f"Radar ({neighbor[:8]}..)", legendgroup=legend_group, line=dict(color='orange', width=1)), row=current_row, col=1)
        else:
            fig.add_annotation(text=f"Data missing for {neighbor}", row=current_row, col=1, showarrow=False)
    current_row += 1
    if 'Alpha' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Alpha'], mode='lines',
                                 name=f"Alpha ({coord})", legendgroup="alpha", line=dict(color='red', width=2)), row=current_row, col=1)
    colors = colormap.viridis(np.linspace(0, 1, len(neighbors)))
    for i, neighbor in enumerate(neighbors):
        if neighbor in all_data and 'Alpha' in all_data[neighbor]:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Alpha'], mode='lines',
                                     name=f"Alpha ({neighbor[:8]}..)", legendgroup="alpha", line=dict(color=color_hex, width=1)), row=current_row, col=1)
    fig.add_hline(y=1, line=dict(color='grey', dash='dash', width=1), row=current_row, col=1)
    plot_height = 300 + num_rows * 150
    fig.update_layout(height=plot_height, title_text=f"Debug Comparison: {coord} and Neighbors", template=CFG.PLOT_TEMPLATE, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig
