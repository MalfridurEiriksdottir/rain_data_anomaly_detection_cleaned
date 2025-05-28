# plotting.py
import json
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from shapely.geometry import Point
from tqdm import tqdm
from config import CFG
from network import get_nearest_neighbors # Keep for debug plots if needed
from matplotlib import cm as colormap
import numpy as np
import warnings

# --- generate_html_dashboard remains the same ---
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

    # Use final data for flagged counts
    for coord_str, df_final in all_data.items():
        if coord_str in temp_gdf_wgs84['name'].values:
            row = temp_gdf_wgs84[temp_gdf_wgs84['name'] == coord_str].iloc[0]
            flagged_count = 0
            if 'Flagged' in df_final.columns:
                flagged_count = int(df_final['Flagged'].sum())
            max_flagged_count = max(max_flagged_count, flagged_count)
            is_svk = False
            if coord_str in coordinate_locations_utm:
                 is_svk = coordinate_locations_utm[coord_str] in svk_coords_set

            map_data.append({'coordinates': [row.geometry.y, row.geometry.x],
                             'flagged_count': flagged_count,
                             'coordinate_name': coord_str,
                             'is_svk': is_svk}) # Use coord_str consistently

    plot_dir_relative = f"../{CFG.PLOTS_OUTPUT_DIR.name}"
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
        var maxFlagged = Math.max(1, {max_flagged_count}); // Ensure maxFlagged is at least 1
        var colorScale = chroma.scale(['blue', 'yellow', 'red']).domain([0, maxFlagged * 0.5, maxFlagged]).mode('lab');

        mapPoints.forEach(pointData => {{
            var marker = L.circleMarker(pointData.coordinates, {{
                radius: 6,
                fillColor: colorScale(pointData.flagged_count).hex(),
                color: pointData.is_svk ? 'black' : '#555', // Dark border for SVK, slightly lighter for others
                weight: pointData.is_svk ? 2 : 1.5, // Thicker border for SVK
                opacity: 1,
                fillOpacity: 0.8
            }});
            marker.bindTooltip(`Coord: ${{pointData.coordinate_name}}<br>Flags: ${{pointData.flagged_count}}`);
            marker.on('mouseover', function (e) {{ info.update(pointData); this.bringToFront(); L.setOptions(this, {{fillOpacity: 1, weight: 3 }}); }}); // Highlight on hover
            marker.on('mouseout', function (e) {{ info.update(); L.setOptions(this, {{ fillOpacity: 0.8, weight: pointData.is_svk ? 2 : 1.5 }}); }}); // Reset style
            marker.on('click', function (e) {{
                updatePlot(pointData.coordinate_name);
                document.getElementById('coordinate-dropdown').value = pointData.coordinate_name;
                // Optional: Fly to clicked marker
                // map.flyTo(e.latlng, map.getZoom());
            }});
            marker.addTo(map);
        }});
        function updatePlot(selectedCoord) {{
            if (selectedCoord) {{
                // Ensure the filename in the path matches how plots are saved (coord + '.html')
                plotFrame.src = '{plot_dir_relative}/' + selectedCoord + '.html';
            }} else {{
                plotFrame.src = ''; // Clear iframe if no selection
            }}
        }}
         // Initialize plot with the first coordinate if available
         // var initialCoord = mapPoints.length > 0 ? mapPoints[0].coordinate_name : '';
         // if (initialCoord) {{
         //     document.getElementById('coordinate-dropdown').value = initialCoord;
         //     updatePlot(initialCoord);
         // }}
    </script></body></html>
    """
    try:
        # Ensure the output directory exists
        CFG.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard generated: {output_file}")
    except Exception as e:
        print(f"Error writing dashboard file {output_file}: {e}")

# --- MODIFIED: create_plots_with_error_markers ---
def create_plots_with_error_markers(all_data: dict, coordinate_locations_utm: dict, sensor_channels: dict,
                                    events_df: pd.DataFrame = None):
    """Create interactive Plotly time series plots for each sensor and save as HTML.
       Shows Raw Radar/Gauge, Self-Alpha Adjusted Radar, Network Adjusted Radar, Flags, and Events.
    """
    warnings.filterwarnings("ignore")
    output_dir = CFG.PLOTS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating time series plots in: {output_dir}")

    events_by_coord = {}
    if events_df is not None and not events_df.empty:
        for coord, group in events_df.groupby('sensor_coord'):
            events_by_coord[coord] = group

    for coord in tqdm(all_data.keys()):
        df = all_data[coord].copy() # Work on a copy
        fig = go.Figure()

        # 1. Raw Gauge Data
        if 'Gauge_Data_mm_per_min' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Gauge_Data_mm_per_min'], mode='lines', name='Gauge',
                                     line=dict(color='blue')))

        # 2. Raw Radar Data
        if 'Radar_Data_mm_per_min' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Data_mm_per_min'], mode='lines', name='Radar',
                                     line=dict(color='orange', dash='solid'))) # Solid orange for raw radar

        # 3. Adjusted Radar (Self Alpha)
        if 'Alpha' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
            # Ensure Alpha is filled if needed (though should be done in feature eng.)
            alpha_filled = df['Alpha'].ffill().bfill().infer_objects(copy=False)
            if not alpha_filled.isnull().all():
                # Use CFG.EPSILON for consistency if adding to radar before multiplying
                initial_adjusted_radar = (df['Radar_Data_mm_per_min'] + CFG.EPSILON) * alpha_filled
                fig.add_trace(go.Scatter(x=initial_adjusted_radar.index, y=initial_adjusted_radar, mode='lines',
                                         name='Adj. Radar (Self Alpha)',
                                         line=dict(color='green', dash='dashdot'))) # Green dash-dot

        # 4. Adjusted Radar (Network Median Alpha) - Plotting the final calculated value
        if 'Network_Adjusted_Radar' in df.columns:
             # Plot the pre-calculated Network_Adjusted_Radar
             fig.add_trace(go.Scatter(x=df.index, y=df['Network_Adjusted_Radar'], mode='lines',
                                      name='Adj. Radar (Network)',
                                      line=dict(color='purple', dash='dot'))) # Purple dot

        # 5. Flagged anomalies
        if 'Flagged' in df.columns and df['Flagged'].any():
            # Ensure we plot markers against the Gauge data where flagged
            gauge_data_for_flags = df['Gauge_Data_mm_per_min'] if 'Gauge_Data_mm_per_min' in df.columns else pd.Series(0, index=df.index)
            fig.add_trace(go.Scatter(x=df.index[df['Flagged']], y=gauge_data_for_flags[df['Flagged']],
                                     mode='markers', name='Anomaly Flag',
                                     marker=dict(color='red', size=6, symbol='x')))

        # 6. Event Shading
        sensor_events = events_by_coord.get(coord, pd.DataFrame())
        if not sensor_events.empty:
            for _, event in sensor_events.iterrows():
                event_color = CFG.EVENT_FLAGGED_COLOR if event['is_flagged_event'] else CFG.EVENT_NORMAL_COLOR
                fig.add_vrect(x0=event['event_start'], x1=event['event_end'],
                              fillcolor=event_color, layer="below", line_width=0,
                              annotation_text="Flagged Event" if event['is_flagged_event'] else "Event",
                              annotation_position="top left",
                              annotation=dict(font_size=10, align="left"))


        # Layout
        sensor_channel_info = f" (Channel: {sensor_channels.get(tuple(map(int, coord.strip('()').split(','))), 'N/A')})" if sensor_channels else ""
        layout_update = dict(
            title=f"Time Series for {coord}{sensor_channel_info}",
            xaxis_title="Time",
            yaxis_title="Rainfall Rate (mm/min)",
            template=CFG.PLOT_TEMPLATE,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legend above plot
        )
        fig.update_layout(**layout_update)

        # Save plot
        output_path = output_dir / f"{coord}.html"
        try:
            fig.write_html(str(output_path), full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing plot {output_path}: {e}")

    print(f"Time series plots saved to {output_dir}")


# --- MODIFIED: plot_flagging_metrics ---
def plot_flagging_metrics(df: pd.DataFrame, coord_name: str = "", events_df: pd.DataFrame = None) -> go.Figure:
    """Create a multi-panel Plotly figure showing flagging metrics with event shading.
       Focuses on Data, Adjusted Diff, and Adjusted Ratio.
    """
    # Reduce to 3 rows as AlphaDev and GaugeZero are not reliably calculated by flag_anomalies (v1)
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=("1. Data & Flags", "2. Adjusted Difference (Network)", "3. Adjusted Ratio (Network)"),
                           vertical_spacing=0.06) # Increased spacing slightly

    # --- Panel 1: Data ---
    # Raw Gauge
    if 'Gauge_Data_mm_per_min' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Gauge_Data_mm_per_min'], mode='lines', name='Gauge', line=dict(color='blue')), row=1, col=1)
    # Raw Radar
    if 'Radar_Data_mm_per_min' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Data_mm_per_min'], mode='lines', name='Radar', line=dict(color='orange')), row=1, col=1)
    # Adj Radar (Self)
    if 'Alpha' in df.columns and 'Radar_Data_mm_per_min' in df.columns:
        alpha_filled = df['Alpha'].ffill().bfill().infer_objects(copy=False)
        if not alpha_filled.isnull().all():
            initial_adjusted_radar = (df['Radar_Data_mm_per_min'] + CFG.EPSILON) * alpha_filled
            fig.add_trace(go.Scatter(x=initial_adjusted_radar.index, y=initial_adjusted_radar, mode='lines', name='Adj. Radar (Self)', line=dict(color='green', dash='dashdot')), row=1, col=1)
    # Adj Radar (Network)
    if 'Network_Adjusted_Radar' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Network_Adjusted_Radar'], mode='lines', name='Adj. Radar (Net)', line=dict(color='purple', dash='dot')), row=1, col=1)
    # Flags
    if 'Flagged' in df.columns and df['Flagged'].any():
         gauge_data_for_flags = df['Gauge_Data_mm_per_min'] if 'Gauge_Data_mm_per_min' in df.columns else pd.Series(0, index=df.index)
         fig.add_trace(go.Scatter(x=df.index[df['Flagged']], y=gauge_data_for_flags[df['Flagged']], mode='markers', name='Flagged', marker=dict(color='red', size=6, symbol='x')), row=1, col=1)

    # --- Panel 2: Adjusted Difference (Network) ---
    # Plot the final adjusted difference used for flagging
    if 'Adjusted_Diff_from_network' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Diff_from_network'], mode='lines', name='Adj. Diff (Net)', line=dict(color='brown')), row=2, col=1) # Changed color
        # Threshold lines
        fig.add_hline(y=CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=-CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='grey', dash='dot', width=1), row=2, col=1) # Zero line
    else:
        fig.add_annotation(text="Adjusted_Diff_from_network missing", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=2, col=1)


    # --- Panel 3: Adjusted Ratio (Network) ---
    # Plot the final adjusted ratio used for flagging
    if 'Adjusted_Ratio_From_Network' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Ratio_From_Network'], mode='lines', name='Adj. Ratio (Net)', line=dict(color='teal')), row=3, col=1) # Changed color
        # Threshold lines
        fig.add_hline(y=1 + CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=3, col=1)
        fig.add_hline(y=1 - CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=3, col=1)
        fig.add_hline(y=1, line=dict(color='grey', dash='dot', width=1), row=3, col=1) # Ratio = 1 line
        # Set y-axis range for better visibility, e.g., 0 to 3 or based on data
        # fig.update_yaxes(range=[0, 3], row=3, col=1) # Example fixed range
    else:
         fig.add_annotation(text="Adjusted_Ratio_From_Network missing", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=3, col=1)

    # Event Shading (apply to all subplots)
    if events_df is not None and not events_df.empty:
        for _, event in events_df.iterrows():
            event_color = CFG.EVENT_FLAGGED_COLOR if event['is_flagged_event'] else CFG.EVENT_NORMAL_COLOR
            # Add vrect to all rows
            for r in range(1, 4): # Rows 1, 2, 3
                 fig.add_vrect(x0=event['event_start'], x1=event['event_end'],
                               fillcolor=event_color, layer="below", line_width=0, row=r, col=1)

    # Layout
    # Adjusted height for 3 panels
    fig.update_layout(height=900, title_text=f"Flagging Metrics for {coord_name}", template=CFG.PLOT_TEMPLATE,
                      showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)) # Legend above

    # Link x-axes (already done by shared_xaxes=True)
    # Add y-axis titles
    fig.update_yaxes(title_text="Rainfall (mm/min)", row=1, col=1)
    fig.update_yaxes(title_text="Difference (mm/min)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=3, col=1)


    return fig


# --- MODIFIED: create_flagging_plots_dashboard ---
def create_flagging_plots_dashboard(all_data: dict, events_df: pd.DataFrame = None,
                                    output_dir: str = str(CFG.FLAGGING_PLOTS_DIR),
                                    dashboard_file: str = str(CFG.FLAGGING_DASHBOARD_FILE)):
    """Create individual HTML plots for flagging metrics and a dashboard."""
    output_path = CFG.FLAGGING_PLOTS_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Creating flagging plots in: {output_path}")

    events_by_coord = {}
    if events_df is not None and not events_df.empty and 'sensor_coord' in events_df.columns:
        # Ensure event start/end are datetime objects if not already
        events_df['event_start'] = pd.to_datetime(events_df['event_start'])
        events_df['event_end'] = pd.to_datetime(events_df['event_end'])
        for coord, group in events_df.groupby('sensor_coord'):
            events_by_coord[coord] = group.copy() # Use copy to avoid SettingWithCopyWarning

    sensor_list = sorted(all_data.keys())
    for coord in tqdm(sensor_list):
        df_sensor = all_data[coord]
        # Ensure index is datetime
        if not isinstance(df_sensor.index, pd.DatetimeIndex):
            print(f"Warning: Index for {coord} is not DatetimeIndex, skipping flagging plot.")
            continue

        sensor_events_df = events_by_coord.get(coord, None)
        # Filter events to match the sensor's time range if necessary
        if sensor_events_df is not None and not df_sensor.empty:
            sensor_start, sensor_end = df_sensor.index.min(), df_sensor.index.max()
            sensor_events_df = sensor_events_df[
                (sensor_events_df['event_end'] >= sensor_start) &
                (sensor_events_df['event_start'] <= sensor_end)
            ]

        # Generate the plot using the (now 3-panel) function
        fig = plot_flagging_metrics(df_sensor, coord_name=coord, events_df=sensor_events_df)

        plot_file = output_path / f"{coord}.html"
        try:
            fig.write_html(str(plot_file), full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing flagging plot {plot_file}: {e}")

    # --- Dashboard HTML Generation ---
    # Adjust iframe height based on the new 3-panel plot height (e.g., 900px plot + padding)
    iframe_container_height = 950
    plot_dir_relative = f"../{CFG.FLAGGING_PLOTS_DIR.name}" # Relative path from dashboard HTML to plots folder

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
    function updatePlot(fileName) {{
        document.getElementById('plot-frame').src = fileName ? '{plot_dir_relative}/' + fileName : '';
    }}
    // Optional: Load the first sensor plot by default
    // var firstSensorFile = "{sensor_list[0]}.html" if sensor_list else "";
    // if (firstSensorFile) {{
    //    document.getElementById('sensor-dropdown').value = firstSensorFile;
    //    updatePlot(firstSensorFile);
    // }}
    </script>"""
    html_footer = """</body></html>"""
    final_html = html_header + html_options + html_middle + html_footer

    try:
         # Ensure the output directory exists
        CFG.DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print(f"Flagging dashboard created: {dashboard_file}")
    except Exception as e:
        print(f"Error writing flagging dashboard file {dashboard_file}: {e}")

# --- debug_alpha_for_coord remains the same ---
def debug_alpha_for_coord(coord: str, all_data: dict, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> go.Figure:
    """Plot gauge/radar and alpha values for the main sensor and its neighbors."""
    if coord not in all_data:
        raise ValueError(f"No data for {coord}")
    df_main = all_data[coord]
    neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=n_neighbors)
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=(f"1. Data {coord}", "2. Neighbors Data", "3. Alpha Comparison"), # Updated titles slightly
                           vertical_spacing=0.05)
    # Panel 1: Main sensor data
    if 'Gauge_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Gauge_Data_mm_per_min'], mode='lines',
                                 name=f"Gauge ({coord})", legendgroup="main", line=dict(color='blue')), row=1, col=1)
    if 'Radar_Data_mm_per_min' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Radar_Data_mm_per_min'], mode='lines',
                                 name=f"Radar ({coord})", legendgroup="main", line=dict(color='orange')), row=1, col=1)
    # Add adjusted radar (self) to main panel for context
    if 'Alpha' in df_main.columns and 'Radar_Data_mm_per_min' in df_main.columns:
        alpha_filled = df_main['Alpha'].ffill().bfill().infer_objects(copy=False)
        if not alpha_filled.isnull().all():
            initial_adjusted_radar = (df_main['Radar_Data_mm_per_min'] + CFG.EPSILON) * alpha_filled
            fig.add_trace(go.Scatter(x=initial_adjusted_radar.index, y=initial_adjusted_radar, mode='lines',
                                     name='Adj. Radar (Self)', legendgroup='main', line=dict(color='green', dash='dashdot')), row=1, col=1)
     # Add adjusted radar (network) to main panel for context
    if 'Network_Adjusted_Radar' in df_main.columns:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Network_Adjusted_Radar'], mode='lines',
                                 name='Adj. Radar (Net)', legendgroup='main', line=dict(color='purple', dash='dot')), row=1, col=1)


    # Panel 2: Neighbor data
    colors = colormap.viridis(np.linspace(0, 1, len(neighbors)))
    for i, neighbor in enumerate(neighbors):
        if neighbor in all_data:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            # Use showlegend=False for individual neighbor traces to avoid clutter, rely on color
            if 'Gauge_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Gauge_Data_mm_per_min'], mode='lines',
                                         name=f"G ({neighbor[:6]}..)", legendgroup=f"neigh_{i}", showlegend=True, # Show legend for neighbors here
                                         line=dict(color=color_hex, width=1)), row=2, col=1)
            if 'Radar_Data_mm_per_min' in df_n:
                fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Radar_Data_mm_per_min'], mode='lines',
                                         name=f"R ({neighbor[:6]}..)", legendgroup=f"neigh_{i}", showlegend=True, # Show legend for neighbors here
                                         line=dict(color=color_hex, dash='dot', width=1)), row=2, col=1)

    # Panel 3: Alpha values
    if 'Alpha' in df_main:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Alpha'], mode='lines',
                                 name=f"Alpha ({coord})", legendgroup="alpha", line=dict(color='red', width=2)), row=3, col=1)
    # Add Median Neighbor Alpha if available
    if 'Median_Neighbor_Alpha' in df_main:
         fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Median_Neighbor_Alpha'], mode='lines',
                                  name="Median Neighbor Alpha", legendgroup="alpha", line=dict(color='black', width=2, dash='dash')), row=3, col=1)

    for i, neighbor in enumerate(neighbors):
        if neighbor in all_data and 'Alpha' in all_data[neighbor]:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Alpha'], mode='lines',
                                     name=f"Alpha ({neighbor[:6]}..)", legendgroup="alpha", showlegend=True, # Show neighbor alphas
                                     line=dict(color=color_hex, width=1)), row=3, col=1)

    fig.add_hline(y=1, line=dict(color='grey', dash='dash', width=1), row=3, col=1) # Alpha=1 line
    fig.update_yaxes(title_text="Rainfall (mm/min)", row=1, col=1)
    fig.update_yaxes(title_text="Neighbor Rainfall (mm/min)", row=2, col=1)
    fig.update_yaxes(title_text="Alpha Value", row=3, col=1)
    fig.update_layout(height=1000, title_text=f"Alpha Debug Comparison: {coord}", template=CFG.PLOT_TEMPLATE,
                      hovermode="x unified", legend=dict(tracegroupgap=5)) # Adjust legend group spacing
    return fig


# --- MODIFIED: debug_alpha_and_neighbors_plot ---
def debug_alpha_and_neighbors_plot(coord: str, all_data: dict, coordinate_locations: dict, n_neighbors: int = CFG.N_NEIGHBORS) -> go.Figure:
    """Plots gauge/radar for the main sensor and each neighbor in separate subplots, plus a final subplot comparing Alpha values."""
    print('Creating debug plot for:', coord)
    if coord not in all_data:
        raise ValueError(f"No data for {coord}")

    df_main = all_data[coord].copy()
    neighbors = get_nearest_neighbors(coord, coordinate_locations, n_neighbors=n_neighbors)
    valid_neighbors = [n for n in neighbors if n in all_data] # Filter for neighbors with data

    if not valid_neighbors:
        print(f"Warning: No valid neighbors with data found for {coord}. Plotting main sensor only.")
        num_rows = 2 # Main sensor + Alpha comparison
        subplot_titles = [f"1. Main Sensor: {coord}", f"2. Alpha Comparison"]
    else:
        num_rows = 2 + len(valid_neighbors)
        subplot_titles = [f"1. Main Sensor: {coord}"]
        subplot_titles.extend([f"{i+2}. Neighbor: {n}" for i, n in enumerate(valid_neighbors)])
        subplot_titles.append(f"{num_rows}. Alpha Comparison")

    fig = sp.make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                           subplot_titles=subplot_titles, vertical_spacing=0.03) # Adjusted spacing

    # --- Panel 1: Main Sensor ---
    current_row = 1
    if 'Gauge_Data_mm_per_min' in df_main.columns:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Gauge_Data_mm_per_min'], mode='lines',
                                 name=f"Gauge ({coord})", legendgroup="main", showlegend=True, line=dict(color='blue')), row=current_row, col=1)
    if 'Radar_Data_mm_per_min' in df_main.columns:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Radar_Data_mm_per_min'], mode='lines',
                                 name=f"Radar ({coord})", legendgroup="main", showlegend=True, line=dict(color='orange')), row=current_row, col=1)
    # Adj Radar (Self)
    if 'Alpha' in df_main.columns and 'Radar_Data_mm_per_min' in df_main.columns:
        sensor_alpha = df_main['Alpha'].ffill().bfill().infer_objects(copy=False)
        if not sensor_alpha.isnull().all():
            adj_radar_self = (df_main['Radar_Data_mm_per_min'] + CFG.EPSILON) * sensor_alpha
            fig.add_trace(go.Scatter(x=adj_radar_self.index, y=adj_radar_self, mode='lines',
                                     name='Adj. Radar (Self)', legendgroup='main', showlegend=True, line=dict(color='green', dash='dashdot')), row=current_row, col=1)
    # Adj Radar (Network) - ADDED
    if 'Network_Adjusted_Radar' in df_main.columns:
         fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Network_Adjusted_Radar'], mode='lines',
                                  name='Adj. Radar (Network)', legendgroup='main', showlegend=True, line=dict(color='purple', dash='dot')), row=current_row, col=1)
    fig.update_yaxes(title_text="mm/min", row=current_row, col=1)


    # --- Panels 2 to N+1: Neighbors ---
    colors = colormap.viridis(np.linspace(0, 1, len(valid_neighbors))) # Color scale based on valid neighbors
    for i, neighbor in enumerate(valid_neighbors):
        current_row += 1
        df_n = all_data[neighbor]
        legend_group = f"neighbor_{i}"
        color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'

        if 'Gauge_Data_mm_per_min' in df_n.columns:
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Gauge_Data_mm_per_min'], mode='lines',
                                     name=f"G ({neighbor[:8]}..)", legendgroup=legend_group, showlegend=True, line=dict(color=color_hex, width=1)), row=current_row, col=1)
        if 'Radar_Data_mm_per_min' in df_n.columns:
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Radar_Data_mm_per_min'], mode='lines',
                                     name=f"R ({neighbor[:8]}..)", legendgroup=legend_group, showlegend=True, line=dict(color=color_hex, width=1, dash='dot')), row=current_row, col=1) # Added dash
        fig.update_yaxes(title_text="mm/min", row=current_row, col=1)


    # --- Final Panel: Alpha Comparison ---
    current_row += 1
    # Main sensor Alpha
    if 'Alpha' in df_main.columns:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Alpha'], mode='lines',
                                 name=f"Alpha ({coord})", legendgroup="alpha", showlegend=True, line=dict(color='red', width=2)), row=current_row, col=1)
    # Median Neighbor Alpha
    if 'Median_Neighbor_Alpha' in df_main.columns:
         fig.add_trace(go.Scatter(x=df_main.index, y=df_main['Median_Neighbor_Alpha'], mode='lines',
                                  name="Median Neighbor Alpha", legendgroup="alpha", showlegend=True, line=dict(color='black', width=2, dash='dash')), row=current_row, col=1)
    # Neighbor Alphas
    for i, neighbor in enumerate(valid_neighbors):
        if 'Alpha' in all_data[neighbor].columns:
            df_n = all_data[neighbor]
            color_hex = f'rgb({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)})'
            fig.add_trace(go.Scatter(x=df_n.index, y=df_n['Alpha'], mode='lines',
                                     name=f"Alpha ({neighbor[:8]}..)", legendgroup="alpha", showlegend=True, line=dict(color=color_hex, width=1)), row=current_row, col=1)

    fig.add_hline(y=1, line=dict(color='grey', dash='dash', width=1), row=current_row, col=1)
    fig.update_yaxes(title_text="Alpha", row=current_row, col=1)

    # Layout
    # Adjust height dynamically based on the number of rows
    plot_height = 250 * num_rows # Approx 250px per subplot
    fig.update_layout(height=plot_height, title_text=f"Debug Comparison: {coord} and Neighbors", template=CFG.PLOT_TEMPLATE,
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, tracegroupgap=10)) # Legend above plot

    return fig