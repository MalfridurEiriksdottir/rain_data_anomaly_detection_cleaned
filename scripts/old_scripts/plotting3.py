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

# plotting.py

# ... other imports ...
import logging # Add logging import if not already there
logger = logging.getLogger(__name__) # Get logger instance

# ... generate_html_dashboard ...

# --- MODIFIED create_plots_with_error_markers function ---
def create_plots_with_error_markers(
    all_data: dict, # Data from the FINAL iteration
    coordinate_locations_utm: dict,
    sensor_channels: dict,
    events_df: pd.DataFrame = None,
    all_data_iter0: dict = None # New optional argument for Iteration 0 data
    ):
    """
    Create interactive Plotly time series plots for each sensor and save as HTML.
    Plots raw data, final adjusted radar, anomalies, and optionally Iteration 0 adjusted radar.
    """
    warnings.filterwarnings("ignore")
    output_dir = CFG.PLOTS_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating time series plots (comparing final vs iter 0) in: {output_dir}")

    # Group events by coordinate (remains the same)
    events_by_coord = {}
    if events_df is not None and not events_df.empty:
        for coord, group in events_df.groupby('sensor_coord'):
            events_by_coord[coord] = group

    for coord in tqdm(all_data.keys(), desc="Generating Plots"):
        df_final = all_data[coord] # DataFrame from the final iteration
        fig = go.Figure()

        # --- Plot Raw Data ---
        if 'Radar_Data_mm_per_min' in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Radar_Data_mm_per_min'], mode='lines', name='Radar',
                                     line=dict(color='orange', width=1)))
        if 'Gauge_Data_mm_per_min' in df_final:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Gauge_Data_mm_per_min'], mode='lines', name='Gauge',
                                     line=dict(color='blue', width=1)))

        # --- Plot FINAL Adjusted Radar ---
        # Use Network_Adjusted_Radar from the final iteration data
        if 'Network_Adjusted_Radar' in df_final.columns:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Network_Adjusted_Radar'], mode='lines',
                                     name='Adj. Radar (Iter 1)',
                                     line=dict(color='purple', dash='dashdot', width=1.5))) # Solid purple
        else:
             # Log instead of just printing
             logger.warning(f"Final 'Network_Adjusted_Radar' missing for plotting {coord}")


        # --- Plot ITERATION 0 Adjusted Radar (if available) ---
        if all_data_iter0 and coord in all_data_iter0:
            df_iter0 = all_data_iter0[coord] # Get the stored iter 0 data
            if 'Network_Adjusted_Radar' in df_iter0.columns:
                # Align Iter0 data index to final data index just in case they differ slightly
                # Use outer join to keep all timestamps, fill missing with NaN
                aligned_iter0_adj_radar = df_iter0['Network_Adjusted_Radar'].reindex(df_final.index)

                fig.add_trace(go.Scatter(x=aligned_iter0_adj_radar.index, y=aligned_iter0_adj_radar, mode='lines',
                                         name='Adj. Radar (Iter 0)',
                                         line=dict(color='green', dash='dashdot', width=1.5))) # Dotted green
            else:
                 logger.warning(f"Iter 0 'Network_Adjusted_Radar' missing for plotting {coord}")

        # --- Plot Flagged Anomalies (from final iteration) ---
        if 'Flagged' in df_final and df_final['Flagged'].any():
            # Ensure we plot flags corresponding to the gauge data points
            flagged_gauge_points = df_final['Gauge_Data_mm_per_min'].where(df_final['Flagged'])
            if not flagged_gauge_points.isnull().all(): # Check if there are any non-NaN flagged points
                 fig.add_trace(go.Scatter(x=flagged_gauge_points.index, y=flagged_gauge_points,
                                          mode='markers', name='Anomalies (Iter 1)',
                                          marker=dict(color='red', size=6, symbol='x')))
                 
                # --- Plot Flagged Anomalies (from ITERATION 0) ---
        if all_data_iter0 and coord in all_data_iter0:
            # print('HALLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
            df_iter0 = all_data_iter0[coord]
            # Check if Iter 0 flags and Final gauge exist
            if 'Flagged' in df_iter0.columns and 'Gauge_Data_mm_per_min' in df_final and df_iter0['Flagged'].any():
                # print('HALLOOOOOOO')
                # Align Iter0 flags to the final index before masking
                flags_iter0_aligned = df_iter0['Flagged'].reindex(df_final.index, fill_value=False)
                # Use aligned iter0 flags to mask the FINAL gauge data for consistent y-positioning
                flagged_gauge_points_iter0 = df_final['Gauge_Data_mm_per_min'].where(flags_iter0_aligned)

                if not flagged_gauge_points_iter0.isnull().all():
                    # print('HALLOOOOOOO2')
                    fig.add_trace(go.Scatter(x=flagged_gauge_points_iter0.index, y=flagged_gauge_points_iter0,
                                             mode='markers', name='Anomalies (Iter 0)',
                                             marker=dict(color='darkorange', size=6, symbol='circle-open'))) # Orange open circle

        # --- Plot Event Shading (based on final iteration events) ---
        sensor_events = events_by_coord.get(coord, pd.DataFrame())
        if not sensor_events.empty:
            for _, event in sensor_events.iterrows():
                # Make sure event times are valid before plotting vrect
                if pd.notna(event['event_start']) and pd.notna(event['event_end']):
                    event_color = CFG.EVENT_FLAGGED_COLOR if event['is_flagged_event'] else CFG.EVENT_NORMAL_COLOR
                    fig.add_vrect(x0=event['event_start'], x1=event['event_end'],
                                  fillcolor=event_color, layer="below", line_width=0)
                else:
                    logger.warning(f"Invalid start/end time for an event in {coord}, skipping vrect.")


        # --- Layout ---
        layout_update = dict(
            title=f"Time Series for {coord} (Final vs Iter 0 Adjustment)",
            xaxis_title="Time",
            yaxis_title="Rainfall Rate (mm/min)",
            template=CFG.PLOT_TEMPLATE,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_layout(**layout_update)

        # --- Save Plot ---
        output_path = output_dir / f"{coord}.html"
        try:
            fig.write_html(str(output_path), full_html=False, include_plotlyjs='cdn')
        except Exception as e:
            print(f"Error writing plot {output_path}: {e}")
            # Log the error with traceback
            logger.error(f"Error writing plot {output_path}: {e}", exc_info=True)

    print(f"Time series plots saved to {output_dir}")

# ... rest of plotting.py (plot_flagging_metrics, create_flagging_plots_dashboard, etc.) ...

# --- Add this corrected function to your plotting.py ---
def plot_flagging_metrics(df: pd.DataFrame, coord_name: str = "", events_df: pd.DataFrame = None) -> go.Figure:
    """Create a multi-panel Plotly figure showing flagging metrics with event shading."""
    # Define the subplot titles based on what you intend to plot
    subplot_titles = (
        "1. Data & Flags",
        "2. Adj.DiffFromNet",
        "3. Adj.RatioFromNet",
        "4. Neighbor Count", # Updated title
        "5. Gauge=0 Cond."   # Optional, plot if column exists
    )
    fig = sp.make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04
    )

    # Row 1: Data & Flags
    if 'Radar_Data_mm_per_min' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Radar_Data_mm_per_min'], mode='lines', name='Radar', line=dict(color='orange')), row=1, col=1)
    if 'Gauge_Data_mm_per_min' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Gauge_Data_mm_per_min'], mode='lines', name='Gauge', line=dict(color='blue')), row=1, col=1)
    if 'Network_Adjusted_Radar' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Network_Adjusted_Radar'], mode='lines', name='Adj. Radar (Network)', line=dict(color='purple', dash='dot')), row=1, col=1)
    if 'Flagged' in df and df['Flagged'].any():
        flagged_gauge = df['Gauge_Data_mm_per_min'].where(df['Flagged'])
        if not flagged_gauge.isnull().all():
            fig.add_trace(go.Scatter(x=flagged_gauge.index, y=flagged_gauge, mode='markers', name='Flagged', marker=dict(color='red', size=6, symbol='x')), row=1, col=1)

    # Event Shading (Apply to all rows)
    if events_df is not None and not events_df.empty:
        sensor_events_for_plot = events_df[events_df['sensor_coord'] == coord_name]
        for _, event in sensor_events_for_plot.iterrows():
            if pd.notna(event['event_start']) and pd.notna(event['event_end']):
                event_color = CFG.EVENT_FLAGGED_COLOR if event['is_flagged_event'] else CFG.EVENT_NORMAL_COLOR
                fig.add_vrect(x0=event['event_start'], x1=event['event_end'], fillcolor=event_color, layer="below", line_width=0, row='all', col=1)

    # Row 2: Adjusted Difference
    if 'Adjusted_Diff_from_network' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Diff_from_network'], mode='lines', name='Adj.DiffFromNet', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=2, col=1)
    fig.add_hline(y=-CFG.ABS_DIFF_THRESHOLD_MM_MIN, line=dict(color='red', dash='dash', width=1), row=2, col=1)

    # Row 3: Adjusted Ratio
    if 'Adjusted_Ratio_From_Network' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Adjusted_Ratio_From_Network'], mode='lines', name='Adj.RatioFromNet', line=dict(color='green')), row=3, col=1)
    fig.add_hline(y=1 + CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=3, col=1)
    fig.add_hline(y=1 - CFG.RATIO_THRESHOLD, line=dict(color='red', dash='dash', width=1), row=3, col=1)
    fig.add_hline(y=1, line=dict(color='grey', dash='dot', width=1), row=3, col=1)
    # Optional: Set y-axis range for ratio if needed, e.g., fig.update_yaxes(range=[0, 3.5], row=3, col=1)

    # Row 4: Neighbor Count Used
    if 'Neighbor_Count_Used' in df: fig.add_trace(go.Scatter(x=df.index, y=df['Neighbor_Count_Used'], mode='lines', name='Neighbor Count', line=dict(color='brown')), row=4, col=1)
    # Set y-axis range for count (e.g., 0 to max possible neighbors + buffer)
    fig.update_yaxes(range=[-0.5, CFG.N_NEIGHBORS + 0.5], row=4, col=1, dtick=1) # Add dtick=1 for integer ticks

    # Row 5: Gauge Zero Condition (Plot if column exists)
    if 'Gauge_Zero_Condition' in df:
         fig.add_trace(go.Scatter(x=df.index, y=df['Gauge_Zero_Condition'].astype(int), mode='lines', name='Gauge=0 Cond', line=dict(color='black', shape='hv')), row=5, col=1)
    # Set y-axis for boolean condition
    fig.update_yaxes(range=[-0.1, 1.1], row=5, col=1, fixedrange=True, dtick=1)


    fig.update_layout(height=1400, title_text=f"Flagging Metrics for {coord_name}", template=CFG.PLOT_TEMPLATE, showlegend=True, hovermode="x unified")

    return fig
# --- End corrected function ---
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
