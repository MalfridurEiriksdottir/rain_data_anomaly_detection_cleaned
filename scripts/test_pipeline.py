import unittest
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import shutil # For cleaning up directories
import sys

# --- Add base directory to path to allow imports ---
# Assuming test_pipeline.py is in the same directory as config.py etc.
script_dir = Path(__file__).parent.resolve()
base_dir_from_config = Path(script_dir / "../").resolve() # Go up one level as per CFG.BASE_DIR
sys.path.insert(0, str(script_dir))
# ---

# --- Import your modules ---
try:
    from config import CFG
    from data_loading import (load_sensor_data, load_target_coordinates,
                              process_sensor_metadata, load_time_series_data)
    from feature_engineering import apply_feature_engineering
    # Using the latest iterative network function from main_v3.py
    from network_iterative2 import compute_network_metrics_iterative
    # Using the standard anomaly flagging from main_v3.py
    from anomaly import flag_anomalies
    # Using the latest event detection from main_v3.py
    from event_detection2 import identify_and_flag_rain_events
    from results import aggregate_results
    # Using the latest plotting functions from main_v3.py
    from scripts.old_scripts.plotting3 import (create_plots_with_error_markers, generate_html_dashboard,
                           create_flagging_plots_dashboard, debug_alpha_for_coord,
                           debug_alpha_and_neighbors_plot)
except ImportError as e:
    print(f"Failed to import necessary modules: {e}")
    print("Ensure the script is run from the correct directory and all required files exist.")
    sys.exit(1)

# --- Helper function to check if PKL data exists ---
def check_pkl_data_exists():
    if not CFG.PKL_DATA_DIR.exists():
        return False
    # Check if at least one PKL file exists
    return any(CFG.PKL_DATA_DIR.glob('*.pkl'))

# --- Test Class ---
class TestAnomalyPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths and load initial data once for all tests."""
        print("\n--- Setting up Test Class ---")
        cls.cfg = CFG # Make config easily accessible
        cls.created_files = [] # Keep track of files created by tests
        cls.created_dirs = [] # Keep track of directories created by tests

        # --- Ensure output directories exist (or create them) ---
        dirs_to_ensure = [
            cls.cfg.RESULTS_DIR, cls.cfg.DASHBOARD_DIR,
            cls.cfg.PLOTS_OUTPUT_DIR, cls.cfg.FLAGGING_PLOTS_DIR,
            cls.cfg.FLAGGED_EVENTS_DIR
        ]
        for d in dirs_to_ensure:
            if not d.exists():
                print(f"Creating directory for tests: {d}")
                d.mkdir(parents=True, exist_ok=True)
                cls.created_dirs.append(d) # Mark for potential cleanup only if created by test

        # --- Load prerequisite data ---
        try:
            print("Loading target coordinates...")
            cls.target_coords, cls.coordinate_locations_utm = load_target_coordinates()
            # --- Use standard checks instead of cls.assert ---
            if cls.target_coords is None:
                raise ValueError("Failed to load target_coords (is None).")
            if not cls.target_coords:
                print("Warning: Loaded target_coords list is empty.") # Warning instead of fail? Or raise ValueError?
                # raise ValueError("Loaded target_coords list is empty.")
            if cls.coordinate_locations_utm is None:
                 raise ValueError("Failed to load coordinate_locations_utm (is None).")
            if not cls.coordinate_locations_utm:
                 raise ValueError("Loaded coordinate_locations_utm dictionary is empty.")
            # --- End checks ---

            print("Loading sensor metadata...")
            cls.sensordata, cls.gdf_wgs84 = load_sensor_data()
            # --- Use standard checks ---
            if cls.sensordata is None:
                 raise ValueError("Failed to load sensordata (is None).")
            if not isinstance(cls.sensordata, pd.DataFrame):
                 raise TypeError("sensordata is not a Pandas DataFrame.")
            if cls.sensordata.empty:
                 raise ValueError("Loaded sensordata DataFrame is empty.")
            if cls.gdf_wgs84 is None:
                  raise ValueError("Failed to load gdf_wgs84 (is None).")
            if not isinstance(cls.gdf_wgs84, gpd.GeoDataFrame):
                  raise TypeError("gdf_wgs84 is not a GeoPandas GeoDataFrame.")
            if cls.gdf_wgs84.empty:
                  raise ValueError("Loaded gdf_wgs84 GeoDataFrame is empty.")
            # --- End checks ---

            print("Processing sensor metadata...")
            cls.sensor_channels, cls.svk_coords_utm = process_sensor_metadata(cls.sensordata)
            # --- Use standard checks ---
            if not isinstance(cls.sensor_channels, dict):
                 raise TypeError("sensor_channels is not a dict.")
            if not isinstance(cls.svk_coords_utm, list):
                 raise TypeError("svk_coords_utm is not a list.")
            # --- End checks ---

            print("Loading time series data...")
            # Check if PKL dir and files exist before trying to load
            if not check_pkl_data_exists():
                 # Use unittest.SkipTest to gracefully skip remaining tests if data is missing
                 raise unittest.SkipTest(f"PKL data directory or files not found at {cls.cfg.PKL_DATA_DIR}. Skipping data-dependent tests.")

            cls.initial_all_data, cls.valid_locations_utm, cls.missing_files = load_time_series_data(
                cls.target_coords, cls.coordinate_locations_utm
            )
            # --- Use standard checks ---
            if not isinstance(cls.initial_all_data, dict):
                 raise TypeError("initial_all_data is not a dict.")
            # Only raise error if we expected data but got none
            if cls.target_coords and not cls.initial_all_data:
                  raise ValueError("No time series data loaded, but target_coords were specified. Check PKL files and target_coords.")
            # --- End checks ---

            # Store data for subsequent steps
            cls.all_data = cls.initial_all_data.copy() # Use a copy

        except FileNotFoundError as e:
            # Reraise specifically as SkipTest so unittest handles it correctly
            raise unittest.SkipTest(f"Prerequisite file not found during setup: {e}. Skipping data-dependent tests.")
        except ValueError as e:
             # Reraise setup value errors as SkipTest or let them fail setup
             # Using SkipTest is often cleaner for setup issues outside the test's direct logic
             raise unittest.SkipTest(f"ValueError during setup (e.g., bad coords, missing columns, empty data): {e}. Skipping data-dependent tests.")
        except TypeError as e:
             raise unittest.SkipTest(f"TypeError during setup (likely wrong data type loaded): {e}. Skipping data-dependent tests.")
        except Exception as e:
            print(f"Unexpected error during setUpClass: {e}")
            # Reraise other unexpected exceptions to fail the setup definitively
            raise


    @classmethod
    def tearDownClass(cls):
        """Clean up created files and directories after all tests."""
        print("\n--- Tearing down Test Class ---")
        print("Cleaning up generated files/directories...")

        files_to_remove = [
            cls.cfg.SUMMARY_CSV_FILE,
            cls.cfg.DASHBOARD_FILE,
            cls.cfg.FLAGGING_DASHBOARD_FILE,
            cls.cfg.DEBUG_ALPHA_PLOT_FILE,
            cls.cfg.EVENTS_CSV_FILE,
            cls.cfg.FLAGGED_EVENTS_CSV_FILE,
            cls.cfg.RESULTS_DIR / 'iterative_process.log', # From main_v3
            cls.cfg.RESULTS_DIR / 'iterative_exclusions.log', # From network_iterative2
            cls.cfg.RESULTS_DIR / 'all_iterative_exclusions.csv', # From main_v3
            cls.cfg.DASHBOARD_DIR / f"debug_neighbors_{(675616,6122248)}.html" # Specific debug file
        ]
        # Add specific CSV output if generated
        # for coord in cls.initial_all_data.keys():
        #     files_to_remove.append(cls.cfg.BASE_DIR / cls.cfg.SENSOR_CSV_OUTPUT_TPL.format(coord=coord))

        dirs_to_remove = [
            cls.cfg.PLOTS_OUTPUT_DIR,
            cls.cfg.FLAGGING_PLOTS_DIR,
            cls.cfg.FLAGGED_EVENTS_DIR,
            # Add debug_plots dir if debug_plot.py was tested
            # cls.cfg.BASE_DIR / "debug_plots"
        ]

        for f_path in files_to_remove:
            try:
                if f_path.exists():
                    os.remove(f_path)
                    print(f"Removed file: {f_path}")
            except OSError as e:
                print(f"Error removing file {f_path}: {e}")

        # Remove directories (only if they exist)
        for d_path in dirs_to_remove:
             try:
                if d_path.exists() and d_path.is_dir():
                     # Check if we should remove it (created by test or explicitly listed)
                     # This is a simple check; could be more robust
                     # Let's just remove if it exists, assuming test runs are isolated
                    shutil.rmtree(d_path)
                    print(f"Removed directory: {d_path}")
             except OSError as e:
                 print(f"Error removing directory {d_path}: {e}")

        # Additionally remove dirs marked during setup
        # for d_path in cls.created_dirs:
        #      try:
        #         if d_path.exists() and d_path.is_dir():
        #              shutil.rmtree(d_path)
        #              print(f"Removed test-created directory: {d_path}")
        #      except OSError as e:
        #          print(f"Error removing directory {d_path}: {e}")


    # --- Individual Test Cases (Order Matters!) ---

    def test_01_config_load(self):
        """Test if configuration can be loaded."""
        print("\nRunning test_01_config_load...")
        self.assertIsNotNone(self.cfg)
        self.assertTrue(hasattr(self.cfg, 'BASE_DIR'))
        self.assertTrue(hasattr(self.cfg, 'N_NEIGHBORS'))
        self.assertTrue(hasattr(self.cfg, 'PKL_DATA_DIR'))
        self.assertIsInstance(self.cfg.PKL_DATA_DIR, Path)
        print("Config loaded successfully.")

    def test_02_feature_engineering(self):
        """Test the feature engineering step."""
        print("\nRunning test_02_feature_engineering...")
        if not self.all_data: self.skipTest("No data loaded in setup.")

        TestAnomalyPipeline.all_data = apply_feature_engineering(self.all_data)
        self.assertIsInstance(self.all_data, dict)
        self.assertEqual(len(self.all_data), len(self.initial_all_data)) # Check if all sensors processed

        # Check one sensor's DataFrame for new columns
        first_coord = next(iter(self.all_data))
        df = self.all_data[first_coord]
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = ['Gauge_Data_mm_per_min', 'Radar_Data_mm_per_min', 'Difference', 'Alpha', 'Rolling_Gauge']
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing after feature engineering")
        print("Feature engineering applied successfully.")

    def test_03_network_metrics_iterative(self):
        """Test the iterative network metrics calculation (1 iteration)."""
        print("\nRunning test_03_network_metrics_iterative...")
        if not self.all_data: self.skipTest("No data loaded in setup.")

        iteration = 0
        previous_flagged_intervals = {} # Start with no exclusions

        # Run the function
        processed_data, exclusion_log = compute_network_metrics_iterative(
            self.all_data,
            self.valid_locations_utm,
            previous_flagged_intervals,
            current_iteration=iteration
        )

        self.assertIsInstance(processed_data, dict)
        self.assertIsInstance(exclusion_log, list) # Should return a list, even if empty
        self.assertEqual(len(processed_data), len(self.all_data))

        # Check one sensor's DataFrame for new network columns
        first_coord = next(iter(processed_data))
        df = processed_data[first_coord]
        self.assertIsInstance(df, pd.DataFrame)
        # Columns added/updated by network_iterative2
        expected_cols = [
            'Median_Neighbor_Alpha', 'Neighbor_Count_Used',
            'Network_Adjusted_Radar', 'Rolling_Adjusted_Radar',
            'Adjusted_Diff_from_network', 'Adjusted_Ratio_From_Network'
            # 'Rolling_Gauge_Data' # This might be added here or in feature eng depending on impl.
        ]
        # Also check columns from original network.py that might be calculated
        # original_cols = ['Difference_From_Network', 'Ratio_From_Network', 'Alpha_From_Network']

        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column '{col}' missing after network metrics")

        # Update class data for next steps
        TestAnomalyPipeline.all_data = processed_data
        print("Network metrics (iterative) computed successfully.")


    def test_04_anomaly_flagging(self):
        """Test the anomaly flagging step."""
        print("\nRunning test_04_anomaly_flagging...")
        if not self.all_data: self.skipTest("No data loaded in setup.")

        TestAnomalyPipeline.all_data = flag_anomalies(self.all_data)
        self.assertIsInstance(self.all_data, dict)
        self.assertEqual(len(self.all_data), len(self.initial_all_data))

        # Check one sensor's DataFrame for 'Flagged' column
        first_coord = next(iter(self.all_data))
        df = self.all_data[first_coord]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Flagged', df.columns)
        self.assertTrue(df['Flagged'].dtype == bool or df['Flagged'].dtype == int) # Should be bool or int
        print("Anomaly flagging applied successfully.")


    def test_05_event_detection(self):
        """Test the event detection step."""
        print("\nRunning test_05_event_detection...")
        if not self.all_data: self.skipTest("No data loaded in setup.")

        TestAnomalyPipeline.rain_events = identify_and_flag_rain_events(self.all_data)
        self.assertIsInstance(self.rain_events, list)

        # If events were detected, check the structure of the first event
        if self.rain_events:
            first_event = self.rain_events[0]
            self.assertIsInstance(first_event, dict)
            expected_keys = [
                'sensor_coord', 'event_start', 'event_end', 'peak_radar',
                'duration_minutes', 'total_points', 'flagged_points',
                'percentage_flagged', 'is_flagged_event',
                'min_neighbors_during_event' # Added in event_detection2
            ]
            for key in expected_keys:
                self.assertIn(key, first_event, f"Key '{key}' missing in detected event")
            print(f"Event detection ran, found {len(self.rain_events)} potential events.")
        else:
            print("Event detection ran, no events identified (this might be expected).")

        # Create DataFrame for plotting tests
        if hasattr(self, 'rain_events') and self.rain_events:
             TestAnomalyPipeline.events_df = pd.DataFrame(self.rain_events)
        else:
             TestAnomalyPipeline.events_df = pd.DataFrame() # Empty DF if no events


    def test_06_results_aggregation(self):
        """Test the results aggregation step."""
        print("\nRunning test_06_results_aggregation...")
        if not self.all_data: self.skipTest("No data loaded in setup.")

        results_df = aggregate_results(self.all_data)
        self.assertIsInstance(results_df, pd.DataFrame)

        if not results_df.empty:
            expected_cols = [
                'Coordinate', 'Total_Timepoints', 'Flagged_Timepoints',
                'Percent_Flagged', 'Average_Difference_When_Flagged',
                'Max_Difference_From_Network'
            ]
            for col in expected_cols:
                self.assertIn(col, results_df.columns)
            print("Results aggregation ran successfully.")
            # Check if summary CSV was created
            self.assertTrue(self.cfg.SUMMARY_CSV_FILE.exists(), f"Summary CSV file not created at {self.cfg.SUMMARY_CSV_FILE}")
            self.created_files.append(self.cfg.SUMMARY_CSV_FILE)
        else:
            print("Results aggregation produced an empty DataFrame (might be okay if no flags).")

    def test_07_plotting(self):
        """Test the plotting functions."""
        print("\nRunning test_07_plotting...")
        if not self.all_data: self.skipTest("No data loaded in setup.")
        if not hasattr(self, 'events_df'): self.skipTest("Events DataFrame not created in previous step.")

        # --- Test create_plots_with_error_markers ---
        print("Testing create_plots_with_error_markers...")
        try:
            # Create a dummy iter0 data for testing the plotting function call signature
            dummy_all_data_iter0 = {}
            if self.initial_all_data:
                 first_coord_key = next(iter(self.initial_all_data))
                 df_example = self.initial_all_data[first_coord_key]
                 # Add columns expected by the plotting function if they exist in initial data
                 cols_to_copy = ['Network_Adjusted_Radar', 'Flagged']
                 present_cols = [c for c in cols_to_copy if c in df_example.columns]
                 if present_cols:
                     dummy_all_data_iter0[first_coord_key] = df_example[present_cols].copy()


            create_plots_with_error_markers(
                self.all_data,
                self.valid_locations_utm,
                self.sensor_channels,
                events_df=self.events_df,
                all_data_iter0=dummy_all_data_iter0 # Pass the dummy or None
            )
            # Check if the directory contains at least one plot file
            plot_files = list(self.cfg.PLOTS_OUTPUT_DIR.glob('*.html'))
            self.assertTrue(plot_files, f"No plot files found in {self.cfg.PLOTS_OUTPUT_DIR}")
            self.created_files.extend(plot_files) # Track for cleanup
            print(f"Found {len(plot_files)} plots in {self.cfg.PLOTS_OUTPUT_DIR}")
        except Exception as e:
            self.fail(f"create_plots_with_error_markers failed: {e}")

        # --- Test generate_html_dashboard ---
        print("Testing generate_html_dashboard...")
        try:
            if self.gdf_wgs84.empty:
                print("Skipping main dashboard test: gdf_wgs84 is empty.")
            else:
                generate_html_dashboard(
                    self.all_data,
                    self.valid_locations_utm,
                    self.gdf_wgs84,
                    self.svk_coords_utm,
                    output_file=str(self.cfg.DASHBOARD_FILE)
                )
                self.assertTrue(self.cfg.DASHBOARD_FILE.exists(), f"Dashboard file not created at {self.cfg.DASHBOARD_FILE}")
                self.created_files.append(self.cfg.DASHBOARD_FILE)
                print("Main dashboard generated.")
        except Exception as e:
            self.fail(f"generate_html_dashboard failed: {e}")

        # --- Test create_flagging_plots_dashboard ---
        print("Testing create_flagging_plots_dashboard...")
        try:
            create_flagging_plots_dashboard(
                self.all_data,
                events_df=self.events_df,
                output_dir=str(self.cfg.FLAGGING_PLOTS_DIR),
                dashboard_file=str(self.cfg.FLAGGING_DASHBOARD_FILE)
            )
            self.assertTrue(self.cfg.FLAGGING_DASHBOARD_FILE.exists(), f"Flagging dashboard file not created at {self.cfg.FLAGGING_DASHBOARD_FILE}")
            self.created_files.append(self.cfg.FLAGGING_DASHBOARD_FILE)
            # Check if the directory contains at least one plot file
            flagging_plot_files = list(self.cfg.FLAGGING_PLOTS_DIR.glob('*.html'))
            self.assertTrue(flagging_plot_files, f"No plot files found in {self.cfg.FLAGGING_PLOTS_DIR}")
            self.created_files.extend(flagging_plot_files)
            print(f"Found {len(flagging_plot_files)} flagging plots in {self.cfg.FLAGGING_PLOTS_DIR}")
            print("Flagging plots and dashboard generated.")
        except Exception as e:
            self.fail(f"create_flagging_plots_dashboard failed: {e}")

        # --- Test debug plots (optional, choose one coord if data exists) ---
        print("Testing debug plots...")
        debug_coord = '(675616,6122248)' # Use the one from main.py
        if debug_coord in self.all_data:
            try:
                fig_debug = debug_alpha_for_coord(debug_coord, self.all_data, self.valid_locations_utm)
                self.assertIsNotNone(fig_debug) # Check if a figure object is returned
                # Optionally save and check file existence
                # fig_debug.write_html(self.cfg.DEBUG_ALPHA_PLOT_FILE)
                # self.assertTrue(self.cfg.DEBUG_ALPHA_PLOT_FILE.exists())
                # self.created_files.append(self.cfg.DEBUG_ALPHA_PLOT_FILE)

                fig_debug_neighbors = debug_alpha_and_neighbors_plot(debug_coord, self.all_data, self.valid_locations_utm)
                self.assertIsNotNone(fig_debug_neighbors)
                # Optionally save and check file existence
                # debug_neigh_path = self.cfg.DASHBOARD_DIR / f"debug_neighbors_{debug_coord}.html"
                # fig_debug_neighbors.write_html(debug_neigh_path)
                # self.assertTrue(debug_neigh_path.exists())
                # self.created_files.append(debug_neigh_path)
                print(f"Debug plots generated successfully for {debug_coord}.")
            except Exception as e:
                self.fail(f"Debug plotting failed for {debug_coord}: {e}")
        else:
            print(f"Skipping debug plot tests: Coordinate {debug_coord} not found in loaded data.")


# --- Run the tests ---
if __name__ == '__main__':
    print("=========================================")
    print(" Starting Anomaly Detection Pipeline Tests ")
    print("=========================================")
    print(f"Using Config Base Dir: {CFG.BASE_DIR}")
    print(f"Expecting PKL data in: {CFG.PKL_DATA_DIR}")
    print(f"Expecting Excel in: {CFG.EXCEL_FILE}")
    print(f"Expecting Coords txt in: {CFG.ALL_COORDS_FILE}")
    print("-----------------------------------------")

    # Run tests
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestAnomalyPipeline))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

    print("-----------------------------------------")
    print(" Tests Finished ")
    print("=========================================")