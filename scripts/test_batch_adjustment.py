import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
import logging

# --- MOCK CFG for Testing ---
# It's often better to mock or create a test-specific config
# Here, we'll modify the existing CFG object for simplicity,
# but be careful as this affects the global state if tests run in parallel
# or if CFG is imported elsewhere before modification.
from config import CFG

# --- Modules to Test ---
# Import functions/classes from your project files
from data_loading import (load_sensor_data, load_target_coordinates,
                          process_sensor_metadata, load_time_series_data)
from feature_engineering import create_features, apply_feature_engineering
from network import get_nearest_neighbors # Add others if needed
from anomaly import flag_anomalies
from batch_adjustment import compute_regional_adjustment
# Import the main function if you want to run end-to-end tests
# from main_v3_batch3 import main_batch_adjustment # Be cautious with full runs

# --- Test Setup ---
TEST_DATA_DIR = Path("./test_data")
TEST_OUTPUT_DIR = Path("./test_output") # Directory for test outputs

# Disable logging spam during tests (optional)
# logging.disable(logging.CRITICAL)

class TestRainfallAdjustment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up paths and create output dir ONCE for all tests in this class."""
        print("Setting up test environment...")
        cls.original_paths = {}

        # Store original CFG paths and override with test paths
        paths_to_override = {
            'BASE_DIR': Path("."), # Assume script runs from project root
            'RESULTS_DIR': TEST_OUTPUT_DIR / 'results_files',
            'DASHBOARD_DIR': TEST_OUTPUT_DIR / 'dashboard_files',
            'PKL_DATA_DIR': TEST_DATA_DIR / 'dummy_all_data_pkl',
            'EXCEL_FILE': TEST_DATA_DIR / 'dummy_SensorOversigt.xlsx',
            'ALL_COORDS_FILE': TEST_DATA_DIR / 'dummy_all_coords.txt',
            'PLOTS_OUTPUT_DIR': TEST_OUTPUT_DIR / 'plots_output',
            'FLAGGING_PLOTS_DIR': TEST_OUTPUT_DIR / 'flagging_plots',
            'ADJUSTMENT_PLOTS_DIR': TEST_OUTPUT_DIR / 'adjustment_plots',
            'BATCH_ADJUSTMENT_DIR': TEST_OUTPUT_DIR / 'batch_adjustment_output',
            'DETAILED_CSV_DIR': TEST_OUTPUT_DIR / 'results_files' / 'detailed_sensor_output' # Assuming sub-dir
        }

        for key, value in paths_to_override.items():
            if hasattr(CFG, key):
                cls.original_paths[key] = getattr(CFG, key)
                setattr(CFG, key, value)
                # Ensure directories exist
                if 'DIR' in key or 'FILE' not in key: # Heuristic for directories
                    value.mkdir(parents=True, exist_ok=True)
            else:
                print(f"Warning: CFG does not have attribute {key}")

        # Create the detailed output dir specifically if not covered above
        detailed_csv_path = paths_to_override.get('DETAILED_CSV_DIR', CFG.RESULTS_DIR / "detailed_sensor_output")
        detailed_csv_path.mkdir(parents=True, exist_ok=True)
        setattr(CFG, 'DETAILED_CSV_DIR', detailed_csv_path) # Ensure it's set

        # --- Ensure Test Data Exists ---
        if not TEST_DATA_DIR.exists():
            raise FileNotFoundError(f"Test data directory not found: {TEST_DATA_DIR}")
        if not CFG.EXCEL_FILE.exists():
             raise FileNotFoundError(f"Dummy Excel file not found: {CFG.EXCEL_FILE}")
        if not CFG.ALL_COORDS_FILE.exists():
             raise FileNotFoundError(f"Dummy coords file not found: {CFG.ALL_COORDS_FILE}")
        if not CFG.PKL_DATA_DIR.exists() or not list(CFG.PKL_DATA_DIR.glob('*.pkl')):
             raise FileNotFoundError(f"Dummy PKL files not found in: {CFG.PKL_DATA_DIR}")

        print("Test environment setup complete.")

    @classmethod
    def tearDownClass(cls):
        """Clean up test output directory and restore CFG paths ONCE."""
        print("\nTearing down test environment...")
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR)
            print(f"Removed test output directory: {TEST_OUTPUT_DIR}")

        # Restore original CFG paths
        for key, value in cls.original_paths.items():
             setattr(CFG, key, value)
        print("CFG paths restored.")
        print("Test environment teardown complete.")

    # --- Individual Test Cases ---

    def test_01_data_loading(self):
        """Test loading of sensor metadata and coordinates."""
        print("\nRunning test_01_data_loading...")
        sensordata, gdf_wgs84 = load_sensor_data(str(CFG.EXCEL_FILE))
        self.assertIsInstance(sensordata, pd.DataFrame)
        self.assertFalse(sensordata.empty)
        self.assertIn('x', sensordata.columns)
        self.assertIn('y', sensordata.columns)
        self.assertIn('Gauge_ID', sensordata.columns) # Assumes dummy has Name

        target_coords, locations_utm = load_target_coordinates()
        self.assertIsInstance(target_coords, list)
        self.assertGreater(len(target_coords), 0)
        self.assertIsInstance(locations_utm, dict)
        self.assertEqual(len(target_coords), len(locations_utm))
        print("test_01_data_loading PASSED")

    def test_02_time_series_loading(self):
        """Test loading pickled time series data."""
        print("\nRunning test_02_time_series_loading...")
        target_coords, locations_utm = load_target_coordinates()
        all_data, valid_locs, missing = load_time_series_data(target_coords, locations_utm)
        self.assertIsInstance(all_data, dict)
        self.assertEqual(len(missing), 0, f"Should find all dummy pkl files, missing: {missing}")
        self.assertEqual(len(all_data), len(target_coords))
        self.assertGreater(len(valid_locs), 0)

        # Check one loaded DataFrame
        first_coord = target_coords[0]
        self.assertIn(first_coord, all_data)
        df = all_data[first_coord]
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertIsNotNone(df.index.tz, "Timestamp index should be timezone-aware (UTC)")
        self.assertIn('Gauge_Data_mm_per_min', df.columns, "Check dummy PKL column names")
        self.assertIn('Radar_Data_mm_per_min', df.columns, "Check dummy PKL column names")
        print("test_02_time_series_loading PASSED")

    def test_03_feature_engineering(self):
        """Test the create_features function."""
        print("\nRunning test_03_feature_engineering...")
        # Create a simple DataFrame for testing
        dates = pd.date_range('2024-01-01', periods=10, freq='10min', tz='UTC')
        data = {
            'Gauge_Data_mm_per_min': np.linspace(0, 2, 10),
            'Radar_Data_mm_per_min': np.linspace(0.1, 2.5, 10)
        }
        df_in = pd.DataFrame(data, index=dates)
        df_out = create_features(df_in)

        self.assertIn('Difference', df_out.columns)
        self.assertIn('Alpha', df_out.columns)
        self.assertIn('Rolling_Gauge', df_out.columns)
        self.assertIn('Rolling_Radar', df_out.columns)
        # Check if NaNs from rolling are handled (simple check)
        self.assertFalse(df_out['Alpha'].isnull().all(), "Alpha column has values")
        # More specific value checks depend on exact rolling/K_PARAM logic
        # Example: check first non-NaN alpha value if calculation is simple
        # self.assertAlmostEqual(df_out['Alpha'].dropna().iloc[0], expected_value)
        print("test_03_feature_engineering PASSED")

    def test_04_anomaly_flagging(self):
        """Test the flag_anomalies function."""
        print("\nRunning test_04_anomaly_flagging...")
        dates = pd.date_range('2024-01-01', periods=5, freq='10min', tz='UTC')
        # Create data designed to trigger flags based on CFG thresholds
        diff_thresh = CFG.ABS_DIFF_THRESHOLD_MM_MIN
        ratio_thresh = CFG.RATIO_THRESHOLD
        data = {
            'Adjusted_Diff_from_network': [0.1, diff_thresh * 2, 0, diff_thresh * -2, 0.1],
            'Adjusted_Ratio_From_Network': [1.0, 1 + ratio_thresh*2, 0.5, 1 - ratio_thresh*2, 0.9]
             # Add other required cols if function needs them (even if dummy)
            ,'Gauge_Data_mm_per_min': [1]*5, 'Radar_Data_mm_per_min': [1]*5
        }
        df_in = pd.DataFrame(data, index=dates)
        all_data_in = {'coord1': df_in}
        all_data_out = flag_anomalies(all_data_in)
        df_out = all_data_out['coord1']

        self.assertIn('Prelim_Flag', df_out.columns)
        self.assertIn('Flagged', df_out.columns)
        expected_flags = [False, True, False, True, False] # Based on diff AND ratio logic
        pd.testing.assert_series_equal(
            df_out['Flagged'],
            pd.Series(expected_flags, index=dates, name='Flagged'),
            check_dtype=False # Allow bool comparison
        )
        print("test_04_anomaly_flagging PASSED")

    def test_05_regional_adjustment(self):
        """Test the compute_regional_adjustment function."""
        print("\nRunning test_05_regional_adjustment...")
        # Create sample data spanning one batch
        batch_start = pd.Timestamp('2024-01-01 00:00:00', tz='UTC')
        batch_end = batch_start + CFG.BATCH_DURATION
        index1 = pd.date_range(batch_start, periods=5, freq='H', tz='UTC')
        df1_data = {'Gauge_Data_mm_per_min': [1, 1, 1, 1, 1], 'Radar_Data_mm_per_min': [2, 2, 2, 2, 2]} # SumG=5, SumR=10 -> f_reg=0.5
        df2_data = {'Gauge_Data_mm_per_min': [1, 1, 1, 1, 1], 'Radar_Data_mm_per_min': [1, 1, 1, 1, 1]} # SumG=5, SumR=5 -> f_reg=1.0
        df3_data = {'Gauge_Data_mm_per_min': [0, 0, 0, 0, 0], 'Radar_Data_mm_per_min': [0, 0, 0, 0, 0]} # SumG=0, SumR=0 -> f_reg=1.0
        df4_data = {'Gauge_Data_mm_per_min': [5, 5, 5, 5, 5], 'Radar_Data_mm_per_min': [1, 1, 1, 1, 1]} # SumG=25, SumR=5 -> f_reg=5.0

        all_data = {
            's1': pd.DataFrame(df1_data, index=index1),
            's2': pd.DataFrame(df2_data, index=index1),
            's3': pd.DataFrame(df3_data, index=index1),
            's4': pd.DataFrame(df4_data, index=index1)
        }

        # Case 1: All sensors reliable (s1, s2, s3, s4)
        batch_flags_1 = {'s1': False, 's2': False, 's3': False, 's4': False}
        # Total G = 5+5+0+25 = 35; Total R = 10+5+0+5 = 20; f_reg = 35/20 = 1.75
        f_reg_1, count_1 = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_1)
        self.assertAlmostEqual(f_reg_1, 1.75)
        self.assertEqual(count_1, 4)

        # Case 2: s1, s4 flagged (unreliable) -> use s2, s3
        batch_flags_2 = {'s1': True, 's2': False, 's3': False, 's4': True}
        # Total G = 5+0 = 5; Total R = 5+0 = 5; f_reg_raw = 5/5 = 1.0
        # Weighting: count=2 -> f_reg = 1.0 * (2/3)
        f_reg_2, count_2 = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_2)
        self.assertAlmostEqual(f_reg_2, 1.0 * (2.0 / 3.0))
        self.assertEqual(count_2, 2)

        # Case 3: Only s1 reliable
        batch_flags_3 = {'s1': False, 's2': True, 's3': True, 's4': True}
        # Total G = 5; Total R = 10; f_reg_raw = 5/10 = 0.5
        # Weighting: count=1 -> f_reg = 0.5 * (1/3)
        f_reg_3, count_3 = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_3)
        self.assertAlmostEqual(f_reg_3, 0.5 * (1.0 / 3.0))
        self.assertEqual(count_3, 1)

        # Case 4: All unreliable
        batch_flags_4 = {'s1': True, 's2': True, 's3': True, 's4': True}
        # Total G = 0; Total R = 0; f_reg = 1.0 (default)
        # Weighting: count=0 -> No weighting applied to the default 1.0
        f_reg_4, count_4 = compute_regional_adjustment(all_data, batch_start, batch_end, batch_flags_4)
        self.assertAlmostEqual(f_reg_4, 1.0)
        self.assertEqual(count_4, 0)

        # Case 5: Zero radar sum (s2 reliable)
        all_data_zero_r = {'s2': pd.DataFrame({'Gauge_Data_mm_per_min': [1]*5, 'Radar_Data_mm_per_min': [0]*5}, index=index1)}
        batch_flags_5 = {'s2': False}
        # Total G = 5; Total R = 0; f_reg = 1.0 (default)
        # Weighting: count=1 -> f_reg = 1.0 * (1/3)
        f_reg_5, count_5 = compute_regional_adjustment(all_data_zero_r, batch_start, batch_end, batch_flags_5)
        self.assertAlmostEqual(f_reg_5, 1.0 * (1.0 / 3.0))
        self.assertEqual(count_5, 1)
        print("test_05_regional_adjustment PASSED")

    # --- End-to-End Test (Example - Requires Careful Data Setup) ---
    # @unittest.skip("End-to-end test requires specific dummy data and is complex to validate fully.")
    def test_99_end_to_end_basic_run(self):
        """ Test if the main function runs without errors and produces output files."""
        print("\nRunning test_99_end_to_end_basic_run...")
        # --- This requires main_batch_adjustment to be importable ---
        # --- and potentially modified to not exit or to return status ---
        try:
            # --- OPTION 1: Run the imported function ---
            # Need to ensure it uses the overridden CFG paths
            # Note: This will run the FULL process on the DUMMY data
            from main_v3_batch3 import main_batch_adjustment
            main_batch_adjustment()

            # --- OPTION 2: Use subprocess (safer isolation) ---
            # import subprocess
            # result = subprocess.run(['python', 'main_v3_batch3.py'], capture_output=True, text=True)
            # self.assertEqual(result.returncode, 0, f"main_v3_batch3.py failed: {result.stderr}")

            # --- Verification ---
            # 1. Check if output CSVs exist
            target_coords, _ = load_target_coordinates() # Use test coords
            output_csv_dir = CFG.DETAILED_CSV_DIR
            found_csvs = 0
            for coord in target_coords:
                 safe_coord = str(coord).replace('(','').replace(')','').replace(',','_').replace(' ','')
                 expected_csv = output_csv_dir / f"{safe_coord}_data_adjustments.csv"
                 if expected_csv.exists():
                     found_csvs += 1
                     # Optional: Load and check basic properties
                     df_out = pd.read_csv(expected_csv, index_col=0, parse_dates=True)
                     self.assertIsInstance(df_out, pd.DataFrame)
                     self.assertIn('Final_Adjusted_Rainfall', df_out.columns)
                 # else: print(f"Warning: Output CSV not found: {expected_csv}") # Debugging
            self.assertGreater(found_csvs, 0, "No output CSVs were generated")
            self.assertEqual(found_csvs, len(target_coords), "Not all output CSVs were generated")

            # 2. Check if dashboards exist
            self.assertTrue((CFG.DASHBOARD_DIR / 'rainfall_anomaly_dashboard.html').exists())
            # self.assertTrue((CFG.DASHBOARD_DIR / 'flagging_dashboard.html').exists()) # If generated

            # 3. Check if plot files exist
            plot_files = list(CFG.ADJUSTMENT_PLOTS_DIR.glob('*.html'))
            self.assertEqual(len(plot_files), len(target_coords), "Number of adjustment plots doesn't match number of sensors")
            # flagging_plot_files = list(CFG.FLAGGING_PLOTS_DIR.glob('*_flagging_metrics.html'))
            # self.assertEqual(len(flagging_plot_files), len(target_coords), "Number of flagging plots doesn't match number of sensors")

            print("test_99_end_to_end_basic_run PASSED (basic checks)")

        except Exception as e:
            self.fail(f"main_batch_adjustment raised an exception: {e}")


# --- Run the tests ---
if __name__ == '__main__':
    # Ensures that the tests are run only when the script is executed directly
    unittest.main()