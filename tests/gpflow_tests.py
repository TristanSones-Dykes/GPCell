# import testing utilities
import unittest
from unittest.mock import patch

# import third-party helpers
import numpy as np

# import pyrocell for testing
import pyrocell.gp.gpflow as pc


class TestOscillatorDetector(unittest.TestCase):
    def setUp(self):
        """
        Set up a minimal environment for the OscillatorDetector class.
        """
        # Create sample test data
        self.sample_data_path = "sample_data.csv"
        self.X_name = "Time"
        self.background_name = "Background"
        self.Y_name = "Cell_Traces"

        # Mocking load_data to avoid loading actual files
        patcher = patch("pyrocell.gp.gpflow.load_data")
        self.mock_load_data = patcher.start()
        self.mock_load_data.side_effect = [
            (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3])),  # Background noise data
            (np.array([1, 2, 3]), np.array([0.5, 1.0, 1.5])),  # Cell trace data
        ]
        self.addCleanup(patcher.stop)

        # Create OscillatorDetector instance
        self.detector = pc.OscillatorDetector(
            path=self.sample_data_path,
            X_name=self.X_name,
            background_name=self.background_name,
            Y_name=self.Y_name,
        )

    def test_initialization(self):
        """
        Test the initialization of OscillatorDetector and loading of data.
        """
        self.assertEqual(len(self.detector.X), 3)
        self.assertEqual(len(self.detector.Y), 3)
        self.assertEqual(len(self.detector.bckgd), 3)

    def test_str_method(self):
        """
        Test the __str__ method.
        """
        summary = str(self.detector)
        self.assertIn("Oscillator Detector", summary)
        self.assertIn("background noise models", summary)

    def test_run_with_invalid_plot(self):
        """
        Test the run method with an invalid plot type.
        """
        with self.assertRaises(ValueError) as context:
            self.detector.run(plots=["invalid_plot_type"])
        self.assertIn("Invalid plot type(s) selected", str(context.exception))


class TestOscillatorDetectorWithRealData(unittest.TestCase):
    def setUp(self):
        """
        Set up a real data environment for the OscillatorDetector class.
        """
        # Create sample test data with real path
        self.real_data_path = "data/hes/Hes1_example.csv"
        self.X_name = "Time (h)"
        self.background_name = "Background"
        self.Y_name = "Cell"

        # Create OscillatorDetector instance with real data
        self.detector = pc.OscillatorDetector(
            path=self.real_data_path,
            X_name=self.X_name,
            background_name=self.background_name,
            Y_name=self.Y_name,
        )

    def test_background_noise_value(self):
        """
        Test the background noise value calculation with real data.
        """
        self.detector.run(verbose=True)
        # Check that mean_std is calculated properly (this is a placeholder, should be adjusted based on real expectations)
        self.assertIsNotNone(self.detector.mean_std)
        self.assertGreater(self.detector.mean_std, 0)
        self.assertAlmostEqual(float(self.detector.mean_std), 7.239964458255131)


if __name__ == "__main__":
    unittest.main()
