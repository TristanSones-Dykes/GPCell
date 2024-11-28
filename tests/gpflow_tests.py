# import testing utilities
import unittest
from unittest.mock import patch

# import third-party helpers
import numpy as np

# import pyrocell for testing
import gpcell as gc


class TestOscillatorDetectorMethods(unittest.TestCase):
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
        self.detector = gc.OscillatorDetector(
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


class TestOscillatorDetectorHes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        path = "data/hes/Hes1_example.csv"
        X_name = "Time (h)"
        background_name = "Background"
        Y_name = "Cell"

        # Create OscillatorDetector instance with Hes dataset
        cls.detector = gc.OscillatorDetector(
            path=path,
            X_name=X_name,
            background_name=background_name,
            Y_name=Y_name,
        )
        cls.detector.run(verbose=True)

    def test_background_noise_value(self):
        """
        Test the background noise value calculation with real data.
        """
        self.assertIsNotNone(self.detector.mean_noise)
        self.assertGreater(self.detector.mean_noise, 0)
        self.assertAlmostEqual(float(self.detector.mean_noise), 7.239964458255131)

    def test_BIC_classification(self):
        """
        Test the number of cells classified as oscillatory based on BIC.
        """
        self.assertEqual(sum(np.array(self.detector.BIC_diffs) > 3.0), 10)

    def test_bootstrap_classification(self):
        """
        Test the number of cells classified as oscillatory based on synthetic-cell bootstrap.
        """
        self.assertEqual(sum(self.detector.osc_filt), 10)


if __name__ == "__main__":
    unittest.main()
