# import testing utilities
import unittest

# import third-party helpers
import numpy as np

# import gpcell for testing
import gpcell as gc


class TestOscillatorDetectorMethods(unittest.TestCase):
    def setUp(self):
        """
        Set up a minimal environment for the OscillatorDetector class.
        """
        # Create sample test data
        self.sample_data_path = "data/hes/Hes1_example.csv"
        self.X_name = "Time (h)"
        self.background_name = "Background"
        self.Y_name = "Cell"

        # Create OscillatorDetector instance
        self.detector = gc.OscillatorDetector.from_data(
            path=self.sample_data_path,
            X_name=self.X_name,
            background_name=self.background_name,
            Y_name=self.Y_name,
        )

    def test_initialization(self):
        """
        Test the initialization of OscillatorDetector and loading of data.
        """
        self.assertEqual(len(self.detector.X), 12)
        self.assertEqual(len(self.detector.Y), 12)
        self.assertEqual(len(self.detector.bckgd), 4)

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
        params = {"plots": ["invalid_plot"]}
        with self.assertRaises(ValueError) as context:
            gc.OscillatorDetector.from_data(
                path=self.sample_data_path,
                X_name=self.X_name,
                background_name=self.background_name,
                Y_name=self.Y_name,
                params=params,
            )
        self.assertIn("Invalid plot type(s) selected", str(context.exception))


class TestOscillatorDetectorHes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        path = "data/hes/Hes1_example.csv"
        X_name = "Time (h)"
        background_name = "Background"
        Y_name = "Cell"
        params = {"verbose": True}

        # Create OscillatorDetector instance with Hes dataset
        cls.detector = gc.OscillatorDetector.from_data(
            path=path,
            X_name=X_name,
            background_name=background_name,
            Y_name=Y_name,
            params=params,
        )

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
        self.detector.fit(methods="BIC")
        self.assertEqual(sum(np.array(self.detector.BIC_diffs) > 3.0), 10)

    def test_bootstrap_classification(self):
        """
        Test the number of cells classified as oscillatory based on synthetic-cell bootstrap.
        """
        self.detector.fit(methods="bootstrap")

        for i, (true, pred) in enumerate(
            zip([False, True, True, False], self.detector.osc_filt[-4:])
        ):
            self.assertEqual(true, pred, f"Cell {i} classification incorrect.")


# for joblib multiprocessing
class TestOscillatorDetectorJoblib(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        path = "data/hes/Hes1_example.csv"
        X_name = "Time (h)"
        background_name = "Background"
        Y_name = "Cell"
        params = {"verbose": True, "joblib": True}

        # Create OscillatorDetector instance with Hes dataset
        cls.detector = gc.OscillatorDetector.from_data(
            path=path,
            X_name=X_name,
            background_name=background_name,
            Y_name=Y_name,
            params=params,
        )

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
        self.detector.fit(methods="BIC")
        self.assertEqual(sum(np.array(self.detector.BIC_diffs) > 3.0), 10)

    def test_bootstrap_classification(self):
        """
        Test the number of cells classified as oscillatory based on synthetic-cell bootstrap.
        """
        self.detector.fit(methods="bootstrap")

        for i, (true, pred) in enumerate(
            zip([False, True, True, False], self.detector.osc_filt[-4:])
        ):
            self.assertEqual(true, pred, f"Cell {i} classification incorrect.")


if __name__ == "__main__":
    unittest.main()
