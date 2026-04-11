import os
import unittest

import numpy as np

import gpcell as gc
from gpcell import utils


class TestDataCharacterization(unittest.TestCase):
    def test_load_data_shapes_and_alignment(self):
        x_data, y_data = utils.load_data("data/hes/Hes1_example.csv", "Time (h)", "Cell")
        self.assertEqual(len(x_data), len(y_data))
        self.assertEqual(len(x_data), 12)
        for x, y in zip(x_data, y_data):
            self.assertEqual(x.shape[0], y.shape[0])
            self.assertEqual(x.shape[1], 1)
            self.assertEqual(y.shape[1], 1)
            self.assertGreater(x.shape[0], 0)

    def test_pad_traces_retains_original_prefix(self):
        traces = [
            np.array([1.0, 2.0, 3.0]),
            np.array([[4.0], [5.0]]),
        ]
        padded, lengths = utils.pad_traces(traces)
        self.assertEqual(lengths, [3, 2])
        self.assertEqual(padded.shape, (2, 3, 1))
        np.testing.assert_allclose(padded[0, :3, 0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(padded[1, :2, 0], np.array([4.0, 5.0]))


class TestDetectorCharacterization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        cls.detector = gc.OscillatorDetector.from_file(
            path="data/hes/Hes1_example.csv",
            X_name="Time (h)",
            background_name="Background",
            Y_name="Cell",
            params={"verbose": False},
        )

    def test_bic_fit_output_lengths(self):
        self.detector.fit(methods="BIC")
        self.assertEqual(len(self.detector.LLRs), self.detector.N)
        self.assertEqual(len(self.detector.BIC_diffs), self.detector.N)
        self.assertEqual(len(self.detector.periods), self.detector.N)

    def test_bic_hes_classification_count(self):
        self.detector.fit(methods="BIC")
        self.assertEqual(int(np.sum(np.array(self.detector.BIC_diffs) > 3.0)), 10)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.fit(methods="does_not_exist")


if __name__ == "__main__":
    unittest.main()
