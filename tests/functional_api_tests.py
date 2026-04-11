import os
import unittest

import gpcell as gc


class TestFunctionalApiUsage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        data = gc.load_detector_data(
            path="data/hes/Hes1_example.csv",
            x_name="Time (h)",
            background_name="Background",
            y_name="Cell",
            set_noise=0.1,
        )
        cls.small_data = gc.DetectorData(X=data.X[:3], Y=data.Y[:3])

    def test_prepare_detector_data_with_set_noise(self):
        prepared = gc.prepare_detector_data(
            self.small_data,
            gc.DetectorConfig(set_noise=0.1, detrend_enabled=False),
        )

        self.assertEqual(len(prepared.X), 3)
        self.assertEqual(len(prepared.Y), 3)
        self.assertEqual(prepared.noise_list, [0.1, 0.1, 0.1])
        self.assertIsNone(prepared.mean_noise)
        self.assertIsNone(prepared.bckgd_GPs)
        self.assertIsNone(prepared.detrend_GPs)

    def test_prepare_detector_data_requires_background_without_set_noise(self):
        with self.assertRaises(ValueError):
            gc.prepare_detector_data(self.small_data, gc.DetectorConfig())

    def test_run_pipeline_bic_returns_typed_result(self):
        result = gc.run_pipeline(
            self.small_data,
            gc.DetectorConfig(
                set_noise=0.1,
                detrend_enabled=False,
                bic_replicates=1,
            ),
            methods=["BIC"],
        )

        self.assertIsNotNone(result.bic)
        self.assertIsNone(result.bootstrap)
        self.assertEqual(len(result.prepared.Y), 3)
        self.assertEqual(len(result.bic.LLRs), 3)
        self.assertEqual(len(result.bic.BIC_diffs), 3)
        self.assertEqual(len(result.bic.periods), 3)


if __name__ == "__main__":
    unittest.main()
