"""
tests/test_pipeline.py
----------------------
Automated test suite for pipeline.py signal processing, feature extraction,
NaN handling, transit geometry, and Safe Mode fallback.
"""
from pathlib import Path
import unittest
import numpy as np
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pipeline as pl
import safe_mode
import llm


class TestPipeline(unittest.TestCase):
    def test_finite_sort_clean(self):
        t = [3.0, 1.0, 2.0, np.nan, 5.0]
        f = [1.0, 0.9, np.nan, 1.2, 1.1]
        t_c, f_c = pl.finite_sort(t, f)
        self.assertEqual(len(t_c), 3)
        self.assertEqual(list(t_c), [1.0, 3.0, 5.0])
        self.assertTrue(np.all(np.isfinite(t_c)))
        self.assertTrue(np.all(np.isfinite(f_c)))

    def test_flatten_edge_cases(self):
        # Empty/short
        t = np.linspace(0, 1, 20)
        f = np.ones(20)
        t_out, f_out = pl.flatten(t, f)
        self.assertEqual(len(t_out), 20)

        # Normal curve with NaN
        t = np.linspace(0, 27, 500)
        f = 1.0 + np.random.normal(0, 0.005, 500)
        f[10] = np.nan
        t_out, f_out = pl.flatten(t, f)
        self.assertTrue(np.all(np.isfinite(t_out)))
        self.assertTrue(np.all(np.isfinite(f_out)))

    def test_bls_fit_edge_cases(self):
        # Less than 80 points
        t = np.linspace(0, 1, 50)
        f = np.ones(50)
        res = pl.bls_fit(t, f)
        self.assertTrue(np.isnan(res["bls_period"]))

        # Flat line
        t = np.linspace(0, 20, 200)
        f = np.ones(200)
        res = pl.bls_fit(t, f)
        self.assertIn("bls_period", res)

    def test_vshape_metric(self):
        # U-shaped flat bottom (planet)
        P, epoch, dur = 3.0, 0.5, 0.2
        t = np.linspace(0, 10, 1000)
        f = np.ones(1000)
        phase = ((t - epoch + 0.5 * P) % P) / P - 0.5
        f[np.abs(phase) < dur / P / 2] = 0.98
        metric = pl.vshape_metric(t, f, P, epoch, dur)
        self.assertTrue(np.isfinite(metric))
        self.assertLessEqual(metric, 0.70, f"Expected U-shape metric <= 0.70, got {metric}")

    def test_all_5_benchmark_targets_safe_mode(self):
        samples = ["sample-1.fits", "toi-700.fits", "wasp-18b.fits", "eclipsing-binary.fits", "sample-2.fits"]
        for s in samples:
            res = safe_mode.get_safe_result(s)
            self.assertIsNotNone(res, f"Missing safe result for {s}")
            r, f = res
            self.assertIn("predicted_label", r)
            self.assertIn("signal_class", r)
            self.assertIn("exoplanet_score", r)
            self.assertTrue(np.isfinite(r["exoplanet_score"]))
            self.assertGreaterEqual(len(f), 47)

    def test_llm_offline_synthesis(self):
        sample_res = {
            "predicted_label": "Planet Candidate",
            "signal_class": "Transit",
            "exoplanet_score": 0.942,
            "decision_threshold": 0.50,
            "confidence_level": "High",
            "bls_period_days": 37.425,
            "transit_depth_ppm": 980.5,
            "transit_duration_hours": 2.85,
            "bls_transit_snr": 12.4,
            "vshape_metric": 0.28,
            "secondary_to_primary_ratio": 0.04,
        }
        rep = llm.explain_result(api_key=None, result=sample_res)
        self.assertIn("Astrophysical Screening Dossier", rep)
        self.assertIn("37.425", rep)
        self.assertIn("980.5", rep)

        ans = llm.chat_about_result(api_key=None, result=sample_res, question="What is the period?")
        self.assertIn("37.425", ans)


if __name__ == "__main__":
    unittest.main()
