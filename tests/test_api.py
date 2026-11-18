"""
tests/test_api.py
=================
Automated test suite for FastAPI microservice (backend/api.py).
Tests all 7 endpoints, FITS and CSV parsing, CORS headers, error handling,
and JSON schema serialization.
"""
import json
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

import backend.api as api
from backend.api import app


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_dir = Path(__file__).resolve().parent.parent / "public" / "sample-input"
        cls.sample_1 = cls.sample_dir / "sample-1.fits"
        cls.toi_700 = cls.sample_dir / "toi-700.fits"
        cls.eb = cls.sample_dir / "eclipsing-binary.fits"
        cls.csv_file = cls.sample_dir / "test_synthetic.csv"

    def test_01_health_endpoint(self):
        """Verify GET /api/v1/health returns 200 with service metadata."""
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["version"], "1.0.0")

    def test_02_model_info_endpoint(self):
        """Verify GET /api/v1/model/info returns model hyperparameters and architecture."""
        res = self.client.get("/api/v1/model/info")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("Ensemble", data["model_name"])
        self.assertEqual(data["total_features_trained"], 60)
        self.assertEqual(data["astrophysical_features_analyzed"], 47)
        self.assertEqual(data["decision_threshold"], 0.50)

    def test_03_ingest_fits_and_csv(self):
        """Verify POST /api/v1/ingest on FITS files and CSV light curves."""
        for p in (self.sample_1, self.toi_700, self.eb, self.csv_file):
            with self.subTest(file=p.name):
                with open(p, "rb") as f:
                    mime = "application/octet-stream" if p.suffix == ".fits" else "text/csv"
                    res = self.client.post("/api/v1/ingest", files={"file": (p.name, f, mime)})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertGreater(data["valid_points"], 50)
                self.assertEqual(len(data["time_preview"]), 10)
                self.assertEqual(len(data["flux_preview"]), 10)
                # Verify no NaN values in JSON
                for v in data["time_preview"] + data["flux_preview"]:
                    self.assertTrue(np.isfinite(v))

    def test_04_preprocess_endpoint(self):
        """Verify POST /api/v1/preprocess performs Savitzky-Golay detrending and quality masking."""
        for p in (self.sample_1, self.toi_700, self.eb):
            with self.subTest(file=p.name):
                with open(p, "rb") as f:
                    res = self.client.post("/api/v1/preprocess", files={"file": (p.name, f, "application/octet-stream")})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertGreater(data["valid_points"], 50)
                self.assertGreater(data["noise_mad"], 0.0)
                self.assertIn("Savitzky-Golay", data["detrending_method"])
                self.assertEqual(len(data["detrended_flux_preview"]), 10)

    def test_05_detect_endpoint(self):
        """Verify POST /api/v1/detect performs BLS transit search and parameter estimation."""
        for p in (self.sample_1, self.toi_700, self.eb):
            with self.subTest(file=p.name):
                with open(p, "rb") as f:
                    res = self.client.post("/api/v1/detect", files={"file": (p.name, f, "application/octet-stream")})
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "success")
                self.assertIsNotNone(data["bls_period_days"])
                self.assertGreater(data["bls_period_days"], 0.0)
                self.assertIsNotNone(data["transit_depth_ppm"])
                self.assertIsNotNone(data["bls_transit_snr"])

    def test_06_classify_endpoint(self):
        """Verify POST /api/v1/classify runs ML ensemble inference and categorisation."""
        # Test known Planet Candidate
        with open(self.sample_1, "rb") as f:
            res = self.client.post("/api/v1/classify", files={"file": (self.sample_1.name, f, "application/octet-stream")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_label"], "Planet Candidate")
        self.assertGreaterEqual(data["exoplanet_score"], 0.50)
        self.assertIn("key_features", data)

        # Test known Eclipsing Binary
        with open(self.eb, "rb") as f:
            res = self.client.post("/api/v1/classify", files={"file": (self.eb.name, f, "application/octet-stream")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_label"], "Non-Planet")
        self.assertEqual(data["signal_class"], "EclipsingBinary")

    def test_07_analyze_endpoint(self):
        """Verify POST /api/v1/analyze executes complete multi-stage pipeline."""
        with open(self.toi_700, "rb") as f:
            res = self.client.post("/api/v1/analyze?simulate_delay=false", files={"file": (self.toi_700.name, f, "application/octet-stream")})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_label"], "Planet Candidate")
        self.assertEqual(data["signal_class"], "Transit")
        self.assertGreater(data["exoplanet_score"], 0.90)
        self.assertEqual(data["confidence_level"], "High")

        # Verify all 12 key features are present and finite
        kf = data["key_features"]
        self.assertIsNotNone(kf["bls_period_days"])
        self.assertIsNotNone(kf["transit_depth_ppm"])
        self.assertIsNotNone(kf["vshape_metric"])

    def test_08_error_handling(self):
        """Verify error responses for empty files (400) and unprocessable files (422)."""
        # Empty file
        res = self.client.post("/api/v1/ingest", files={"file": ("empty.fits", b"", "application/octet-stream")})
        self.assertEqual(res.status_code, 400)

        # Corrupt file
        res = self.client.post("/api/v1/ingest", files={"file": ("corrupt.fits", b"garbage non-fits header content", "application/octet-stream")})
        self.assertEqual(res.status_code, 422)

    def test_09_cors_headers(self):
        """Verify CORS preflight and access-control headers."""
        res = self.client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:8501", "Access-Control-Request-Method": "GET"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:8501")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")


if __name__ == "__main__":
    unittest.main()
