# 🪐 CosmiClassifiers: AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

> Automated detection, periodic transit search, and tree ensemble screening of spaceborne astronomical light curves from NASA's Transiting Exoplanet Survey Satellite (TESS).

---

## 📌 Project Architecture

```
Raw TESS Light Curve (FITS / CSV)
       │
       ▼
Telemetry Quality Bitmask Filter & Sigma-Clipping
       │
       ▼
Savitzky-Golay Baseline Detrending (Flattening)
       │
       ▼
Box Least Squares (BLS) Frequency Period Search
       │
       ▼
Astrophysical Feature Engineering (12 Interpretable Features / 47 Signal Metrics)
       │
       ▼
Calibrated Tree Ensemble Inference (XGBoost + Random Forest Soft Voting)
       │
       ├──► 🪐 Transit (Planet Candidate)
       ├──► ⭐ Eclipsing Binary (False Positive)
       ├──► 🔄 Blend Contamination
       └──► ❌ Stellar Variability / Photon Noise
```

---

## 🚀 Quick Start (One Command)

To run both the **FastAPI Backend (port 8000)** and the **Streamlit Frontend (port 8501)** together:

```bash
git clone https://github.com/Krish3101/exoplanet-detection.git
cd exoplanet-detection
./run_demo.sh
```

- **Frontend Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Backend REST API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📁 Project Directory Structure

```
exoplanet-demo/
├── app.py                      # Interactive Streamlit 6-module scientific analytics studio
├── pipeline.py                 # Core signal processing & feature extraction routines
├── llm.py                      # OpenRouter astrophysical reasoning assistant
├── run_demo.sh                 # One-command dual launcher (Backend + Frontend)
├── backend/
│   └── api.py                  # Production FastAPI REST microservice (7 endpoints)
├── safe_mode/
│   ├── __init__.py             # Zero-fail fallback controller
│   └── precomputed.json        # Pre-calculated verified results for demo targets
├── model/
│   ├── tess_exoplanet_model_v4.joblib  # Trained ensemble model (XGBoost + RF)
│   └── labels_reference.csv    # Benchmark TESS reference catalog
├── public/
│   └── sample-input/           # Verified NASA TESS FITS sample files
│       ├── sample-1.fits       # High-confidence planet candidate
│       ├── toi-700.fits        # TOI-700 d (Habitable zone Earth-sized planet)
│       ├── wasp-18b.fits       # WASP-18 b (Hot Jupiter with deep transit)
│       ├── eclipsing-binary.fits # Eclipsing binary star (false positive)
│       └── sample-2.fits       # Quiet star / non-transit baseline
└── requirements.txt
```

---

## 🔬 12 Key Astrophysical Features

| Feature | Unit | Astrophysical Purpose |
|---|---|---|
| **BLS Period** | Days | Orbital periodicity detected by Box Least Squares |
| **Transit Depth** | ppm | Radius ratio proxy ($R_p / R_*$) |
| **Transit Duration** | Hours | Orbital chord traversal duration |
| **BLS Transit SNR** | $\sigma$ | Detection significance over background noise |
| **Phase-Folded SNR** | $\sigma$ | Folded profile signal-to-noise ratio |
| **V-Shape Metric** | ratio | Discriminates flat U-shaped planets from V-shaped grazing binaries |
| **Odd-Even Depth Disparity** | ratio | Identifies twin-star eclipsing binary depth discrepancies |
| **Secondary Eclipse Ratio** | ratio | Detects occultation of secondary luminous companion |
| **Stellar Variability Amp** | flux | Measures starspot rotation and stellar activity |
| **Blend Contamination Proxy**| ratio | Gauges susceptibility to background eclipsing binaries |
| **BLS Peak Power** | norm | Spectral peak prominence in periodogram |
| **In-Transit Cadence Fraction**| frac | Geometric transit duty cycle |

