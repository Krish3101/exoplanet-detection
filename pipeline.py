"""
pipeline.py
-----------
Core TESS exoplanet-detection logic ported from the
`tess_exoplanet_pipeline.ipynb` (pipeline v4/v5).

This module is self-contained so it can be imported by the Streamlit app
without needing to run the whole notebook. It reproduces, function-for-
function, the signal processing / feature extraction / scoring logic used to
train `tess_exoplanet_model_v4.joblib`, so that predictions made here are
consistent with the original pipeline.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.timeseries import BoxLeastSquares
from astropy.stats import sigma_clip

warnings.filterwarnings("ignore")

MAX_BLS_POINTS = 8_000

# ──────────────────────────────────────────────────────────────────────────
# Model artifact classes — MUST be importable as `__main__.SoftVotingEnsemble`
# / `__main__.CalibratedEnsemble` for joblib to unpickle the saved model,
# because that's the module they were pickled under when the notebook was
# run. `load_model()` below takes care of registering these classes into
# `__main__` before calling joblib.load.
# ──────────────────────────────────────────────────────────────────────────
class SoftVotingEnsemble:
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights or [1.0] * len(estimators)

    def predict_proba(self, X):
        probas = np.array([e.predict_proba(X) for e in self.estimators])
        w = np.array(self.weights)[:, None, None]
        return (probas * w).sum(0) / w.sum()


class CalibratedEnsemble:
    def __init__(self, ensemble, calibrator):
        self.ensemble = ensemble
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.ensemble.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


def load_model(model_path: str | Path):
    """Load the joblib artifact, registering the custom classes in
    __main__ first so pickle can resolve them."""
    import sys
    import joblib

    main_mod = sys.modules["__main__"]
    main_mod.SoftVotingEnsemble = SoftVotingEnsemble
    main_mod.CalibratedEnsemble = CalibratedEnsemble

    artifact = joblib.load(model_path)
    return artifact


# ──────────────────────────────────────────────────────────────────────────
# File I/O — FITS / CSV light curve readers
# ──────────────────────────────────────────────────────────────────────────
TIME_COLUMNS = ["time", "btjd", "bjd", "jd", "mjd", "cadenceno", "cadence"]
FLUX_COLUMNS = [
    "pdcsap_flux", "sap_flux", "kspsap_flux", "flux",
    "flux_norm", "normalized_flux", "relative_flux", "pdc_flux",
]


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_tic(val) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    text = re.sub(r"^tic\s*", "", str(val).strip(), flags=re.IGNORECASE)
    text = text.split(".")[0]
    m = re.search(r"\d+", text)
    return str(int(m.group(0))) if m else None


def tic_from_text(text: str) -> str | None:
    m = re.search(r"(?:tic|TIC)[_\-\s]*(\d+)", str(text))
    if m:
        return normalize_tic(m.group(1))
    m = re.search(r"-0*(\d{5,16})-", str(text))
    return normalize_tic(m.group(1)) if m else None


def read_fits_lc(path: Path) -> dict | None:
    """Read a TESS FITS light curve file (SPOC-style)."""
    path = Path(path)
    try:
        with fits.open(path, memmap=False) as hdul:
            tic_id = None
            for hdu in hdul:
                for key in ("TICID", "TIC_ID", "TIC", "TARGETID", "OBJECT"):
                    if key in hdu.header:
                        tic_id = normalize_tic(hdu.header[key])
                        break
                if tic_id:
                    break
            if tic_id is None:
                tic_id = tic_from_text(path.name)
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                names = getattr(data, "names", None)
                if not names:
                    continue
                cols = {normalize_col(n): n for n in names}
                tc = next((cols[c] for c in TIME_COLUMNS if c in cols), None)
                fc = next((cols[c] for c in FLUX_COLUMNS if c in cols), None)
                if tc is None or fc is None:
                    continue
                time = np.asarray(data[tc], dtype=float)
                flux = np.asarray(data[fc], dtype=float)
                qc = cols.get("quality")
                if qc:
                    q = np.asarray(data[qc])
                    gq = (q == 0) & np.isfinite(time) & np.isfinite(flux)
                    if gq.sum() >= 50:
                        time, flux = time[gq], flux[gq]
                # Filter non-finite and sort chronologically
                mask = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[mask], flux[mask]
                if len(time) < 50:
                    continue
                order = np.argsort(time)
                time, flux = time[order], flux[order]
                return {"tic_id": tic_id, "file_name": path.name,
                        "path": str(path), "time": time, "flux": flux}
    except Exception as exc:
        print(f"FITS read failed for {path.name}: {exc}")
    return None


def read_csv_lc(path: Path, max_rows: int = 250_000) -> dict | None:
    """Read a CSV/TSV light curve with time+flux columns."""
    path = Path(path)
    sep = "\t" if path.name.lower().endswith(".tsv") else None
    try:
        sample = pd.read_csv(path, sep=sep, engine="python", nrows=5)
    except Exception:
        return None
    cols = {normalize_col(c): c for c in sample.columns}
    tc = next((cols[c] for c in TIME_COLUMNS if c in cols), None)
    fc = next((cols[c] for c in FLUX_COLUMNS if c in cols), None)
    if tc is None or fc is None:
        return None
    tic_col = next((cols[c] for c in ("tic_id", "ticid", "tic", "targetid") if c in cols), None)
    try:
        usecols = list(dict.fromkeys([tc, fc] + ([tic_col] if tic_col else [])))
        df = pd.read_csv(path, sep=sep, engine="python", usecols=usecols, nrows=max_rows)
    except Exception as exc:
        print(f"CSV read failed for {path.name}: {exc}")
        return None
    tic_id = tic_from_text(path.name)
    if tic_id is None and tic_col and tic_col in df.columns and len(df):
        tic_id = normalize_tic(df[tic_col].iloc[0])
    raw_time = df[tc].to_numpy(dtype=float)
    raw_flux = df[fc].to_numpy(dtype=float)
    mask = np.isfinite(raw_time) & np.isfinite(raw_flux)
    time, flux = raw_time[mask], raw_flux[mask]
    if len(time) < 50:
        return None
    order = np.argsort(time)
    return {"tic_id": tic_id, "file_name": path.name, "path": str(path),
            "time": time[order],
            "flux": flux[order]}


# ──────────────────────────────────────────────────────────────────────────
# Signal processing
# ──────────────────────────────────────────────────────────────────────────
def finite_sort(time, flux):
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    o = np.argsort(t)
    return t[o], f[o]


def mad(x):
    x = np.asarray(x, dtype=float)
    return np.nanmedian(np.abs(x - np.nanmedian(x)))


def flatten(time, flux):
    """Adaptive detrending (0.5-0.75 day window Savitzky-Golay)."""
    time, flux = finite_sort(time, flux)
    if len(time) < 50:
        return time, flux
    med = np.nanmedian(flux)
    if not np.isfinite(med) or abs(med) < 1e-12:
        med = 1.0
    flux = flux / med
    clipped = sigma_clip(flux, sigma=5, maxiters=3)
    flux = np.where(clipped.mask, np.nanmedian(flux), flux)
    n = len(flux)
    cad = np.nanmedian(np.diff(time)) if n > 2 else 1.0 / 720.0
    if cad <= 0 or not np.isfinite(cad):
        cad = 1.0 / 720.0
    ptp = float(np.nanpercentile(flux, 95) - np.nanpercentile(flux, 5))
    win_days = 0.5 if ptp > 0.005 else 0.75
    win = int(win_days / cad)
    win = max(5, win | 1)
    win = min(win, n - (1 if n % 2 == 0 else 0))
    poly = min(3, win - 1)
    try:
        trend = savgol_filter(flux, window_length=win, polyorder=poly, mode="mirror")
    except Exception:
        trend = (pd.Series(flux)
                 .rolling(max(5, win // 2), center=True, min_periods=3)
                 .median().bfill().ffill().to_numpy(copy=True))
    valid_trend = trend[np.isfinite(trend)]
    fallback_val = float(np.nanmedian(valid_trend)) if len(valid_trend) > 0 else 1.0
    if not np.isfinite(fallback_val) or abs(fallback_val) < 1e-12:
        fallback_val = 1.0
    trend = np.where(np.isfinite(trend) & (np.abs(trend) >= 1e-12), trend, fallback_val)
    flat = flux / trend
    lo, hi = np.nanpercentile(flat, [0.1, 99.9])
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        flat = np.clip(flat, lo, hi)
    return time, flat


def bls_fit(time, flux) -> dict:
    default = {k: np.nan for k in
               ("bls_period", "bls_power", "bls_depth", "bls_duration",
                "bls_transit_snr", "bls_in_transit_fraction", "bls_epoch",
                "bls_transit_count", "bls_snr_sigma")}
    if len(time) < 80:
        return default
    span = float(np.nanmax(time) - np.nanmin(time))
    if span < 2 or not np.isfinite(span):
        return default
    if len(time) > MAX_BLS_POINTS:
        idx = np.linspace(0, len(time) - 1, MAX_BLS_POINTS).astype(int)
        time, flux = time[idx], flux[idx]
    t0 = time - np.nanmin(time)
    y = flux - np.nanmedian(flux)
    min_p = 0.3
    max_p = max(min_p + 0.1, min(15.0, span / 2.0))
    if max_p <= min_p:
        return default
    periods = np.concatenate([
        np.linspace(0.3, 1.0, 80),
        np.linspace(1.0, 5.0, 120),
        np.linspace(5.0, 15.0, 80),
    ])
    periods = np.unique(periods)
    periods = periods[(periods >= min_p) & (periods <= max_p)]
    if len(periods) == 0:
        return default
    durations = np.array([0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.15, 0.2])
    durations = durations[durations < min_p / 2]
    if len(durations) == 0:
        durations = np.array([0.02, 0.03, 0.05])
    try:
        bls_obj = BoxLeastSquares(t0, y)
        res = bls_obj.power(periods, durations, objective="snr")
        if not np.any(np.isfinite(res.power)):
            return default
        best = int(np.nanargmax(res.power))
        P = float(res.period[best])
        dur = float(res.duration[best])
        t_epoch = float(res.transit_time[best])
        power = float(res.power[best])
        raw_depth = getattr(res, "depth", np.array([np.nan]))
        depth = float(raw_depth[best]) if len(raw_depth) > best else np.nan
        phase = ((t0 - t_epoch + 0.5 * P) % P) / P - 0.5
        in_mask = np.abs(phase) < max(dur / P / 2.0, 1e-4)
        out_mask = ~in_mask
        if in_mask.sum() < 3 or out_mask.sum() < 3:
            snr = np.nan
        else:
            noise = mad(y[out_mask]) * 1.4826 + 1e-12
            snr = float((np.nanmedian(y[out_mask]) - np.nanmedian(y[in_mask]))
                        / noise * math.sqrt(in_mask.sum()))
        n_transits = max(1, int(span / P))
        return {
            "bls_period": P, "bls_power": power, "bls_depth": depth,
            "bls_duration": dur, "bls_transit_snr": snr,
            "bls_in_transit_fraction": float(np.mean(in_mask)),
            "bls_epoch": t_epoch, "bls_transit_count": float(n_transits),
            "bls_snr_sigma": snr,
        }
    except Exception:
        return default


def secondary_eclipse_depth(t, f, P, epoch) -> float:
    if any(v is None or np.isnan(v) for v in (P, epoch)) or P <= 0:
        return np.nan
    phase = ((t - epoch + 0.5 * P) % P) / P
    sec = np.abs(phase - 0.5) < 0.05
    if sec.sum() < 3 or (~sec).sum() < 3:
        return np.nan
    return float(np.nanmedian(f[~sec]) - np.nanmedian(f[sec]))


def odd_even_ratio(t, f, P, epoch, dur) -> float:
    if any(v is None or np.isnan(v) for v in (P, epoch, dur)) or P <= 0 or dur <= 0:
        return np.nan
    ph = (t - epoch) % P
    odd = (ph < dur / 2) | (ph > P - dur / 2)
    even = np.abs(ph - P / 2) < dur / 2
    if odd.sum() < 3 or even.sum() < 3:
        return np.nan
    od = float(np.nanmedian(f[~odd]) - np.nanmedian(f[odd]))
    ed = float(np.nanmedian(f[~even]) - np.nanmedian(f[even]))
    return float(abs(od - ed) / (abs(ed) + 1e-12)) if abs(ed) > 1e-9 else np.nan


def vshape_metric(t, f, P, epoch, dur) -> float:
    if any(v is None or np.isnan(v) for v in (P, epoch, dur)) or P <= 0 or dur <= 0:
        return np.nan
    phase = ((t - epoch + 0.5 * P) % P) / P - 0.5
    half = dur / P / 2
    in_tr = np.abs(phase) < half
    if in_tr.sum() < 6:
        return np.nan
    ph_in = phase[in_tr]
    fl_in = f[in_tr]
    sort_idx = np.argsort(np.abs(ph_in))
    fl_sorted = fl_in[sort_idx]
    n = len(fl_sorted)
    centre_flux = np.nanmean(fl_sorted[:max(1, n // 3)])
    edge_flux = np.nanmean(fl_sorted[max(1, 2 * n // 3):])
    flat_flux = np.nanmedian(f[~in_tr]) if (~in_tr).sum() > 3 else 1.0
    dip_total = float(flat_flux - centre_flux)
    dip_edge = float(flat_flux - edge_flux)
    if abs(dip_total) < 1e-9:
        return np.nan
    ratio = dip_edge / (dip_total + 1e-12)
    # Metric represents V-shape prominence: <=0.7 U-shape (flat bottom), >0.7 V-shape (grazing EB)
    metric = float(np.clip(1.0 - ratio, 0.0, 1.5))
    return metric


def phase_folded_snr(t, f, P, epoch, dur) -> float:
    if any(v is None or np.isnan(v) for v in (P, epoch, dur)) or P <= 0:
        return np.nan
    phase = ((t - epoch + 0.5 * P) % P) / P - 0.5
    half = max(dur / P / 2, 1e-4)
    in_m = np.abs(phase) < half
    out_m = (np.abs(phase) > half) & (np.abs(phase) < 0.4)
    if in_m.sum() < 3 or out_m.sum() < 3:
        return np.nan
    noise = mad(f[out_m]) * 1.4826 + 1e-12
    signal = float(np.nanmedian(f[out_m]) - np.nanmedian(f[in_m]))
    return float(signal / noise * math.sqrt(in_m.sum()))


def transit_depth_ppm(bls_depth: float) -> float:
    return float(bls_depth * 1e6) if (bls_depth is not None and np.isfinite(bls_depth)) else np.nan


def sec_to_primary_ratio(sec_dep: float, pri_dep: float) -> float:
    if sec_dep is None or pri_dep is None or not np.isfinite(sec_dep) or not np.isfinite(pri_dep) or abs(pri_dep) < 1e-9:
        return np.nan
    return float(sec_dep / (abs(pri_dep) + 1e-12))


def extract_features(time, flux) -> dict | None:
    """Full 47-dim signal-derived feature vector (matches training pipeline).
    The remaining 13 model features are optional TIC catalog metadata
    (ra, dec, tmag, teff, ...) which are simply left NaN/imputed here."""
    raw_t, raw_f = finite_sort(time, flux)
    if len(raw_f) > 0:
        raw_f_norm = raw_f / (np.nanmedian(raw_f) + 1e-12)
        stellar_var_amp = float(np.nanpercentile(raw_f_norm, 95)
                                 - np.nanpercentile(raw_f_norm, 5))
    else:
        stellar_var_amp = np.nan

    time, flat = flatten(time, flux)
    if len(time) < 50:
        return None
    c = flat - np.nanmedian(flat)
    q = np.nanpercentile(flat, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    m = mad(c)
    dip = c < (-3 * m if m > 0 else np.nanpercentile(c, 5))

    def _autocorr(lag):
        if len(c) <= lag + 2:
            return np.nan
        x, y2 = c[:-lag], c[lag:]
        if np.nanstd(x) < 1e-12 or np.nanstd(y2) < 1e-12:
            return np.nan
        return float(np.corrcoef(x, y2)[0, 1])

    def _longest_run(mask):
        best = cur = 0
        for v in mask:
            cur = (cur + 1) if v else 0
            best = max(best, cur)
        return best

    feats = {
        "n_points": float(len(flat)),
        "time_span": float(np.nanmax(time) - np.nanmin(time)),
        "cadence_median": float(np.nanmedian(np.diff(time))) if len(time) > 2 else np.nan,
        "flux_mean": float(np.nanmean(flat)),
        "flux_std": float(np.nanstd(flat)),
        "flux_mad": float(m),
        "flux_skew": float(stats.skew(c, nan_policy="omit")),
        "flux_kurtosis": float(stats.kurtosis(c, nan_policy="omit")),
        "stellar_var_amp": stellar_var_amp,
        **{f"flux_q{p:02d}": float(q[i])
           for i, p in enumerate((1, 5, 10, 25, 50, 75, 90, 95, 99))},
        "dip_depth_q01": float(1.0 - q[0]),
        "dip_depth_q05": float(1.0 - q[1]),
        "bright_spike_q99": float(q[8] - 1.0),
        "deep_dip_fraction": float(np.nanmean(dip)),
        "deep_dip_count": float(np.nansum(dip)),
        "longest_dip_run": float(_longest_run(dip)),
        "autocorr_1": _autocorr(1),
        "autocorr_5": _autocorr(5),
        "autocorr_20": _autocorr(20),
    }

    bls = bls_fit(time, flat)
    feats.update(bls)

    P, ep, dur, dep = (bls["bls_period"], bls["bls_epoch"],
                       bls["bls_duration"], bls["bls_depth"])

    sec_dep = secondary_eclipse_depth(time, flat, P, ep)
    oe_rat = odd_even_ratio(time, flat, P, ep, dur)

    feats["secondary_eclipse_depth"] = sec_dep
    feats["odd_even_depth_ratio"] = oe_rat
    feats["sec_to_primary_ratio"] = sec_to_primary_ratio(sec_dep, dep)
    feats["vshape_metric"] = vshape_metric(time, flat, P, ep, dur)
    feats["phase_folded_snr"] = phase_folded_snr(time, flat, P, ep, dur)
    feats["transit_depth_ppm"] = transit_depth_ppm(dep)
    feats["log_transit_depth_ppm"] = (float(np.log10(dep * 1e6 + 1))
                                       if np.isfinite(dep) and dep > 0 else np.nan)
    feats["transit_duration_ratio"] = (
        float(dur / P) if not np.isnan(dur) and not np.isnan(P) and P > 0 else np.nan
    )
    feats["depth_snr_ratio"] = (
        float(dep / (feats["flux_std"] + 1e-12)) if not np.isnan(dep) else np.nan
    )
    if not np.isnan(P) and not np.isnan(ep) and P > 0 and len(time) > 10:
        ph = ((time - ep) % P) / P
        im = ph < 0.1
        feats["bls_epoch_consistency"] = float(np.nanstd(ph[im])) if im.sum() >= 3 else np.nan
    else:
        feats["bls_epoch_consistency"] = np.nan
    feats["blend_contamination_proxy"] = (
        float(dep / (m + 1e-12)) if not np.isnan(dep) and m > 0 else np.nan
    )

    return feats


# ──────────────────────────────────────────────────────────────────────────
# Transit-parameter refinement (trapezoid fit)
# ──────────────────────────────────────────────────────────────────────────
def trapezoid_model(t, t0, period, duration, depth, baseline=1.0):
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    half = duration / (2 * period)
    ingress = 0.1 * half
    model = np.ones_like(t) * baseline
    flat_bottom = np.abs(phase) < (half - ingress)
    ingress_mask = (np.abs(phase) >= (half - ingress)) & (np.abs(phase) < half)
    model[flat_bottom] -= depth
    model[ingress_mask] -= depth * (1 - (np.abs(phase[ingress_mask]) - (half - ingress)) / ingress)
    return model


def estimate_transit_params(time, flux, bls: dict) -> dict:
    P, ep, dur, dep = (bls.get("bls_period"), bls.get("bls_epoch"),
                       bls.get("bls_duration"), bls.get("bls_depth"))
    t, f = flatten(time, flux)

    result = {
        "period_days": P,
        "epoch_btjd": ep,
        "duration_hours": dur * 24 if dur and not np.isnan(dur) else np.nan,
        "depth_ppm": dep * 1e6 if dep and not np.isnan(dep) else np.nan,
        "depth_uncertainty_ppm": np.nan,
        "fit_method": "BLS",
    }

    if all(v is not None and not np.isnan(v) for v in (P, ep, dur, dep)) and P > 0:
        try:
            p0 = [ep % P, P, dur, dep]
            bounds = ([0, P * 0.9, dur * 0.5, dep * 0.2],
                      [P, P * 1.1, dur * 2.0, dep * 5.0])
            popt, pcov = curve_fit(
                lambda t_, t0_, P_, dur_, dep_: trapezoid_model(t_, t0_, P_, dur_, dep_),
                t, f, p0=p0, bounds=bounds, maxfev=4000,
            )
            perr = np.sqrt(np.diag(pcov))
            result.update({
                "period_days": float(popt[1]),
                "epoch_btjd": float(popt[0]),
                "duration_hours": float(popt[2]) * 24,
                "depth_ppm": float(popt[3]) * 1e6,
                "depth_uncertainty_ppm": float(perr[3]) * 1e6,
                "fit_method": "Trapezoid",
            })
        except Exception:
            pass
    return result


# ──────────────────────────────────────────────────────────────────────────
# Confidence & classification
# ──────────────────────────────────────────────────────────────────────────
def compute_confidence(score: float, bls_snr: float, phase_snr: float = 0.0) -> dict:
    snr = bls_snr if (bls_snr and np.isfinite(bls_snr)) else 0.0
    psnr = phase_snr if (phase_snr and np.isfinite(phase_snr)) else 0.0
    snr_norm = min(snr, 20.0) / 20.0
    psnr_norm = min(psnr, 15.0) / 15.0
    composite = 0.50 * float(score) + 0.30 * snr_norm + 0.20 * psnr_norm

    if composite >= 0.70 and snr >= 7.0:
        level = "High"
    elif composite >= 0.50 or snr >= 5.0:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "composite_confidence": round(composite, 4),
        "bls_snr": round(snr, 2) if np.isfinite(snr) else 0.0,
        "confidence_level": level,
    }


def categorise(score: float, feats: dict, threshold: float) -> str:
    """Depth-aware signal classification (v5 logic)."""
    power = feats.get("bls_power", 0) or 0
    dep_ppm = feats.get("transit_depth_ppm", 0) or 0
    vsm = feats.get("vshape_metric", 0) or 0
    sec_rat = feats.get("sec_to_primary_ratio", 0) or 0
    bl = feats.get("blend_contamination_proxy", 0) or 0

    power = power if np.isfinite(power) else 0.0
    dep_ppm = dep_ppm if np.isfinite(dep_ppm) else 0.0
    vsm = vsm if np.isfinite(vsm) else 0.0
    sec_rat = sec_rat if np.isfinite(sec_rat) else 0.0
    bl = bl if np.isfinite(bl) else 0.0

    no_signal = power < 0.05
    is_deep_eclipse = dep_ppm > 30_000
    is_secondary_eb = sec_rat > 0.5
    is_vshape_eb = (vsm > 0.7) and (power > 0.05)
    is_planet_depth = 0 < dep_ppm < 30_000
    is_ushape = vsm <= 0.7

    if no_signal:
        return "Other"
    if is_deep_eclipse or is_secondary_eb or is_vshape_eb:
        return "EclipsingBinary"
    if score >= threshold and power >= 0.05 and is_planet_depth and is_ushape and sec_rat < 0.5:
        return "Transit"
    if score < 0.4 and power < 0.15 and bl > 3:
        return "Blend"
    return "Other"


# ──────────────────────────────────────────────────────────────────────────
# Synthetic demo light curves (used only when the user has no data on hand)
# ──────────────────────────────────────────────────────────────────────────
def simulate_lc(has_planet: bool, n: int = 1500, seed: int | None = None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 27, n))
    trend = 1.0 + 0.0015 * np.sin(2 * np.pi * t / rng.uniform(8, 18))
    noise = rng.normal(0, rng.uniform(0.0008, 0.003), n)
    f = trend + noise
    meta = {}
    if has_planet:
        P = rng.uniform(1.2, 9.0)
        dur = rng.uniform(0.05, 0.22)
        dep = rng.uniform(0.004, 0.025)
        ep = rng.uniform(0, P)
        ph = ((t - ep + 0.5 * P) % P) - 0.5 * P
        tr = np.abs(ph) < dur / 2
        ing = np.abs(np.abs(ph) - dur / 2) < dur / 5
        f[tr] -= dep
        f[ing] += dep * 0.25 * rng.random(ing.sum())
        meta = {"period": P, "duration": dur, "depth": dep, "epoch": ep, "kind": "transit"}
    else:
        kind = rng.choice(["quiet", "eb", "spot"])
        if kind == "eb":
            P = rng.uniform(0.5, 5.0)
            dep = rng.uniform(0.02, 0.15)
            ep = rng.uniform(0, P)
            ph = ((t - ep + 0.5 * P) % P) - 0.5 * P
            f[np.abs(ph) < 0.05 * P] -= dep
            ph2 = ((t - ep - P / 2 + 0.5 * P) % P) - 0.5 * P
            f[np.abs(ph2) < 0.04 * P] -= dep * rng.uniform(0.3, 0.9)
        elif kind == "spot":
            f += rng.uniform(0, 0.01) * np.sin(2 * np.pi * t / rng.uniform(3, 15))
        meta = {"kind": str(kind)}
    return t, f, meta
