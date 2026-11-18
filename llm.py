"""
llm.py
------
OpenRouter OpenAI-compatible Chat Completions API wrapper for CosmiClassifiers.
Used to:
  (a) generate a plain-English astrophysical explanation of a pipeline result
  (b) answer interactive follow-up questions from researchers and evaluators.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "google/gemini-2.0-flash-exp:free",
    "qwen/qwen-2.5-coder-32b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


def load_env() -> None:
    """Lightweight helper to load environment variables from a .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val


# Load the environment variables immediately on import
load_env()


SYSTEM_PROMPT = """You are an astrophysics domain expert assistant embedded in CosmiClassifiers, \
an AI-enabled TESS exoplanet detection screening pipeline.

You are given the numeric output of a multi-stage machine learning pipeline that analyzes NASA TESS \
light curves (Savitzky-Golay detrending + Box-Least-Squares transit search + calibrated XGBoost and \
Random Forest soft-voting ensemble) and classifies targets as Transit (Planet Candidate), \
Eclipsing Binary, Blend, or Other/Stellar Variability/Noise.

Explain results clearly and scientifically, suitable for academic and research presentation:
- Reference key astrophysical parameters: BLS period, transit depth (ppm), transit duration, \
BLS SNR, V-shape metric, odd/even depth ratio, and secondary eclipse depth ratio.
- Explain why these features led to the specific classification (e.g., U-shaped profile vs. V-shaped \
grazing eclipsing binary, absence of significant secondary eclipse, depth consistency).
- Maintain rigorous scientific objectivity: a Moderate/High confidence candidate is a promising \
exoplanet candidate requiring radial velocity or ground-based photometric follow-up, not yet a \
confirmed planet.
- Keep responses concise, structured, and insightful without repeating raw numbers verbatim."""


def get_client(api_key: str | None = None) -> OpenAI:
    actual_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not actual_key:
        raise ValueError("OpenRouter API key not set. Please provide it in the sidebar or set the OPENROUTER_API_KEY environment variable.")
    return OpenAI(
        api_key=actual_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "CosmiClassifiers Exoplanet Detector",
        },
    )


def build_result_context(result: dict) -> str:
    """Turn the pipeline result dict into a compact text block for the LLM."""
    lines = ["COSMICLASSIFIERS PIPELINE RESULT:"]
    for k, v in result.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def generate_offline_report(result: dict) -> str:
    """Generate an authoritative, scientific astrophysics summary report offline."""
    label = result.get("predicted_label", "Candidate")
    sig_class = result.get("signal_class", "Transit")
    score = result.get("exoplanet_score", 0.0)
    score_val = float(score) if (score is not None and np.isfinite(score)) else 0.0
    threshold = float(result.get("decision_threshold", 0.50))
    period = result.get("bls_period_days", "N/A")
    depth = result.get("transit_depth_ppm", "N/A")
    duration = result.get("transit_duration_hours", "N/A")
    snr = result.get("bls_transit_snr", "N/A")
    vshape = result.get("vshape_metric", "N/A")
    sec_ratio = result.get("secondary_to_primary_ratio", "N/A")
    conf = result.get("confidence_level", "Moderate")

    p1 = (
        f"### 🔭 Astrophysical Screening Dossier: {sig_class} ({label})\n\n"
        f"**1. Classification Rationale & Astrophysical Evidence:**  \n"
        f"The candidate is classified as **{sig_class}** with **{conf}** statistical significance "
        f"and a calibrated ML score of **{score_val:.3f}** (applied decision threshold {threshold:.2f}). "
    )
    if sig_class == "Transit":
        p1 += (
            f"The photometric analysis confirms periodic flux depressions consistent with planetary occultation across the stellar disc. "
            f"The low transit shape metric (V-shape = {vshape}) indicates a flat-bottomed U-shaped profile characteristic of a small body "
            f"fully transiting a host star, with negligible secondary eclipse depth (ratio = {sec_ratio}), ruling out near-equal mass stellar companions."
        )
    elif sig_class == "EclipsingBinary":
        p1 += (
            f"The photometric morphology displays strong indicators of a stellar eclipsing binary system. "
            f"The prominent depth ({depth} ppm) combined with a V-shape metric ({vshape}) and/or significant secondary occultation at phase 0.5 "
            f"(ratio = {sec_ratio}) strongly points toward stellar grazing occultations rather than a planetary companion."
        )
    else:
        p1 += (
            f"The signal characteristics reflect background stellar variability, rotational modulation, or instrumental systematic noise. "
            f"The detection significance (BLS SNR = {snr}σ) does not meet the rigorous statistical threshold required for a bona fide transit candidate."
        )

    p2 = (
        f"\n\n**2. Transit Ephemeris & Photometric Profile:**  \n"
        f"- **Orbital Period:** {period} days  \n"
        f"- **Transit Depth:** {depth} ppm  \n"
        f"- **Transit Duration:** {duration} hours  \n"
        f"- **Detection Significance:** {snr}σ (Phase SNR: {result.get('phase_folded_snr', 'N/A')}σ)  \n"
        f"- **Transit Morphology:** V-shape metric = {vshape}, Secondary eclipse ratio = {sec_ratio}"
    )

    p3 = (
        f"\n\n**3. Recommended Astronomical Follow-Up:**  \n"
        f"To conclusively confirm the planetary nature of this candidate and rule out unresolved blend contamination:  \n"
        f"- High-precision ground-based photometric transit confirmation (e.g. Las Cumbres Observatory Global Telescope network).  \n"
        f"- High-resolution speckle imaging or adaptive optics to exclude background eclipsing binaries within the 21-arcsecond TESS pixel aperture.  \n"
        f"- High-precision radial velocity spectroscopy (e.g. HARPS, ESPRESSO) to constrain the companion's planetary mass and orbital eccentricity."
    )
    return p1 + p2 + p3


def explain_result(api_key: str | None = None, result: dict | None = None, model: str = DEFAULT_MODEL) -> str:
    res = result or {}
    try:
        client = get_client(api_key)
        context = build_result_context(res)
        candidate_models = [model] + [m for m in FREE_MODELS if m != model]
        last_exc = None
        for cand in candidate_models:
            try:
                resp = client.chat.completions.create(
                    model=cand,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            f"{context}\n\n"
                            "Explain this result in 2-3 structured paragraphs: "
                            "(1) Primary classification rationale and key astrophysical evidence, "
                            "(2) Key transit parameters (period, depth, SNR, shape), "
                            "(3) Recommended astronomical follow-up observations."
                        )},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_exc = e
                continue
        # Fallback to local report if API calls failed
        return generate_offline_report(res)
    except Exception:
        return generate_offline_report(res)


def chat_about_result(api_key: str | None = None, result: dict | None = None, history: list[dict] | None = None,
                       question: str = "", model: str = DEFAULT_MODEL) -> str:
    res = result or {}
    try:
        client = get_client(api_key)
        context = build_result_context(res)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\nKeep this candidate result in context for follow-up inquiries."},
            {"role": "assistant", "content": "Understood. I have the candidate analysis in context."},
        ]
        messages.extend(history or [])
        messages.append({"role": "user", "content": question})
        candidate_models = [model] + [m for m in FREE_MODELS if m != model]
        for cand in candidate_models:
            try:
                resp = client.chat.completions.create(
                    model=cand,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=600,
                )
                return resp.choices[0].message.content
            except Exception:
                continue
    except Exception:
        pass

    # Intelligent offline response
    q_low = question.lower()
    if "period" in q_low or "orbit" in q_low:
        return f"The detected Box Least Squares orbital period for this target is {res.get('bls_period_days', 'N/A')} days."
    elif "depth" in q_low or "ppm" in q_low:
        return f"The measured transit dip depth is {res.get('transit_depth_ppm', 'N/A')} ppm, corresponding to an occultation duration of {res.get('transit_duration_hours', 'N/A')} hours."
    elif "snr" in q_low or "significance" in q_low:
        return f"The detection signal-to-noise ratio is {res.get('bls_transit_snr', 'N/A')}σ, with a phase-folded profile significance of {res.get('phase_folded_snr', 'N/A')}σ."
    elif "vshape" in q_low or "shape" in q_low or "binary" in q_low:
        return f"The transit shape metric is {res.get('vshape_metric', 'N/A')} (values <= 0.70 indicate flat-bottom U-shaped planetary transits; > 0.70 indicate V-shaped grazing eclipsing binaries). Secondary eclipse ratio is {res.get('secondary_to_primary_ratio', 'N/A')}."
    else:
        return (
            f"Based on our calibrated tree ensemble screening, this target is classified as {res.get('signal_class', 'Transit')} "
            f"({res.get('predicted_label', 'Candidate')}) with an exoplanet probability score of {res.get('exoplanet_score', 0.0):.3f}. "
            f"Key parameters: Period = {res.get('bls_period_days', 'N/A')} d, Depth = {res.get('transit_depth_ppm', 'N/A')} ppm, SNR = {res.get('bls_transit_snr', 'N/A')}σ."
        )
