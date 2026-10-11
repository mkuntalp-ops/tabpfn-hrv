"""
Unit tests for src/features/hrv_extraction.py.

Run:  python -m pytest tests/test_feature_extraction.py -q
  or: python tests/test_feature_extraction.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.hrv_extraction import (  # noqa: E402
    clean_rr_intervals,
    detect_r_peaks,
    epoch_features,
    frequency_domain,
    tinn,
)

FS = 100.0


def synthetic_ecg(rr_s=0.8, duration_s=60, fs=FS, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(0, duration_s, 1 / fs)
    ecg = np.zeros_like(t)
    beat_t = 0.5
    while beat_t < duration_s:
        ecg += np.exp(-((t - beat_t) ** 2) / (2 * 0.008 ** 2))
        beat_t += rr_s + (rng.rand() - 0.5) * jitter
    ecg += rng.normal(0, 0.05, len(t))
    return ecg


def test_detect_r_peaks_count():
    ecg = synthetic_ecg()
    peaks = detect_r_peaks(ecg, FS)
    assert 70 <= len(peaks) <= 80, f"expected ~75 R-peaks, got {len(peaks)}"


def test_epoch_features_regular_rhythm():
    feats = epoch_features(synthetic_ecg(), FS)
    assert feats["num_r_peaks"] >= 70
    assert abs(feats["HRV_MeanNN"] - 800) < 40, feats["HRV_MeanNN"]
    for k in ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
              "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_TINN"]:
        assert k in feats, f"missing feature {k}"


def test_epoch_features_flat_signal():
    # flat (zero) signal: no detectable R-peaks -> NaN features, num_r_peaks = 0
    feats = epoch_features(np.zeros(int(60 * FS)), FS)
    assert feats["num_r_peaks"] == 0
    assert np.isnan(feats["HRV_MeanNN"])
    assert np.isnan(feats["HRV_SDNN"])
    assert np.isnan(feats["HRV_LF"]) and np.isnan(feats["HRV_LFHF"])


def test_clean_rr_removes_ectopic():
    rr = np.array([0.8] * 10 + [0.3] + [0.8] * 10)
    assert len(clean_rr_intervals(rr)) == 20


def test_clean_rr_removes_implausible():
    rr = np.array([0.8] * 5 + [2.5] * 5)
    assert len(clean_rr_intervals(rr)) == 5


def test_frequency_domain_lf_dominant():
    rr = 0.8 + 0.05 * np.sin(2 * np.pi * 0.1 * np.arange(200))
    lf, hf, ratio = frequency_domain(rr)
    assert np.isfinite(lf) and np.isfinite(hf)
    assert lf > hf, "0.1 Hz modulation must yield LF > HF"


def test_tinn_sane():
    rng = np.random.RandomState(1)
    assert 0 < tinn(rng.normal(0.8, 0.03, 100)) < 1.5


def test_short_series_returns_nan():
    lf, hf, ratio = frequency_domain(np.array([0.8, 0.9, 0.7]))
    assert all(np.isnan(v) for v in (lf, hf, ratio))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[pass] {fn.__name__}")
    print(f"{len(fns)} tests passed.")
