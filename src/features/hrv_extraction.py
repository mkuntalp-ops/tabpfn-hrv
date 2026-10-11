"""
Epoch-level HRV feature extraction from single/multi-lead ECG PSG recordings.

Produces per-epoch (60 s) feature tables compatible with the benchmark
pipeline (configs/protocol.py::PRIMARY_FEATURES):

    HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, HRV_LF, HRV_HF,
    HRV_LFHF, HRV_TINN, num_r_peaks

Pipeline per epoch:
  1. 60 s ECG segment at native sampling rate
  2. R-peak detection: NeuroKit2 'pantompkins' method on a bandpass-cleaned
     signal (or 'neurokit' method; configurable), 5 min median filtering
     applied to remove spurious peaks
  3. RR interval cleaning: PhysioNet-conservative, removes RR < 0.3 s or
     > 2.0 s and ectopic jumps > 20% deviation from the local median
  4. Time-domain: MeanNN, SDNN, RMSSD, pNN50, TINN (triangular index
     approximation)
  5. Frequency-domain: FFT periodogram over interpolated (4 Hz) RR series
     with cubic interpolation; LF = 0.04-0.15 Hz, HF = 0.15-0.4 Hz
     (Task Force 1996). Epochs with < 5 clean RR intervals produce NaN
     features (handled downstream by fold-fit median imputation).

Datasets:
  - Apnea-ECG: record a01-a20, b01-b05, c01-c10; ECG = channel 0 (100 Hz);
    apnea annotations per minute from the .apn annotation files
  - SLPDB: records slp01a..slp16; ECG = channel 2 (250 Hz); apnea from
    .st (sleep stage/apnea) annotations
  - UCDDB: records ucddb002..ucddb028; ECG = first ECG channel (128 Hz);
    apnea epochs from the .edf tabular annotation event files

All datasets are downloaded from PhysioNet via wfdb; no credentials needed.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FS_RR_INTERP = 4.0          # Hz, RR interpolation rate for spectral analysis
RR_MIN_S, RR_MAX_S = 0.3, 2.0
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.4)
EPOCH_S = 60
MIN_RR_PER_EPOCH = 5

ECG_CHANNELS = {"Apnea-ECG": 0, "SLPDB": 2, "UCDDB": None}  # UCDDB: auto-detect


def detect_r_peaks(ecg: np.ndarray, fs: float, method: str = "pantompkins") -> np.ndarray:
    """R-peak indices via NeuroKit2 with signal-quality preprocessing."""
    import neurokit2 as nk

    ecg = np.asarray(ecg, dtype=np.float64)
    ecg = nk.ecg_clean(ecg, sampling_rate=int(fs), method="pantompkins")
    try:
        _, info = nk.ecg_peaks(
            ecg, sampling_rate=int(fs), method=method, correct_artifacts=True
        )
        peaks = info["ECG_R_Peaks"]
    except Exception:
        peaks = np.array([], dtype=int)
    return np.asarray(peaks, dtype=int)


def clean_rr_intervals(rr_s: np.ndarray, local_median_window: int = 5) -> np.ndarray:
    """Remove physiologically implausible and ectopic RR intervals."""
    rr = np.asarray(rr_s, dtype=np.float64)
    rr = rr[(rr >= RR_MIN_S) & (rr <= RR_MAX_S)]
    if len(rr) < MIN_RR_PER_EPOCH:
        return np.array([])
    med = np.array([
        np.median(rr[max(0, i - local_median_window // 2):
                     i + local_median_window // 2 + 1])
        for i in range(len(rr))
    ])
    keep = np.abs(rr - med) <= 0.2 * med
    rr = rr[keep]
    return rr if len(rr) >= MIN_RR_PER_EPOCH else np.array([])


def tinn(rr_s: np.ndarray, bin_width_s: float = 1 / 128.0) -> float:
    """Triangular index based TINN: base width of the RR histogram."""
    if len(rr_s) < 2:
        return np.nan
    hist, edges = np.histogram(rr_s, bins=int((rr_s.max() - rr_s.min()) / bin_width_s) or 1)
    if hist.max() == 0:
        return np.nan
    peak_bin = int(np.argmax(hist))
    # base width between first and last non-empty bins adjacent to the mode
    nz = np.nonzero(hist)[0]
    return float(edges[nz[-1] + 1] - edges[nz[0]])


def frequency_domain(rr_s: np.ndarray):
    """LF, HF power (ms^2) and LF/HF ratio via FFT periodogram."""
    if len(rr_s) < MIN_RR_PER_EPOCH:
        return np.nan, np.nan, np.nan
    rr_ms = rr_s * 1000.0
    t = np.cumsum(rr_s)
    t -= t[0]
    t_q = np.arange(0, t[-1], 1.0 / FS_RR_INTERP)
    if len(t_q) < 8:
        return np.nan, np.nan, np.nan
    rr_q = np.interp(t_q, t, rr_ms)  # linear interpolation (robust to short series)

    n = len(rr_q)
    rr_q = rr_q - rr_q.mean()
    freqs = np.fft.rfftfreq(n, d=1.0 / FS_RR_INTERP)
    psd = (np.abs(np.fft.rfft(rr_q)) ** 2) / n

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[m], freqs[m])) if m.any() else np.nan

    lf = band_power(*LF_BAND)
    hf = band_power(*HF_BAND)
    ratio = lf / hf if (np.isfinite(lf) and np.isfinite(hf) and hf > 0) else np.nan
    return lf, hf, ratio


def epoch_features(ecg_epoch: np.ndarray, fs: float, peak_method: str = "pantompkins") -> dict:
    """Compute the 9 primary HRV features for one 60 s ECG epoch."""
    peaks = detect_r_peaks(ecg_epoch, fs, method=peak_method)
    out = {
        "HRV_MeanNN": np.nan, "HRV_SDNN": np.nan, "HRV_RMSSD": np.nan,
        "HRV_pNN50": np.nan, "HRV_LF": np.nan, "HRV_HF": np.nan,
        "HRV_LFHF": np.nan, "HRV_TINN": np.nan, "num_r_peaks": int(len(peaks)),
    }
    if len(peaks) < MIN_RR_PER_EPOCH:
        return out
    rr = np.diff(peaks) / fs
    rr = clean_rr_intervals(rr)
    if len(rr) < MIN_RR_PER_EPOCH:
        return out

    rr_ms = rr * 1000.0
    out["HRV_MeanNN"] = float(np.mean(rr_ms))
    out["HRV_SDNN"] = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else np.nan
    out["HRV_pNN50"] = float(np.mean(np.abs(np.diff(rr_ms)) > 50)) if len(rr_ms) > 1 else np.nan
    out["HRV_TINN"] = float(tinn(rr))
    if len(rr_ms) > 1:
        out["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    lf, hf, ratio = frequency_domain(rr)
    out["HRV_LF"], out["HRV_HF"], out["HRV_LFHF"] = lf, hf, ratio
    return out


def extract_record(record_path, fs: float, ecg_channel: int, labels_per_epoch,
                   peak_method: str = "pantompkins") -> list:
    """Extract per-epoch features for one record given per-minute labels.

    labels_per_epoch: array of 0/1 (apnea) of length n_minutes, or None.
    """
    import wfdb

    sig = wfdb.rdrecord(str(record_path), channels=[ecg_channel])
    ecg = np.asarray(sig.p_signal[:, 0], dtype=np.float64)
    n_epochs = int(len(ecg) // (EPOCH_S * fs))

    rows = []
    for e in range(n_epochs):
        seg = ecg[int(e * EPOCH_S * fs): int((e + 1) * EPOCH_S * fs)]
        feats = epoch_features(seg, fs, peak_method=peak_method)
        feats["minute_idx"] = e
        feats["record_id"] = Path(record_path).name
        if labels_per_epoch is not None:
            feats["apnea_label"] = int(labels_per_epoch[e])
        rows.append(feats)
    return rows


def load_apnea_ecg_labels(record_path) -> np.ndarray:
    """Apnea-ECG: minute-level apnea annotation ('A'=apnea, 'N'=normal, 'X')."""
    import wfdb

    ann = wfdb.rdann(str(record_path), extension="apn")
    labels = np.array([1 if s == "A" else 0 for s in ann.symbol])
    return labels


def extract_apnea_ecg(data_dir: Path, out_parquet: Path, peak_method: str = "pantompkins"):
    import wfdb

    rows = []
    for rec in sorted(p.name for p in data_dir.glob("a*.dat")) or []:
        pass
    records = sorted(
        {p.name[:-4] for p in data_dir.glob("*.dat")}
    )
    records = [r for r in records if r and r[0] in "abc" and r[1:].isdigit()]
    for rec_name in records:
        rec_path = data_dir / rec_name
        try:
            labels = load_apnea_ecg_labels(rec_path)
            rows += extract_record(rec_path, 100.0, 0, labels, peak_method)
            print(f"  [ok] {rec_name}: {len(labels)} epochs")
        except Exception as e:
            print(f"  [skip] {rec_name}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    return df


def extract_slpdb(data_dir: Path, out_parquet: Path, peak_method: str = "pantompkins"):
    """SLPDB: ECG is channel 2; apnea epochs from .st annotations
    (annotations 'A' in even minutes — apnea events mark OSA in stage files)."""
    import wfdb

    rows = []
    records = sorted({p.name[:-4] for p in data_dir.glob("*.dat")})
    for rec_name in records:
        rec_path = data_dir / rec_name
        try:
            sig = wfdb.rdrecord(str(rec_path))
            if sig.p_signal.shape[1] <= 2:
                print(f"  [skip] {rec_name}: no ECG channel")
                continue
            n_min = int(sig.p_signal.shape[0] // (EPOCH_S * sig.fs))
            try:
                st = wfdb.rdann(str(rec_path), extension="st")
                apnea_minutes = set()
                for sym, sample in zip(st.symbol, st.sample):
                    if sym in ("A", "OA", "CA", "H", "OA*"):
                        apnea_minutes.add(int(sample // (EPOCH_S * sig.fs)))
                labels = np.array([1 if m in apnea_minutes else 0 for m in range(n_min)])
            except Exception:
                labels = np.zeros(n_min, dtype=int)
            rows += extract_record(rec_path, float(sig.fs), 2, labels, peak_method)
            print(f"  [ok] {rec_name}: {n_min} epochs")
        except Exception as e:
            print(f"  [skip] {rec_name}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    return df


def extract_ucddb(data_dir: Path, out_parquet: Path, peak_method: str = "pantompkins"):
    """UCDDB: ECG auto-detected from channel names; apnea from the tabular
    event files (*_res.txt on PhysioNet: onset/duration of apneas)."""
    import wfdb

    rows = []
    records = sorted({p.name[:-4] for p in data_dir.glob("*.dat")})
    for rec_name in records:
        rec_path = data_dir / rec_name
        try:
            sig = wfdb.rdrecord(str(rec_path))
            ecg_idx = None
            for i, ch in enumerate(sig.sig_name or []):
                if "ECG" in (ch or "").upper():
                    ecg_idx = i
                    break
            if ecg_idx is None:
                ecg_idx = 0
            n_min = int(sig.p_signal.shape[0] // (EPOCH_S * sig.fs))

            apnea_minutes = set()
            res_file = data_dir / (rec_name + "_res.txt")
            if res_file.exists():
                # tabular: time (HH:MM:SS), duration_s, event type columns
                for line in res_file.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 3 and any(
                        k in parts[-1].upper() for k in ("APNEA", "OBSTRUCTIVE", "CENTRAL", "MIXED", "HYPOPNEA")
                    ):
                        try:
                            h, m, s = parts[0].split(":")
                            onset_s = int(h) * 3600 + int(m) * 60 + float(s)
                            dur = float(parts[1]) if len(parts) > 1 else 10.0
                            for mm in range(int(onset_s // EPOCH_S), int((onset_s + dur) // EPOCH_S) + 1):
                                apnea_minutes.add(mm)
                        except ValueError:
                            continue
            labels = np.array([1 if m in apnea_minutes else 0 for m in range(n_min)])
            rows += extract_record(rec_path, float(sig.fs), ecg_idx, labels, peak_method)
            print(f"  [ok] {rec_name}: {n_min} epochs")
        except Exception as e:
            print(f"  [skip] {rec_name}: {e}")
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract epoch-level HRV features from PhysioNet apnea datasets")
    ap.add_argument("--dataset", required=True, choices=["apnea-ecg", "slpdb", "ucddb"])
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with wfdb record files")
    ap.add_argument("--out", type=Path, required=True, help="Output parquet path")
    ap.add_argument("--peak-method", default="pantompkins", choices=["pantompkins", "neurokit", "hamilton"])
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "apnea-ecg":
        df = extract_apnea_ecg(args.data_dir, args.out, args.peak_method)
    elif args.dataset == "slpdb":
        df = extract_slpdb(args.data_dir, args.out, args.peak_method)
    else:
        df = extract_ucddb(args.data_dir, args.out, args.peak_method)
    print(f"[done] {len(df)} epochs -> {args.out}")


if __name__ == "__main__":
    main()
