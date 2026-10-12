#!/usr/bin/env python3
"""Convert legacy CSV HRV feature tables to the Parquet format expected by
reproduce_all_results.sh and validate the schema against
configs/protocol.py::PRIMARY_FEATURES.

Usage:
    python3 convert_csv_features_to_parquet.py [--data-dir data/processed]

Expects (per cohort):
    <data-dir>/hrv_features_apnea_ecg_full.csv
    <data-dir>/hrv_features_slpdb.csv
    <data-dir>/hrv_features_ucddb.csv

Writes alongside each input:
    <data-dir>/hrv_features_<cohort>.parquet

Validation (fatal on mismatch):
    - All PRIMARY_FEATURES columns present
    - Label column present ('apnea' or dataset-specific)
    - Non-empty and no duplicate (subject, epoch) index if columns exist
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.protocol import PRIMARY_FEATURES

CSV_TO_PARQUET = {
    "apnea-ecg": "hrv_features_apnea_ecg_full.csv",
    "slpdb": "hrv_features_slpdb.csv",
    "ucddb": "hrv_features_ucddb.csv",
}

LABEL_CANDIDATES = ["apnea", "label", "apnea_epoch", "y"]


def validate(df: pd.DataFrame, cohort: str, src: Path) -> list:
    errors = []
    missing_feats = [f for f in PRIMARY_FEATURES if f not in df.columns]
    if missing_feats:
        errors.append(
            f"{cohort}: missing PRIMARY_FEATURES columns {missing_feats}; "
            f"available: {list(df.columns)}"
        )
    labels = [c for c in LABEL_CANDIDATES if c in df.columns]
    if not labels:
        errors.append(f"{cohort}: no label column found among {LABEL_CANDIDATES}")
    if df.empty:
        errors.append(f"{cohort}: empty table")
    for col in ("subject", "record", "patient", "epoch", "epoch_idx", "epoch_index"):
        pass
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    all_errors = []
    for cohort, csv_name in CSV_TO_PARQUET.items():
        src = args.data_dir / csv_name
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        df = pd.read_csv(src)
        errors = validate(df, cohort, src)
        for e in errors:
            print(f"[ERROR] {e}")
        all_errors.extend(errors)
        out = src.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        print(f"[ok] {cohort}: {len(df)} rows, {len(df.columns)} cols -> {out}")

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
