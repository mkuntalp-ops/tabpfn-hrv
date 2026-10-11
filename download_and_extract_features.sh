#!/usr/bin/env bash
# Download PhysioNet apnea datasets and extract epoch-level HRV features.
#
# Usage: bash download_and_extract_features.sh
#
# Outputs (consumed by reproduce_all_results.sh):
#   data/processed/hrv_features_apnea_ecg_full.parquet
#   data/processed/hrv_features_slpdb.parquet
#   data/processed/hrv_features_ucddb.parquet
set -euo pipefail

RAW=data/raw
OUT=data/processed
mkdir -p "$RAW" "$OUT"

echo "=== [1/3] Apnea-ECG (Marburg) ==="
mkdir -p "$RAW/apnea-ecg"
cd "$RAW/apnea-ecg"
for f in a01 a02 a03 a04 a05 a06 a07 a08 a09 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 \
         b01 b02 b03 b04 b05 c01 c02 c03 c04 c05 c06 c07 c08 c09 c10; do
  for ext in dat hea apn; do
    [ -f "$f.$ext" ] || wget -q "https://physionet.org/files/apnea-ecg/1.0.0/$f.$ext" -O "$f.$ext"
  done
done
cd - >/dev/null
python3 src/features/hrv_extraction.py --dataset apnea-ecg --data-dir "$RAW/apnea-ecg" \
  --out "$OUT/hrv_features_apnea_ecg_full.parquet"

echo "=== [2/3] SLPDB (MIT-BIH Polysomnographic) ==="
mkdir -p "$RAW/slpdb"
cd "$RAW/slpdb"
for f in slp01a slp01b slp02a slp02b slp03 slp04 slp05 slp06 slp07 slp08 \
         slp09 slp10 slp11 slp12 slp13 slp14 slp15 slp16; do
  for ext in dat hea st; do
    [ -f "$f.$ext" ] || wget -q "https://physionet.org/files/slpdb/1.0.0/$f.$ext" -O "$f.$ext"
  done
done
cd - >/dev/null
python3 src/features/hrv_extraction.py --dataset slpdb --data-dir "$RAW/slpdb" \
  --out "$OUT/hrv_features_slpdb.parquet"

echo "=== [3/3] UCDDB (Dublin) ==="
mkdir -p "$RAW/ucddb"
cd "$RAW/ucddb"
for f in $(seq -w 2 28); do
  rec="ucddb0$f"
  for ext in dat hea; do
    [ -f "$rec.$ext" ] || wget -q "https://physionet.org/files/ucddb/1.0.0/$rec.$ext" -O "$rec.$ext"
  done
  # annotation/event files
  wget -q "https://physionet.org/files/ucddb/1.0.0/${rec}_res.txt" -O "${rec}_res.txt" 2>/dev/null || true
done
cd - >/dev/null
python3 src/features/hrv_extraction.py --dataset ucddb --data-dir "$RAW/ucddb" \
  --out "$OUT/hrv_features_ucddb.parquet"

echo ""
echo "Feature extraction complete:"
ls -lh "$OUT"
