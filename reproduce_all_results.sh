#!/usr/bin/env bash
# Regenerates ALL study outputs with the fixed (leakage-free) pipeline.
#
# Prerequisites (see README.md "Data access"):
#   - data/processed/hrv_features_apnea_ecg_full.parquet
#   - data/processed/hrv_features_slpdb.parquet
#   - data/processed/hrv_features_ucddb.parquet
#   - TabPFN checkpoint path set in configs/protocol.py::TABPFN_CHECKPOINT_PATH
#   - pip install -r requirements.txt
#
# Usage:  bash reproduce_all_results.sh
set -euo pipefail

echo "=== [0/4] Pre-flight checks ==="
python3 - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")

missing = [p for p in [
    "data/processed/hrv_features_apnea_ecg_full.parquet",
    "data/processed/hrv_features_slpdb.parquet",
    "data/processed/hrv_features_ucddb.parquet",
] if not Path(p).exists()]
if missing:
    sys.exit("Missing input data (see README 'Data access'):\n  " + "\n  ".join(missing))

from configs.protocol import TABPFN_CHECKPOINT_PATH, PRIMARY_FEATURES
if not Path(TABPFN_CHECKPOINT_PATH).exists():
    sys.exit(f"TabPFN checkpoint not found at: {TABPFN_CHECKPOINT_PATH}\n"
             "Set configs/protocol.py::TABPFN_CHECKPOINT_PATH to your local checkpoint.")
print(f"[ok] inputs present; {len(PRIMARY_FEATURES)} primary features; checkpoint found")
EOF

echo "=== [1/4] Development cohort study (GroupKFold, calibration, DCA, subject-level) ==="
python3 run_full_development_study.py

echo "=== [2/4] Label efficiency benchmark ==="
python3 label_efficiency.py

echo "=== [3/4] Tri-center LODO + pairwise transfer matrix + few-shot ==="
python3 leave_one_database_out_3center.py

echo "=== [4/4] Provenance-fixed tri-center LODO + clustered bootstrap + Holm MC ==="
python3 run_03_tri_center_lodo_provenance_fix.py

echo ""
echo "All outputs regenerated. Key result directories:"
echo "  results/full_development_study/    (development cohort)"
echo "  results/label_efficiency/          (low-N efficiency)"
echo "  results/three_center_lodo/         (LODO + transfer matrix)"
echo "  results/runs/run_20261006_lodo_provenance_fix/  (clustered bootstrap, Holm MC)"
