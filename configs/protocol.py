"""
Study protocol constants: primary features, seeds, data paths, subject maps.
Referenced by run_03_tri_center_lodo_provenance_fix.py.
"""
from pathlib import Path

RANDOM_SEED = 42

# Strict 9 primary HRV features used in the tri-center LODO provenance run.
# If PRIMARY_FEATURES was changed, update this list to match the run that
# produced results/runs/run_20261006_unified_9feat (see metrics_errata_log.md).
PRIMARY_FEATURES = [
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_RMSSD",
    "HRV_pNN50",
    "HRV_LF",
    "HRV_HF",
    "HRV_LFHF",
    "HRV_TINN",
    "num_r_peaks",
]

# Local TabPFN checkpoint used by run_03 (212,804,803 bytes, v3 family).
# Update this path to your local checkpoint location; it is not redistributed.
TABPFN_CHECKPOINT_PATH = Path("models/tabpfn_v3_checkpoint.ckpt")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATHS = {
    "Apnea-ECG": str(REPO_ROOT / "data/processed/hrv_features_apnea_ecg_full.parquet"),
    "SLPDB": str(REPO_ROOT / "data/processed/hrv_features_slpdb.parquet"),
    "UCDDB": str(REPO_ROOT / "data/processed/hrv_features_ucddb.parquet"),
}

# SLPDB records belong to the same subject when they share the numeric part
# (e.g. slp01a/slp01b -> slp01).
SLPDB_SUBJECT_MAP = {}
for _rec in [
    "slp01a", "slp01b", "slp02a", "slp02b", "slp03", "slp04", "slp05",
    "slp06", "slp07", "slp08", "slp09", "slp10", "slp11", "slp12",
    "slp13", "slp14", "slp15", "slp16",
]:
    SLPDB_SUBJECT_MAP[_rec] = _rec.rstrip("ab")

APNEA_ECG_EXCLUDED_RECORDS = ["c06"]
