# TabPFN-HRV-Sleep-Apnea

HRV-based sleep apnea classification using TabPFN under low-sample and cross-database settings.

## Overview

This repository contains the code, analysis scripts, configuration files, and derived
summary outputs used in a study of ECG-derived heart rate variability (HRV) and sleep
apnea classification with TabPFN, benchmarked against Logistic Regression, Random
Forest, SVM, LightGBM, and XGBoost.

## Study design

- **Development cohort**: PhysioNet Apnea-ECG (Philipps University Marburg; 100 Hz
  single-lead ECG), 5-fold patient-independent GroupKFold cross-validation
  (`run_full_development_study.py`). Record `c06` is excluded.
- **External cohorts**: MIT-BIH Polysomnographic (SLPDB, Boston) and UCD Sleep Apnea
  (UCDDB, Dublin) databases, evaluated under Leave-One-Database-Out (LODO) transfer
  (`leave_one_database_out_3center.py`).
- **Provenance-fixed tri-center LODO** with patient-clustered bootstrap CIs
  (`run_03_tri_center_lodo_provenance_fix.py`), including paired few-shot minus
  zero-shot TabPFN differences.
- **Label efficiency** benchmark on subsampled training sets
  (`label_efficiency.py`).

### Leakage controls (zero-leakage policy)

- `SimpleImputer` and `StandardScaler` are always fit on the training split /
  source cohorts only, never on test data or the pooled dataset.
- Subject isolation is enforced by `GroupKFold` on `record_id`; in the tri-center
  runs, SLPDB records sharing a numeric subject prefix are mapped to one subject
  (`configs/protocol.py::SLPDB_SUBJECT_MAP`).
- Few-shot adaptation samples context epochs from one calibration patient and
  evaluates on strictly disjoint held-out patients.

### Subject-level diagnostic rule

A subject is labeled OSA-positive when the true apnea burden is >= 5 apnea minutes
per recorded hour (duration-normalized AHI>=5 analogue), not a raw minute count.
Operating thresholds are selected leave-one-subject-out (Youden J on all other
subjects) to avoid optimistic self-tuning.

## Repository layout

| Path | Purpose |
|---|---|
| `models/train_and_evaluate.py` | Baselines, TabPFN wrapper, `evaluate_predictions` (also mirrored at `src/models/` for the provenance run) |
| `evaluation/clinical_metrics.py` | ECE, DCA, bootstrap CIs |
| `configs/protocol.py` | Primary features, seeds, data paths, SLPDB subject map |
| `run_full_development_study.py` | Development cohort benchmark (GroupKFold, calibration, DCA, subject-level) |
| `leave_one_database_out_3center.py` | Tri-center LODO, pairwise transfer matrix, few-shot adaptation |
| `run_03_tri_center_lodo_provenance_fix.py` | Provenance-fixed LODO with patient-clustered bootstrap |
| `label_efficiency.py` | Low-N training efficiency benchmark |
| `clinical_metrics.py` (root) | Legacy copy of the calibration module (kept for reference) |
| `development_severity_metrics.csv` | Development cohort severity-tier metrics |
| `lodo_patient_clustered_metrics.csv` | Tri-center LODO metrics with patient-clustered 95% CIs (SLPDB N=1 CI columns are empty; see Uncertainty_Status) |
| `lodo_paired_differences.csv` | Paired few-shot vs zero-shot TabPFN deltas |
| `metrics_errata_log.md` | Metric reproducibility errata (float32/float64, terminology) |

> Note: a Holm-corrected multi-comparison table (each model vs TabPFN on identical
> patient-clustered bootstrap resamples) is emitted by `run_03_tri_center_lodo_provenance_fix.py`
> to `metrics/lodo_multiple_comparison_holm.csv`.
| `lodo_bootstrap_resamples_manifest.json` | Bootstrap resample validity manifest |

## Data access

The original physiological recordings are **not redistributed** in this repository.
Source datasets were accessed from PhysioNet and remain subject to their own
licenses, access conditions, and citation requirements. Users must obtain the
original datasets directly from the source providers and comply with all applicable
terms.

Expected processed inputs (per-epoch HRV features with `record_id`, `apnea_label`):

```
data/processed/hrv_features_apnea_ecg_full.parquet
data/processed/hrv_features_slpdb.parquet
data/processed/hrv_features_ucddb.parquet
```

The exact HRV feature extraction pipeline (R-peak detection, artifact filtering,
feature definitions) must be documented in the manuscript methods; features used by
the tri-center run are listed in `configs/protocol.py::PRIMARY_FEATURES`.

## Interpreting results (framing for the manuscript)

- Checked-in summary CSVs were produced by an earlier pipeline revision with a
  preprocessing leakage defect; regenerate all numbers with the fixed scripts
  before resubmission.
- The few-shot vs zero-shot TabPFN paired deltas are **not significant** (clustered
  95% CIs cross zero in every cohort), and the external test sets are small
  (UCDDB: 6 held-out subjects; SLPDB: 1 held-out subject, CIs unestimable).
  Claims should therefore be framed as **exploratory / hypothesis-generating**.
- SLPDB inter-subject CIs are mathematically unestimable with a single test
  subject; the corresponding CI columns are left empty and the reason recorded
  in `Uncertainty_Status`.

## Installation and running

```bash
pip install -r requirements.txt

python run_full_development_study.py
python label_efficiency.py
python leave_one_database_out_3center.py
python run_03_tri_center_lodo_provenance_fix.py
```

The provenance-fixed run expects a local TabPFN checkpoint; set its path in
`configs/protocol.py::TABPFN_CHECKPOINT_PATH`. The checkpoint is not redistributed.

## TabPFN and third-party components

The MIT License in this repository applies only to the original code authored for
this project. It does **not** apply to PhysioNet source datasets, TabPFN model
weights or checkpoints, third-party libraries, frameworks, or external tools.

## License

The original source code in this repository is licensed under the MIT License.
