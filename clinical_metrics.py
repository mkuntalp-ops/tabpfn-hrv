"""
DEPRECATED: this root-level copy is kept only for backwards compatibility.
Use `evaluation.clinical_metrics` instead, which contains the corrected ECE
implementation (the last calibration bin includes y_prob == 1.0).

This file re-exports the canonical implementations.
"""
from evaluation.clinical_metrics import (  # noqa: F401
    compute_expected_calibration_error,
    calculate_dca,
    bootstrap_metric_ci,
)
