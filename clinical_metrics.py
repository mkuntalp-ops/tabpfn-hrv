"""
Clinical Evaluation Metrics: Calibration, Decision Curve Analysis (DCA), and Bootstrap CIs
Essential for high-impact clinical/medical journal reporting (TRIPOD+AI standard).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def calculate_dca(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.05, 0.95, 91)
) -> pd.DataFrame:
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.
    Reference: Vickers & Elkin (2006).
    """
    n = len(y_true)
    prevalence = np.mean(y_true)

    rows = []
    for pt in thresholds:
        # Classifier prediction at threshold pt
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Net Benefit for model
        weight = pt / (1.0 - pt)
        net_benefit_model = (tp / n) - (fp / n) * weight

        # Net Benefit for Treat All
        net_benefit_all = prevalence - (1.0 - prevalence) * weight

        # Net Benefit for Treat None
        net_benefit_none = 0.0

        rows.append({
            "threshold": pt,
            "net_benefit_model": net_benefit_model,
            "net_benefit_all": net_benefit_all,
            "net_benefit_none": net_benefit_none
        })

    return pd.DataFrame(rows)

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Calculates 95% Confidence Intervals for ROC-AUC, PR-AUC, Brier score, and ECE using bootstrapping.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    auc_scores = []
    pr_scores = []
    brier_scores = []
    ece_scores = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_boot = y_true[idx]
        yp_boot = y_prob[idx]

        if len(np.unique(yt_boot)) < 2:
            continue

        auc_scores.append(roc_auc_score(yt_boot, yp_boot))
        pr_scores.append(average_precision_score(yt_boot, yp_boot))
        brier_scores.append(brier_score_loss(yt_boot, yp_boot))
        ece_scores.append(compute_expected_calibration_error(yt_boot, yp_boot))

    results = {}
    for name, vals in [("ROC-AUC", auc_scores), ("PR-AUC", pr_scores), ("BrierScore", brier_scores), ("ECE", ece_scores)]:
        arr = np.array(vals)
        results[name] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 100 * (alpha / 2))),
            "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2)))
        }

    return results
