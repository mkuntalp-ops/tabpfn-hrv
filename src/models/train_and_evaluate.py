"""
Baseline models, TabPFN wrapper, and prediction evaluation.
Shared by all benchmark scripts. Fixed random seeds for reproducibility.
"""
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    roc_curve,
)

RANDOM_SEED = 42


def get_baseline_models(random_state: int = RANDOM_SEED) -> dict:
    """Deterministic baseline classifiers with fixed hyperparameters."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=1, random_state=random_state,
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=random_state
        ),
        "LightGBM": _lgbm_or_none(random_state),
        "XGBoost": _xgb_or_none(random_state),
    }


def _lgbm_or_none(random_state):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            random_state=random_state, n_jobs=1, verbose=-1,
        )
    except ImportError:
        return None


def _xgb_or_none(random_state):
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            random_state=random_state, n_jobs=1, eval_metric="logloss",
        )
    except ImportError:
        return None


def get_tabpfn_model(n_estimators: int = 4, device: str = "cpu", model_path: str = None):
    """TabPFN classifier wrapper. Falls back to TabPFN v2 API if v3 unavailable."""
    from tabpfn import TabPFNClassifier

    kwargs = dict(n_estimators=n_estimators, device=device)
    if model_path is not None:
        return TabPFNClassifier(
            n_estimators=n_estimators,
            device=device,
            model_path=str(model_path),
            ignore_pretraining_limits=True,
        )
    try:
        return TabPFNClassifier(**kwargs)
    except TypeError:
        return TabPFNClassifier(
            n_estimators=n_estimators, device=device, ignore_pretraining_limits=True
        )


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Segment-level metrics plus patient-independent bootstrap 95% CIs.

    Bootstrap resamples are epoch-level here; for cluster-aware CIs use
    the patient-clustered bootstrap in run_03_tri_center_lodo_provenance_fix.py.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    auc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    auc_boot = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_boot.append(roc_auc_score(yt, y_prob[idx]))
    if auc_boot:
        ci_lower, ci_upper = np.percentile(auc_boot, [2.5, 97.5])
    else:
        ci_lower = ci_upper = np.nan

    return {
        "ROC-AUC": float(auc),
        "ROC-AUC_CI_lower": float(ci_lower),
        "ROC-AUC_CI_upper": float(ci_upper),
        "PR-AUC": float(pr),
        "BrierScore": float(brier),
        "Accuracy": float(acc),
        "F1-Score": float(f1),
    }


def select_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden J-optimal probability threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])
