"""
Full Development Cohort Benchmark: TabPFN v2 vs Baselines (All 35 Apnea-ECG Records)
Evaluates:
1. 5-Fold Patient-Independent GroupKFold Cross-Validation (TRIPOD+AI compliant)
2. Segment-level discrimination (ROC-AUC, PR-AUC, Accuracy, F1)
3. Calibration (Brier Score, Expected Calibration Error - ECE, Calibration Curves)
4. Clinical Utility (Decision Curve Analysis - DCA Net Benefit)
5. Subject-Level Diagnostic Classification (Apnea vs Normal per patient, Sensitivity, Specificity,
   duration-normalized apnea burden correlation, LOSO-selected Youden thresholds)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import sys
import warnings
import torch
torch.set_num_threads(1)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train_and_evaluate import get_baseline_models, get_tabpfn_model, evaluate_predictions
from evaluation.clinical_metrics import compute_expected_calibration_error, calculate_dca

warnings.filterwarnings("ignore")

def run_full_study(
    data_path: str = "data/processed/hrv_features_apnea_ecg_full.parquet",
    output_dir: str = "results/full_development_study",
    n_splits: int = 5
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading full development dataset: {data_path}")
    if not Path(data_path).exists():
        print(f"[-] File {data_path} not found.")
        return

    df = pd.read_parquet(data_path)
    df = df[df["record_id"] != "c06"].reset_index(drop=True)
    print(f"[+] Loaded {len(df)} minute epochs across {df['record_id'].nunique()} patient records.")
    print(f"[+] Class distribution: {df['apnea_label'].value_counts(normalize=True).to_dict()}")

    feature_cols = [c for c in df.columns if c.startswith("HRV_") or c == "num_r_peaks"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # ZERO LEAKAGE: imputation is fit inside each CV fold on the training split only
    X = df[feature_cols].values
    y = df["apnea_label"].values.astype(int)
    groups = df["record_id"].values

    # Determine device for TabPFN
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] TabPFN execution device: {device}")

    # ==============================================================
    # 1. Subject-Wise (Patient-Independent) GroupKFold CV
    # ==============================================================
    print("\n" + "="*65)
    print(f"[*] Running {n_splits}-Fold Patient-Independent GroupKFold Cross-Validation")
    print("="*65)

    gkf = GroupKFold(n_splits=n_splits)
    baseline_names = list(get_baseline_models().keys())
    all_models = baseline_names + ["TabPFN"]

    # Pre-allocate OOF prediction containers
    oof_probs = {m: np.zeros(len(df), dtype=float) for m in all_models}
    gkf_results = []
    dca_records = []

    tabpfn_fold_failures = 0
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        test_records = sorted(list(np.unique(groups[test_idx])))
        print(f"\n--- Group Fold {fold}/{n_splits} (Train: {len(train_idx)}, Test: {len(test_idx)} | Test Patients: {test_records}) ---")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ZERO LEAKAGE: imputer and scaler are fit on the training fold only
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(imputer.transform(X_test))

        # 1. Baseline models
        baselines = get_baseline_models()
        for name, clf in baselines.items():
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
            oof_probs[name][test_idx] = probs

            m = evaluate_predictions(y_test, probs)
            m["ECE"] = compute_expected_calibration_error(y_test, probs)
            m["Fold"] = fold
            m["Model"] = name
            m["CV_Type"] = "Patient-Independent (GroupKFold)"
            gkf_results.append(m)
            print(f"  {name:20s} | ROC-AUC: {m['ROC-AUC']:.4f} | PR-AUC: {m['PR-AUC']:.4f} | Brier: {m['BrierScore']:.4f} | ECE: {m['ECE']:.4f}")

            df_dca = calculate_dca(y_test, probs)
            df_dca["Model"] = name
            df_dca["Fold"] = fold
            dca_records.append(df_dca)

        # 2. TabPFN Foundation Model
        try:
            tabpfn = get_tabpfn_model(n_estimators=4, device=device)
            tabpfn.fit(X_train_scaled, y_train)
            t_probs = tabpfn.predict_proba(X_test_scaled)
            if hasattr(t_probs, "ndim") and t_probs.ndim == 2:
                t_probs = t_probs[:, 1]
            oof_probs["TabPFN"][test_idx] = t_probs

            t_m = evaluate_predictions(y_test, t_probs)
            t_m["ECE"] = compute_expected_calibration_error(y_test, t_probs)
            t_m["Fold"] = fold
            t_m["Model"] = "TabPFN"
            t_m["CV_Type"] = "Patient-Independent (GroupKFold)"
            gkf_results.append(t_m)
            print(f"  {'TabPFN':20s} | ROC-AUC: {t_m['ROC-AUC']:.4f} | PR-AUC: {t_m['PR-AUC']:.4f} | Brier: {t_m['BrierScore']:.4f} | ECE: {t_m['ECE']:.4f}")

            df_dca = calculate_dca(y_test, t_probs)
            df_dca["Model"] = "TabPFN"
            df_dca["Fold"] = fold
            dca_records.append(df_dca)
        except Exception as e:
            tabpfn_fold_failures += 1
            print(f"[-] TabPFN fold {fold} error: {e}")

    if tabpfn_fold_failures > 0:
        print(f"[!] WARNING: TabPFN failed in {tabpfn_fold_failures}/{n_splits} folds; "
              f"its summary metrics are computed on {n_splits - tabpfn_fold_failures} folds only.")

    # Save raw fold metrics & summary
    results_df = pd.DataFrame(gkf_results)
    results_df.to_csv(out_dir / "group_cv_metrics_raw.csv", index=False)

    summary = results_df.groupby("Model")[["ROC-AUC", "PR-AUC", "BrierScore", "ECE", "Accuracy", "F1-Score"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "group_cv_summary.csv")

    print("\n" + "="*65)
    print("PATIENT-INDEPENDENT (GROUP-KFOLD) SEGMENT-LEVEL CV SUMMARY:")
    print("="*65)
    print(summary)

    # Save OOF predictions DataFrame
    oof_df = df[["record_id", "minute_idx", "apnea_label"]].copy()
    for m in all_models:
        oof_df[f"prob_{m}"] = oof_probs[m]
    oof_df.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    # ==============================================================
    # 2. Subject-Level Diagnostic Classification (AHI / Total Apnea)
    # ==============================================================
    print("\n" + "="*65)
    print(f"SUBJECT-LEVEL DIAGNOSTIC CLASSIFICATION ({len(patient_ids)} PATIENTS):")
    print("="*65)

    subject_rows = []
    patient_ids = df["record_id"].unique()

    # Recording duration per subject (hours) for AHI-style definitions
    subject_hours = df.groupby("record_id")["minute_idx"].max() + 1
    subject_hours = (subject_hours / 60.0).to_dict()

    for pid in patient_ids:
        pdf = oof_df[oof_df["record_id"] == pid]
        true_apnea_min = (pdf["apnea_label"] == 1).sum()
        total_min = len(pdf)
        # Clinical rule: OSA positive if average apnea burden >= 5 minutes per
        # recorded hour (AHI >= 5 equivalent, duration-normalized)
        rec_hours = max(subject_hours.get(pid, total_min / 60.0), 1e-9)
        true_class = 1 if (true_apnea_min / rec_hours) >= 5.0 else 0

        row = {
            "record_id": pid,
            "total_minutes": total_min,
            "true_apnea_minutes": true_apnea_min,
            "true_apnea_minutes_per_hour": true_apnea_min / rec_hours,
            "recording_hours": rec_hours,
            "true_subject_class": true_class,
        }

        # Duration-normalized predicted apnea burden (apnea minutes per hour)
        for m in all_models:
            pred_apnea_min = (pdf[f"prob_{m}"] >= 0.5).sum()
            total_min = len(pdf)
            patient_pred_prob = (pred_apnea_min / rec_hours) if rec_hours > 0 else 0
            row[f"{m}_pred_burden"] = patient_pred_prob
            row[f"{m}_pred_apnea_min"] = pred_apnea_min

        subject_rows.append(row)

    subj_df = pd.DataFrame(subject_rows)

    # Leave-one-subject-out Youden threshold selection: each subject is classified
    # with a threshold optimized on all OTHER subjects (no optimistic self-tuning).
    from sklearn.metrics import roc_curve
    from models.train_and_evaluate import select_youden_threshold

    n_subj = len(subj_df)
    for m in all_models:
        y_true_s = subj_df["true_subject_class"].values
        y_burden_s = subj_df[f"{m}_pred_burden"].values
        subj_df[f"{m}_pred_class"] = 0
        for i in range(n_subj):
            rest = np.arange(n_subj) != i
            if len(np.unique(y_true_s[rest])) < 2:
                opt_thresh = 5.0  # fall back to the clinical AHI>=5 rule
            else:
                opt_thresh = select_youden_threshold(y_true_s[rest], y_burden_s[rest])
            subj_df.loc[subj_df.index[i], f"{m}_opt_thresh"] = opt_thresh
            subj_df.loc[subj_df.index[i], f"{m}_pred_class"] = int(y_burden_s[i] >= opt_thresh)
        subj_df[f"{m}_pred_class"] = subj_df[f"{m}_pred_class"].astype(int)

    subj_df.to_csv(out_dir / "subject_level_predictions.csv", index=False)

    subj_metrics = []
    for m in all_models:
        y_true_s = subj_df["true_subject_class"].values
        y_pred_s = subj_df[f"{m}_pred_class"].values
        pred_mins = subj_df[f"{m}_pred_apnea_min"].values
        true_mins = subj_df["true_apnea_minutes"].values
        # also report duration-normalized burden correlation
        r_burden, _ = pearsonr(subj_df["true_apnea_minutes_per_hour"].values, subj_df[f"{m}_pred_burden"].values)

        tp = np.sum((y_true_s == 1) & (y_pred_s == 1))
        tn = np.sum((y_true_s == 0) & (y_pred_s == 0))
        fp = np.sum((y_true_s == 0) & (y_pred_s == 1))
        fn = np.sum((y_true_s == 1) & (y_pred_s == 0))

        acc = (tp + tn) / len(y_true_s)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        r_corr, _ = pearsonr(true_mins, pred_mins)

        subj_metrics.append({
            "Model": m,
            "Subject_Accuracy": acc,
            "Subject_Sensitivity": sens,
            "Subject_Specificity": spec,
            "Pearson_r_Apnea_Minutes": r_corr,
            "Pearson_r_Apnea_Burden_per_Hour": r_burden,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

    subj_metrics_df = pd.DataFrame(subj_metrics).sort_values("Subject_Accuracy", ascending=False)
    subj_metrics_df.to_csv(out_dir / "subject_level_metrics.csv", index=False)
    print(subj_metrics_df.to_string(index=False))

    # ==============================================================
    # 3. Visualizations
    # ==============================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # A. Bar Chart: ROC-AUC & Brier Score
    plt.figure(figsize=(10, 6))
    df_plot = results_df[["Model", "ROC-AUC", "BrierScore"]].melt(id_vars="Model", var_name="Metric", value_name="Score")
    palette = sns.color_palette("viridis", n_colors=len(all_models))
    ax = sns.barplot(data=df_plot, x="Metric", y="Score", hue="Model", palette="crest", errorbar="sd", capsize=0.08)
    plt.title("Patient-Independent Group-KFold CV (Segment-Level Performance)", fontsize=13, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    chart_file = out_dir / "patient_independent_cv_barchart.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()
    print(f"[+] Saved GroupKFold plot to: {chart_file}")

    # B. Decision Curve Analysis (DCA)
    if dca_records:
        all_dca = pd.concat(dca_records, ignore_index=True)
        all_dca.to_csv(out_dir / "dca_net_benefit_all_folds.csv", index=False)
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=all_dca, x="threshold", y="net_benefit_model", hue="Model", lw=2, errorbar="sd")
        ref_all = all_dca[all_dca["Model"] == "TabPFN"][["threshold", "net_benefit_all"]]
        plt.plot(ref_all["threshold"], ref_all["net_benefit_all"], label="Treat All", color="gray", linestyle="--")
        plt.axhline(0, label="Treat None", color="black", linestyle=":")
        plt.title("Decision Curve Analysis (DCA): Clinical Net Benefit", fontweight="bold")
        plt.xlabel("Threshold Probability")
        plt.ylabel("Net Benefit")
        plt.ylim(-0.05, 0.60)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        dca_file = out_dir / "decision_curve_analysis.png"
        plt.savefig(dca_file, dpi=300)
        plt.close()
        print(f"[+] Saved DCA plot to: {dca_file}")

    # C. Calibration Curves (Reliability Diagram)
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    for m in all_models:
        prob_m = oof_probs[m]
        frac_pos, mean_pred = calibration_curve(y, prob_m, n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, "s-", label=f"{m}")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (True Frequency)", fontsize=12)
    plt.title("Reliability Diagram (Out-of-Fold Calibration)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    calib_file = out_dir / "calibration_curves.png"
    plt.savefig(calib_file, dpi=300)
    plt.close()
    print(f"[+] Saved Calibration curves to: {calib_file}")

    # D. Subject-Level Correlation Plot: TabPFN vs True Apnea Minutes
    plt.figure(figsize=(7, 6))
    sns.regplot(
        data=subj_df,
        x="true_apnea_minutes",
        y="TabPFN_pred_apnea_min",
        scatter_kws={"alpha": 0.7, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "lw": 2}
    )
    plt.xlabel("True Apnea Minutes (Polysomnography Gold Standard)", fontsize=11)
    plt.ylabel("TabPFN Predicted Apnea Minutes (Single-Lead ECG)", fontsize=11)
    plt.title(f"Subject-Level Apnea Burden Estimation (r = {subj_metrics_df.loc[subj_metrics_df['Model']=='TabPFN', 'Pearson_r_Apnea_Minutes'].values[0]:.3f})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_file = out_dir / "subject_apnea_minutes_correlation.png"
    plt.savefig(corr_file, dpi=300)
    plt.close()
    print(f"[+] Saved Subject correlation plot to: {corr_file}")

    print("\n[✓] Full Development Study successfully completed!")
    return summary, subj_metrics_df

if __name__ == "__main__":
    run_full_study()
