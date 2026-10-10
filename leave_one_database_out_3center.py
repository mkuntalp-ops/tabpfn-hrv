"""
Three-Center Leave-One-Database-Out (LODO) Multi-Cohort Benchmark
Evaluates TabPFN vs Baselines across three international clinical sleep cohorts:
1. Center 1: PhysioNet Apnea-ECG (Philipps University Marburg, Germany; 100 Hz single-lead ECG)
2. Center 2: MIT-BIH Polysomnographic Database (slpdb, Beth Israel Hospital, Boston; 250 Hz multi-lead ECG)
3. Center 3: UCD Sleep Apnea Database (ucddb, St. Vincent's University Hospital Dublin; 128 Hz multi-channel PSG ECG)

Evaluates:
- Full LODO: Leave each center out as test, train on the other two centers
- Pairwise Cross-Cohort Transferability Matrix (3x3)
- In-Context Few-Shot Adaptation (N=50, 100) on each unseen target hospital
- Calibration (Brier Score, ECE) under multi-source distribution shift
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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.train_and_evaluate import get_baseline_models, get_tabpfn_model, evaluate_predictions
from evaluation.clinical_metrics import compute_expected_calibration_error, calculate_dca

warnings.filterwarnings("ignore")

def run_three_center_lodo(
    apnea_path: str = "data/processed/hrv_features_apnea_ecg_full.parquet",
    slpdb_path: str = "data/processed/hrv_features_slpdb.parquet",
    ucddb_path: str = "data/processed/hrv_features_ucddb.parquet",
    output_dir: str = "results/three_center_lodo"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading Three Multi-Center Clinical Cohorts...")
    df_apnea = pd.read_parquet(apnea_path)
    df_apnea["cohort"] = "Apnea-ECG (Germany)"

    df_slpdb = pd.read_parquet(slpdb_path)
    df_slpdb["cohort"] = "SLPDB (Boston, USA)"

    df_ucddb = pd.read_parquet(ucddb_path)
    df_ucddb["cohort"] = "UCDDB (Dublin, Ireland)"

    cohorts_dict = {
        "Apnea-ECG": df_apnea,
        "SLPDB": df_slpdb,
        "UCDDB": df_ucddb
    }

    # Identify shared HRV features
    feature_cols = [
        c for c in df_apnea.columns
        if c in df_slpdb.columns and c in df_ucddb.columns and (c.startswith("HRV_") or c == "num_r_peaks")
    ]
    print(f"[+] Found {len(feature_cols)} harmonized multi-domain HRV features:")
    print(f"    {feature_cols}")

    for name, df_c in cohorts_dict.items():
        pos = (df_c["apnea_label"] == 1).sum()
        neg = (df_c["apnea_label"] == 0).sum()
        print(f"[+] {name:10s}: {len(df_c):5d} epochs across {df_c['record_id'].nunique():2d} patients | Apnea: {pos:4d} ({pos/len(df_c)*100:.1f}%), Normal: {neg:4d}")

    # Combine into unified dataset
    combined_df = pd.concat([df_apnea, df_slpdb, df_ucddb], ignore_index=True)
    combined_df[feature_cols] = combined_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # ZERO LEAKAGE: imputation is deferred to each fold and fit on source cohorts only
    X_all = combined_df[feature_cols].values
    y_all = combined_df["apnea_label"].values.astype(int)
    cohort_labels = combined_df["cohort"].values
    patient_records = combined_df["record_id"].values

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] TabPFN execution device: {device}")

    cohort_names = ["Apnea-ECG", "SLPDB", "UCDDB"]
    cohort_keys = {
        "Apnea-ECG": "Apnea-ECG (Germany)",
        "SLPDB": "SLPDB (Boston, USA)",
        "UCDDB": "UCDDB (Dublin, Ireland)"
    }

    # ==============================================================
    # 1. Leave-One-Database-Out (LODO) Evaluation (Train on 2, Test on 1)
    # ==============================================================
    print("\n" + "="*70)
    print("[*] 1. Leave-One-Database-Out (LODO) Multi-Center Cross-Validation")
    print("="*70)

    lodo_results = []

    for test_name in cohort_names:
        test_key = cohort_keys[test_name]
        train_keys = [cohort_keys[c] for c in cohort_names if c != test_name]

        test_mask = (cohort_labels == test_key)
        train_mask = np.isin(cohort_labels, train_keys)

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]

        # ZERO LEAKAGE: imputer and scaler are fit on the source (train) cohorts only
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(imputer.transform(X_test))

        train_cohort_names = [c for c in cohort_names if c != test_name]
        print(f"\n--- LODO Fold: Test on [{test_name}] ({len(X_test)} samples) | Train on {train_cohort_names} ({len(X_train)} samples) ---")

        # Baselines
        baselines = get_baseline_models()
        for b_name, clf in baselines.items():
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
            m = evaluate_predictions(y_test, probs)
            m["ECE"] = compute_expected_calibration_error(y_test, probs)
            m["Model"] = b_name
            m["Target_Cohort"] = test_name
            m["Train_Cohorts"] = "+".join(train_cohort_names)
            m["Paradigm"] = "Zero-Shot LODO"
            lodo_results.append(m)
            print(f"  {b_name:20s} | ROC-AUC: {m['ROC-AUC']:.4f} [95% CI: {m['ROC-AUC_CI_lower']:.4f}-{m['ROC-AUC_CI_upper']:.4f}] | PR-AUC: {m['PR-AUC']:.4f} | Brier: {m['BrierScore']:.4f}")

        # TabPFN Zero-Shot Foundation Model
        try:
            tabpfn = get_tabpfn_model(n_estimators=4, device=device)
            tabpfn.fit(X_train_scaled, y_train)
            t_probs = tabpfn.predict_proba(X_test_scaled)
            if hasattr(t_probs, "ndim") and t_probs.ndim == 2:
                t_probs = t_probs[:, 1]

            t_m = evaluate_predictions(y_test, t_probs)
            t_m["ECE"] = compute_expected_calibration_error(y_test, t_probs)
            t_m["Model"] = "TabPFN"
            t_m["Target_Cohort"] = test_name
            t_m["Train_Cohorts"] = "+".join(train_cohort_names)
            t_m["Paradigm"] = "Zero-Shot LODO"
            lodo_results.append(t_m)
            print(f"  {'TabPFN':20s} | ROC-AUC: {t_m['ROC-AUC']:.4f} [95% CI: {t_m['ROC-AUC_CI_lower']:.4f}-{t_m['ROC-AUC_CI_upper']:.4f}] | PR-AUC: {t_m['PR-AUC']:.4f} | Brier: {t_m['BrierScore']:.4f}")
        except Exception as e:
            print(f"[-] TabPFN LODO error on {test_name}: {e}")

        # TabPFN Few-Shot In-Context Target Adaptation with STRICT PATIENT ISOLATION (Zero Leakage)
        test_record_ids = patient_records[test_mask]
        distinct_test_patients = sorted(list(np.unique(test_record_ids)))

        if len(distinct_test_patients) > 1:
            calib_patient = distinct_test_patients[0]
            eval_patients = distinct_test_patients[1:]

            calib_cand_indices = np.where(test_record_ids == calib_patient)[0]
            eval_indices = np.where(test_record_ids != calib_patient)[0]

            np.random.seed(42)
            for n_target in [50, 100]:
                n_sample = min(n_target, len(calib_cand_indices))
                t_idx = np.random.choice(calib_cand_indices, size=n_sample, replace=False)

                # Verify 0% patient leakage
                calib_pats = np.unique(test_record_ids[t_idx])
                eval_pats = np.unique(test_record_ids[eval_indices])
                assert len(np.intersect1d(calib_pats, eval_pats)) == 0, "Patient leakage detected!"

                X_t_adapt = X_test_scaled[t_idx]
                y_t_adapt = y_test[t_idx]
                X_t_eval = X_test_scaled[eval_indices]
                y_t_eval = y_test[eval_indices]

                try:
                    tabpfn_fs = get_tabpfn_model(n_estimators=4, device=device)
                    tabpfn_fs.fit(X_t_adapt, y_t_adapt)
                    fs_probs = tabpfn_fs.predict_proba(X_t_eval)
                    if hasattr(fs_probs, "ndim") and fs_probs.ndim == 2:
                        fs_probs = fs_probs[:, 1]

                    fs_m = evaluate_predictions(y_t_eval, fs_probs)
                    fs_m["ECE"] = compute_expected_calibration_error(y_t_eval, fs_probs)
                    fs_m["Model"] = f"TabPFN (Target-N={n_target})"
                    fs_m["Target_Cohort"] = test_name
                    fs_m["Train_Cohorts"] = f"CalibPatient-{calib_patient} (N={n_sample})"
                    fs_m["Paradigm"] = f"Strict Patient-Independent Few-Shot (N={n_target})"
                    lodo_results.append(fs_m)
                    print(f"  TabPFN (Strict Few-Shot N={n_sample:3d}) | Calib Patient: {calib_patient} -> Evaluated on {eval_patients} | ROC-AUC: {fs_m['ROC-AUC']:.4f} [95% CI: {fs_m['ROC-AUC_CI_lower']:.4f}-{fs_m['ROC-AUC_CI_upper']:.4f}] | PR-AUC: {fs_m['PR-AUC']:.4f}")
                except Exception as e:
                    print(f"[-] Strict few-shot error on {test_name}: {e}")

    lodo_df = pd.DataFrame(lodo_results)
    lodo_df.to_csv(out_dir / "three_center_lodo_metrics.csv", index=False)
    print(f"\n[+] Saved 3-Center LODO metrics to: {out_dir / 'three_center_lodo_metrics.csv'}")

    # ==============================================================
    # 2. Pairwise Cross-Database Transfer Matrix (3x3)
    # ==============================================================
    print("\n" + "="*70)
    print("[*] 2. Pairwise Cross-Database Transfer Matrix (Source vs Target)")
    print("="*70)

    pairwise_results = []
    models_to_test = ["TabPFN", "LightGBM", "RandomForest"]

    for src_name in cohort_names:
        for tgt_name in cohort_names:
            src_key = cohort_keys[src_name]
            tgt_key = cohort_keys[tgt_name]

            src_mask = (cohort_labels == src_key)
            tgt_mask = (cohort_labels == tgt_key)

            X_src, y_src = X_all[src_mask], y_all[src_mask]
            X_tgt, y_tgt = X_all[tgt_mask], y_all[tgt_mask]

            # ZERO LEAKAGE: fit on source only
            pair_imputer = SimpleImputer(strategy="median")
            X_src_imp = pair_imputer.fit_transform(X_src)
            scaler = StandardScaler()
            X_src_scaled = scaler.fit_transform(X_src_imp)
            X_tgt_scaled = scaler.transform(pair_imputer.transform(X_tgt))

            # TabPFN
            try:
                clf_tab = get_tabpfn_model(n_estimators=4, device=device)
                clf_tab.fit(X_src_scaled, y_src)
                p_tab = clf_tab.predict_proba(X_tgt_scaled)
                if hasattr(p_tab, "ndim") and p_tab.ndim == 2:
                    p_tab = p_tab[:, 1]
                m_tab = evaluate_predictions(y_tgt, p_tab)
                pairwise_results.append({
                    "Source": src_name,
                    "Target": tgt_name,
                    "Model": "TabPFN",
                    "ROC-AUC": m_tab["ROC-AUC"],
                    "BrierScore": m_tab["BrierScore"]
                })
            except Exception as e:
                print(f"[-] Pairwise TabPFN error {src_name}->{tgt_name}: {e}")

            # LightGBM and RandomForest
            for b_name in ["LightGBM", "RandomForest"]:
                try:
                    clf_b = get_baseline_models()[b_name]
                    clf_b.fit(X_src_scaled, y_src)
                    p_b = clf_b.predict_proba(X_tgt_scaled)[:, 1]
                    m_b = evaluate_predictions(y_tgt, p_b)
                    pairwise_results.append({
                        "Source": src_name,
                        "Target": tgt_name,
                        "Model": b_name,
                        "ROC-AUC": m_b["ROC-AUC"],
                        "BrierScore": m_b["BrierScore"]
                    })
                except Exception as e:
                    print(f"[-] Pairwise {b_name} error {src_name}->{tgt_name}: {e}")

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv(out_dir / "pairwise_transfer_matrix.csv", index=False)

    # ==============================================================
    # 3. Visualizations
    # ==============================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # A. 3-Center LODO Performance Bar Chart
    plt.figure(figsize=(12, 6))
    zero_df = lodo_df[lodo_df["Paradigm"] == "Zero-Shot LODO"]
    sns.barplot(data=zero_df, x="Target_Cohort", y="ROC-AUC", hue="Model", palette="crest")
    plt.title("Three-Center Leave-One-Database-Out (LODO): External ROC-AUC per Target Hospital", fontsize=13, fontweight="bold")
    plt.ylabel("External Test ROC-AUC", fontsize=12)
    plt.xlabel("Unseen Target Hospital Cohort", fontsize=12)
    plt.ylim(0.45, 0.85)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    chart1 = out_dir / "three_center_lodo_barchart.png"
    plt.savefig(chart1, dpi=300)
    plt.close()
    print(f"[+] Saved LODO bar chart to: {chart1}")

    # B. Heatmap: Pairwise Transfer Matrix (TabPFN vs LightGBM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, m_name in enumerate(["TabPFN", "LightGBM"]):
        sub_pw = pairwise_df[pairwise_df["Model"] == m_name]
        pivot_auc = sub_pw.pivot(index="Source", columns="Target", values="ROC-AUC")
        sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.50, vmax=0.85, ax=axes[idx], cbar=(idx==1))
        axes[idx].set_title(f"{m_name} Cross-Cohort Transfer (ROC-AUC)", fontweight="bold")
        axes[idx].set_ylabel("Source Training Cohort" if idx==0 else "")
        axes[idx].set_xlabel("Target Testing Cohort")
    plt.tight_layout()
    chart2 = out_dir / "cross_center_transfer_heatmap.png"
    plt.savefig(chart2, dpi=300)
    plt.close()
    print(f"[+] Saved Transfer Heatmap to: {chart2}")

    # C. In-Context Few-Shot Adaptation Curve across Hospitals
    fs_df = lodo_df[lodo_df["Paradigm"].str.contains("Few-Shot|Zero-Shot") & (lodo_df["Model"].str.contains("TabPFN"))]
    paradigm_order = ["Zero-Shot LODO", "Strict Patient-Independent Few-Shot (N=50)", "Strict Patient-Independent Few-Shot (N=100)"]
    fs_df = fs_df.copy()
    fs_df["Paradigm"] = pd.Categorical(fs_df["Paradigm"], categories=paradigm_order, ordered=True)
    sns.lineplot(data=fs_df, x="Paradigm", y="ROC-AUC", hue="Target_Cohort", marker="o", lw=2.5, markersize=8)
    plt.title("TabPFN In-Context Few-Shot Hospital Adaptation", fontsize=13, fontweight="bold")
    plt.xlabel("Adaptation Paradigm", fontsize=12)
    plt.ylabel("ROC-AUC on Target Center", fontsize=12)
    plt.ylim(0.55, 0.85)
    plt.tight_layout()
    chart3 = out_dir / "three_center_few_shot_adaptation.png"
    plt.savefig(chart3, dpi=300)
    plt.close()
    print(f"[+] Saved Few-Shot Adaptation chart to: {chart3}")

    print("\n[✓] Three-Center LODO Evaluation completed successfully!")
    return lodo_df, pairwise_df

if __name__ == "__main__":
    run_three_center_lodo()
