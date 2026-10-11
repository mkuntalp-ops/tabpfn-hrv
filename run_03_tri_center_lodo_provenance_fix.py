"""
Script 3 (Provenance Fix & Clustered Bootstrap):
Tri-Center LODO Benchmark with Full Raw Predictions Export and Patient-Clustered Bootstrap.

Conditions strictly locked:
- Strict 9 primary features.
- Local TabPFN checkpoint (212,804,803 bytes), version pinned via configs/protocol.py.
- Deterministic seed & baseline hyperparameters.
- Exact calibration context keys verification via assertions against previous provenance log.
- Saves raw epoch-level probabilities for all zero-shot and few-shot models to:
  results/runs/run_20261006_lodo_provenance_fix/data/lodo_raw_predictions.parquet
- Compares original unresampled point estimates with old lodo_metrics_benchmark.csv.
- Computes patient-clustered bootstrap with subject-level duplication.
- Computes paired few-shot minus zero-shot differences on exact same patient draws.
- SLPDB N=1 held-out test subject marked as UNESTIMABLE (no i.i.d. epoch bootstrap).
"""

import sys, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, f1_score
from tabpfn import TabPFNClassifier

from configs.protocol import (
    PRIMARY_FEATURES, TABPFN_CHECKPOINT_PATH, RANDOM_SEED,
    DATA_PATHS, SLPDB_SUBJECT_MAP
)
from src.models.train_and_evaluate import get_baseline_models

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

def calc_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if i == n_bins - 1:
            mask = mask | (y_prob == 1.0)
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n_samples) * np.abs(bin_acc - bin_conf)
    return float(ece)

def run():
    run_dir = Path("results/runs/run_20261006_lodo_provenance_fix")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)

    log_file = run_dir / "logs/03_tri_center_lodo.log"
    out_raw_parquet = run_dir / "data/lodo_raw_predictions.parquet"
    out_old_cmp_csv = run_dir / "metrics/old_vs_new_point_estimates_comparison.csv"
    out_clustered_csv = run_dir / "metrics/lodo_patient_clustered_metrics.csv"
    out_paired_diff_csv = run_dir / "metrics/lodo_paired_differences.csv"
    out_mc_csv = run_dir / "metrics/lodo_multiple_comparison_holm.csv"
    out_boot_manifest = run_dir / "data/lodo_bootstrap_resamples_manifest.json"

    # Reference old provenance
    old_prov_path = Path("results/runs/run_20261006_unified_9feat/data/lodo_split_provenance.csv")
    assert old_prov_path.exists(), f"Old provenance file not found: {old_prov_path}"
    old_prov_df = pd.read_csv(old_prov_path)

    old_metrics_path = Path("results/runs/run_20261006_unified_9feat/metrics/lodo_metrics_benchmark.csv")
    assert old_metrics_path.exists(), f"Old metrics file not found: {old_metrics_path}"
    old_metrics_df = pd.read_csv(old_metrics_path)

    with open(log_file, "w") as log:
        def log_print(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        log_print("=" * 65)
        log_print("BENCHMARK 3 (PROVENANCE FIX): TRI-CENTER LODO & CLUSTERED BOOTSTRAP")
        log_print(f"Features: {PRIMARY_FEATURES}")
        log_print(f"Checkpoint: {TABPFN_CHECKPOINT_PATH}")
        log_print("=" * 65)

        # 1. Load Data
        cohort_dfs = {}
        for name, p in DATA_PATHS.items():
            df_c = pd.read_parquet(p)
            missing_feats = [f for f in PRIMARY_FEATURES if f not in df_c.columns]
            if missing_feats:
                available = [c for c in df_c.columns if c.startswith("HRV_") or c == "num_r_peaks"]
                log_print(f"[!] PRIMARY_FEATURES mismatch in {name}: missing {missing_feats}")
                log_print(f"    Available candidate features in {name}: {available}")
                raise AssertionError(
                    f"PRIMARY_FEATURES in configs/protocol.py is not aligned with the data schema of {name}. "
                    f"Missing: {missing_feats}. Update configs/protocol.py::PRIMARY_FEATURES to match "
                    f"the feature set of the run that produced the reference outputs."
                )
            if "subject_id" not in df_c.columns:
                if name == "SLPDB":
                    df_c["subject_id"] = df_c["record_id"].map(SLPDB_SUBJECT_MAP)
                else:
                    df_c["subject_id"] = df_c["record_id"]
            if "epoch_idx" not in df_c.columns:
                df_c["epoch_idx"] = df_c["minute_idx"] if "minute_idx" in df_c.columns else np.arange(len(df_c))
            cohort_dfs[name] = df_c

        lodo_folds = [
            {"target": "Apnea-ECG", "train": ["SLPDB", "UCDDB"], "calib_subj": "a01", "calib_rec": ["a01"]},
            {"target": "SLPDB", "train": ["Apnea-ECG", "UCDDB"], "calib_subj": "slp01", "calib_rec": ["slp01a"]},
            {"target": "UCDDB", "train": ["Apnea-ECG", "SLPDB"], "calib_subj": "ucddb002", "calib_rec": ["ucddb002"]},
        ]

        raw_pred_records = []
        new_point_metrics = []

        for fold in lodo_folds:
            target_name = fold["target"]
            train_names = fold["train"]
            calib_subj = fold["calib_subj"]
            calib_records = fold["calib_rec"]

            log_print(f"\n" + "=" * 60)
            log_print(f"--- LODO FOLD: Target = [{target_name}] | Train = {train_names} ---")
            log_print("=" * 60)

            # Assemble training data
            train_dfs = [cohort_dfs[n] for n in train_names]
            df_train = pd.concat(train_dfs, ignore_index=True)
            X_train = df_train[PRIMARY_FEATURES].values
            y_train = df_train["apnea_label"].values.astype(int)

            df_target = cohort_dfs[target_name]
            X_target = df_target[PRIMARY_FEATURES].values
            y_target = df_target["apnea_label"].values.astype(int)

            # Fit preprocessor on Source ONLY
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)

            X_target_scaled = scaler.transform(imputer.transform(X_target))

            # Train Zero-Shot Models on Source
            models = get_baseline_models()
            for m_name, model in models.items():
                t0 = time.time()
                model.fit(X_train_scaled, y_train)
                t_fit = time.time() - t0

                # Predict on full target cohort
                t0 = time.time()
                p_full = model.predict_proba(X_target_scaled)[:, 1]
                t_pred = time.time() - t0

                # Record raw predictions
                for i in range(len(df_target)):
                    raw_pred_records.append({
                        "run_id": "run_20261006_lodo_provenance_fix",
                        "target_cohort": target_name,
                        "subject_id": str(df_target.iloc[i]["subject_id"]),
                        "record_id": str(df_target.iloc[i]["record_id"]),
                        "epoch_idx": int(df_target.iloc[i]["epoch_idx"]),
                        "y_true": int(y_target[i]),
                        "model": m_name,
                        "condition": "zero_shot",
                        "context_n": 0,
                        "sampling_seed": -1,
                        "y_prob": float(p_full[i])
                    })

                log_print(f"  [Zero-Shot] {m_name:18s} trained & predicted on {target_name} ({len(p_full)} epochs)")

            # TabPFN Zero-Shot
            tabpfn_zs = TabPFNClassifier(
                n_estimators=4,
                device="cpu",
                model_path=str(TABPFN_CHECKPOINT_PATH),
                ignore_pretraining_limits=True
            )
            tabpfn_zs.fit(X_train_scaled, y_train)
            p_tab_full = tabpfn_zs.predict_proba(X_target_scaled)[:, 1]

            for i in range(len(df_target)):
                raw_pred_records.append({
                    "run_id": "run_20261006_lodo_provenance_fix",
                    "target_cohort": target_name,
                    "subject_id": str(df_target.iloc[i]["subject_id"]),
                    "record_id": str(df_target.iloc[i]["record_id"]),
                    "epoch_idx": int(df_target.iloc[i]["epoch_idx"]),
                    "y_true": int(y_target[i]),
                    "model": "TabPFN",
                    "condition": "zero_shot",
                    "context_n": 0,
                    "sampling_seed": -1,
                    "y_prob": float(p_tab_full[i])
                })
            log_print(f"  [Zero-Shot] TabPFN             trained & predicted on {target_name} ({len(p_tab_full)} epochs)")

            # Target Calibration Masks & Evaluation Subsets
            calib_cand_mask = (df_target["subject_id"] == calib_subj) & (df_target["record_id"].isin(calib_records))
            eval_mask = (df_target["subject_id"] != calib_subj)
            calib_indices = np.where(calib_cand_mask)[0]
            eval_indices = np.where(eval_mask)[0]

            X_eval_scaled = X_target_scaled[eval_mask]
            y_eval = y_target[eval_mask]

            # Few-shot conditions
            for n_ctx in [50, 100]:
                sampling_seed = RANDOM_SEED + n_ctx
                np.random.seed(sampling_seed)
                n_sample = min(n_ctx, len(calib_indices))
                chosen_calib = np.random.choice(calib_indices, size=n_sample, replace=False)

                # ASSERTION: verify against old provenance log
                old_row = old_prov_df[
                    (old_prov_df["Target_Cohort"] == target_name) &
                    (old_prov_df["Context_N"] == n_sample)
                ]
                assert len(old_row) == 1, f"Expected 1 matching old provenance row for {target_name} N={n_sample}, got {len(old_row)}"
                old_indices = eval(old_row.iloc[0]["Sampled_Target_Indices"])
                assert chosen_calib.tolist() == old_indices, (
                    f"FATAL: Sampled context indices do not match old run for {target_name} N={n_sample}!\n"
                    f"New: {chosen_calib.tolist()[:5]}...\nOld: {old_indices[:5]}..."
                )
                log_print(f"  [✓] Verified Context N={n_sample} indices strictly match old run (seed={sampling_seed}).")

                X_calib = X_target_scaled[chosen_calib]
                y_calib = y_target[chosen_calib]

                tabpfn_fs = TabPFNClassifier(
                    n_estimators=4,
                    device="cpu",
                    model_path=str(TABPFN_CHECKPOINT_PATH),
                    ignore_pretraining_limits=True
                )
                tabpfn_fs.fit(X_calib, y_calib)
                p_fs = tabpfn_fs.predict_proba(X_eval_scaled)[:, 1]

                # Record raw predictions on held-out eval subset
                eval_rows = df_target.iloc[eval_indices]
                for idx_in_eval, orig_row_idx in enumerate(eval_indices):
                    raw_pred_records.append({
                        "run_id": "run_20261006_lodo_provenance_fix",
                        "target_cohort": target_name,
                        "subject_id": str(df_target.iloc[orig_row_idx]["subject_id"]),
                        "record_id": str(df_target.iloc[orig_row_idx]["record_id"]),
                        "epoch_idx": int(df_target.iloc[orig_row_idx]["epoch_idx"]),
                        "y_true": int(y_target[orig_row_idx]),
                        "model": "TabPFN",
                        "condition": "few_shot",
                        "context_n": n_sample,
                        "sampling_seed": int(sampling_seed),
                        "y_prob": float(p_fs[idx_in_eval])
                    })

        # Save Full Raw Predictions to Parquet
        raw_df = pd.DataFrame(raw_pred_records)
        raw_df.to_parquet(out_raw_parquet, index=False)
        log_print(f"\n[✓] Saved {len(raw_df)} raw prediction records to {out_raw_parquet}")

        # 2. Compare Unresampled Point Estimates with Old CSV
        log_print("\n" + "=" * 60)
        log_print("VERIFYING ORIGINAL UNRESAMPLED POINT ESTIMATES AGAINST OLD RUN:")
        log_print("=" * 60)

        cmp_records = []
        for _, old_r in old_metrics_df.iterrows():
            tgt = old_r["Target_Cohort"]
            scope = old_r["Evaluation_Scope"]
            m_name = old_r["Model"]
            n_ctx = old_r["Context_N"]

            # Filter raw_df
            if "Few-Shot" in old_r["Paradigm"] or n_ctx > 0:
                sub_preds = raw_df[
                    (raw_df["target_cohort"] == tgt) &
                    (raw_df["condition"] == "few_shot") &
                    (raw_df["context_n"] == n_ctx)
                ]
            else:
                base_m = "TabPFN" if "TabPFN" in m_name else m_name
                sub_preds = raw_df[
                    (raw_df["target_cohort"] == tgt) &
                    (raw_df["model"] == base_m) &
                    (raw_df["condition"] == "zero_shot")
                ]
                if scope == "Common Held-Out Eval Subset":
                    calib_s = "a01" if tgt == "Apnea-ECG" else ("slp01" if tgt == "SLPDB" else "ucddb002")
                    sub_preds = sub_preds[sub_preds["subject_id"] != calib_s]

            assert len(sub_preds) == old_r["N_Eval"], f"Count mismatch for {tgt} {scope} {m_name}: raw={len(sub_preds)}, old={old_r['N_Eval']}"

            y_t = sub_preds["y_true"].values
            y_p = sub_preds["y_prob"].values

            new_auc = roc_auc_score(y_t, y_p)
            new_pr = average_precision_score(y_t, y_p)
            new_brier = brier_score_loss(y_t, y_p)

            diff_auc = abs(new_auc - old_r["ROC_AUC"])
            diff_pr = abs(new_pr - old_r["PR_AUC"])
            diff_brier = abs(new_brier - old_r["Brier_Score"])

            cmp_records.append({
                "Target_Cohort": tgt,
                "Evaluation_Scope": scope,
                "Model": m_name,
                "Context_N": n_ctx,
                "N_Eval": len(y_t),
                "Old_ROC_AUC": old_r["ROC_AUC"],
                "New_ROC_AUC": new_auc,
                "Diff_ROC_AUC": diff_auc,
                "Old_PR_AUC": old_r["PR_AUC"],
                "New_PR_AUC": new_pr,
                "Diff_PR_AUC": diff_pr,
                "Old_Brier": old_r["Brier_Score"],
                "New_Brier": new_brier,
                "Diff_Brier": diff_brier
            })
            log_print(f"  {tgt:10s} | {scope[:18]:18s} | {m_name:20s} | Old-AUC: {old_r['ROC_AUC']:.4f} | New-AUC: {new_auc:.4f} | Diff: {diff_auc:.2e}")

        cmp_df = pd.DataFrame(cmp_records)
        cmp_df.to_csv(out_old_cmp_csv, index=False)
        log_print(f"[✓] Point estimates comparison saved to {out_old_cmp_csv}. Max AUC diff: {cmp_df['Diff_ROC_AUC'].max():.2e}")

        # 3. Patient-Clustered Bootstrap & Paired Differences
        log_print("\n" + "=" * 60)
        log_print("PATIENT-CLUSTERED BOOTSTRAP (1,000 RESAMPLES PER COHORT):")
        log_print("=" * 60)

        n_bootstraps = 1000
        boot_rng = np.random.RandomState(42)

        clustered_metrics_records = []
        paired_diff_records = []
        mc_paired_records = []
        manifest_records = []

        # Target definitions for bootstrap
        eval_targets = [
            {
                "cohort": "UCDDB",
                "scope": "Common Held-Out Eval Subset",
                "calib_subj": "ucddb002",
                "expected_n_subjects": 6,
                "can_cluster": True
            },
            {
                "cohort": "Apnea-ECG",
                "scope": "Common Held-Out Eval Subset",
                "calib_subj": "a01",
                "expected_n_subjects": 33,
                "can_cluster": True
            },
            {
                "cohort": "SLPDB",
                "scope": "Common Held-Out Eval Subset",
                "calib_subj": "slp01",
                "expected_n_subjects": 1,
                "can_cluster": False,
                "reason": "Only 1 held-out test subject (slp02). Inter-subject generalization variance is mathematically unestimable."
            }
        ]

        for target_cfg in eval_targets:
            tgt = target_cfg["cohort"]
            calib_s = target_cfg["calib_subj"]
            can_cluster = target_cfg["can_cluster"]

            sub_target_df = raw_df[
                (raw_df["target_cohort"] == tgt) &
                (raw_df["subject_id"] != calib_s)
            ]

            unique_subjects = sorted(list(sub_target_df["subject_id"].unique()))
            log_print(f"\n--- Cohort: {tgt} (Held-Out Test Subjects: {len(unique_subjects)} {unique_subjects}) ---")

            if not can_cluster:
                log_print(f"  [!] {target_cfg['reason']}")
                # Only record point estimates; mark CIs as UNESTIMABLE
                for cond_key in [("zero_shot", "TabPFN", 0), ("few_shot", "TabPFN", 50), ("few_shot", "TabPFN", 100)]:
                    c_type, m_name, c_n = cond_key
                    pdf = sub_target_df[
                        (sub_target_df["condition"] == c_type) &
                        (sub_target_df["model"] == m_name) &
                        (sub_target_df["context_n"] == c_n)
                    ]
                    if len(pdf) == 0:
                        continue
                    y_t = pdf["y_true"].values
                    y_p = pdf["y_prob"].values
                    auc_pt = roc_auc_score(y_t, y_p)
                    pr_pt = average_precision_score(y_t, y_p)
                    br_pt = brier_score_loss(y_t, y_p)

                    clustered_metrics_records.append({
                        "Target_Cohort": tgt,
                        "Evaluation_Scope": "Common Held-Out Eval Subset",
                        "Model": f"{m_name} (N={c_n})" if c_n > 0 else m_name,
                        "Condition": c_type,
                        "Context_N": c_n,
                        "N_Test_Subjects": len(unique_subjects),
                        "N_Test_Epochs": len(y_t),
                        "Original_ROC_AUC": auc_pt,
                        "Clustered_ROC_AUC_CI_lower": np.nan,
                        "Clustered_ROC_AUC_CI_upper": np.nan,
                        "Original_PR_AUC": pr_pt,
                        "Clustered_PR_AUC_CI_lower": np.nan,
                        "Clustered_PR_AUC_CI_upper": np.nan,
                        "Original_Brier_Score": br_pt,
                        "Clustered_Brier_CI_lower": np.nan,
                        "Clustered_Brier_CI_upper": np.nan,
                        "Bootstrap_Valid_Iterations": 0,
                        "Bootstrap_Invalid_Iterations": 0,
                        "Uncertainty_Status": "Inter-subject CI unestimable (N=1 test subject)"
                    })

                # Paired diff for SLPDB marked as UNESTIMABLE
                for fs_n in [50, 100]:
                    paired_diff_records.append({
                        "Target_Cohort": tgt,
                        "Evaluation_Scope": "Common Held-Out Eval Subset",
                        "Comparison": f"Few-Shot (N={fs_n}) minus Zero-Shot TabPFN",
                        "N_Test_Subjects": 1,
                        "Original_Delta_ROC_AUC": float(
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Context_N']==fs_n), 'New_ROC_AUC'].values[0] -
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Model']=='TabPFN')&(cmp_df['Context_N']==0)&(cmp_df['Evaluation_Scope']=='Common Held-Out Eval Subset'), 'New_ROC_AUC'].values[0]
                        ),
                        "Clustered_Delta_ROC_AUC_CI_lower": np.nan,
                        "Clustered_Delta_ROC_AUC_CI_upper": np.nan,
                        "Original_Delta_PR_AUC": float(
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Context_N']==fs_n), 'New_PR_AUC'].values[0] -
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Model']=='TabPFN')&(cmp_df['Context_N']==0)&(cmp_df['Evaluation_Scope']=='Common Held-Out Eval Subset'), 'New_PR_AUC'].values[0]
                        ),
                        "Clustered_Delta_PR_AUC_CI_lower": np.nan,
                        "Clustered_Delta_PR_AUC_CI_upper": np.nan,
                        "Original_Delta_Brier": float(
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Context_N']==fs_n), 'New_Brier'].values[0] -
                            cmp_df.loc[(cmp_df['Target_Cohort']==tgt)&(cmp_df['Model']=='TabPFN')&(cmp_df['Context_N']==0)&(cmp_df['Evaluation_Scope']=='Common Held-Out Eval Subset'), 'New_Brier'].values[0]
                        ),
                        "Clustered_Delta_Brier_CI_lower": np.nan,
                        "Clustered_Delta_Brier_CI_upper": np.nan,
                        "Uncertainty_Status": "Inter-subject CI unestimable (N=1 test subject)"
                    })
                continue

            # Multi-subject cohort: perform patient-clustered bootstrap
            # Pre-group epochs by subject for fast duplication
            subject_subsets = {s: sub_target_df[sub_target_df["subject_id"] == s] for s in unique_subjects}

            # 1,000 resamples
            boot_seeds = [int(boot_rng.randint(0, 1000000)) for _ in range(n_bootstraps)]
            valid_draws = 0
            invalid_draws = 0
            invalid_reasons = []

            # Containers for bootstrap distribution
            models_to_track = [
                ("zero_shot", "TabPFN", 0),
                ("zero_shot", "LightGBM", 0),
                ("zero_shot", "RandomForest", 0),
                ("zero_shot", "SVM", 0),
                ("zero_shot", "LogisticRegression", 0),
                ("zero_shot", "XGBoost", 0),
                ("few_shot", "TabPFN", 50),
                ("few_shot", "TabPFN", 100),
            ]

            boot_metrics = {k: {"auc": [], "pr": [], "brier": []} for k in models_to_track}
            boot_diffs = {
                50: {"delta_auc": [], "delta_pr": [], "delta_brier": []},
                100: {"delta_auc": [], "delta_pr": [], "delta_brier": []}
            }

            for b_idx, s_seed in enumerate(boot_seeds):
                b_rng = np.random.RandomState(s_seed)
                # Resample subjects WITH REPLACEMENT
                sampled_subjects = b_rng.choice(unique_subjects, size=len(unique_subjects), replace=True)

                # Concatenate all rows, duplicating subjects chosen multiple times
                sampled_dfs = [subject_subsets[s] for s in sampled_subjects]
                b_df = pd.concat(sampled_dfs, ignore_index=True)

                # Check if sample contains both classes
                # Check for TabPFN zero-shot
                sample_zs_tab = b_df[(b_df["condition"] == "zero_shot") & (b_df["model"] == "TabPFN")]
                y_b = sample_zs_tab["y_true"].values
                if len(np.unique(y_b)) < 2:
                    invalid_draws += 1
                    invalid_reasons.append(f"Iteration {b_idx}: single class present ({np.unique(y_b)})")
                    continue

                valid_draws += 1

                # Calculate metrics for all models
                iter_res = {}
                for cond_key in models_to_track:
                    c_type, m_name, c_n = cond_key
                    sub_m = b_df[
                        (b_df["condition"] == c_type) &
                        (b_df["model"] == m_name) &
                        (b_df["context_n"] == c_n)
                    ]
                    if len(sub_m) == 0:
                        continue
                    y_true_m = sub_m["y_true"].values
                    y_prob_m = sub_m["y_prob"].values

                    auc_val = roc_auc_score(y_true_m, y_prob_m)
                    pr_val = average_precision_score(y_true_m, y_prob_m)
                    brier_val = brier_score_loss(y_true_m, y_prob_m)

                    boot_metrics[cond_key]["auc"].append(auc_val)
                    boot_metrics[cond_key]["pr"].append(pr_val)
                    boot_metrics[cond_key]["brier"].append(brier_val)
                    iter_res[cond_key] = {"auc": auc_val, "pr": pr_val, "brier": brier_val}

                # Paired differences on exact same sample: Few-Shot minus Zero-Shot TabPFN
                zs_tab = iter_res[("zero_shot", "TabPFN", 0)]
                for fs_n in [50, 100]:
                    fs_key = ("few_shot", "TabPFN", fs_n)
                    if fs_key in iter_res:
                        fs_val = iter_res[fs_key]
                        boot_diffs[fs_n]["delta_auc"].append(fs_val["auc"] - zs_tab["auc"])
                        boot_diffs[fs_n]["delta_pr"].append(fs_val["pr"] - zs_tab["pr"])
                        boot_diffs[fs_n]["delta_brier"].append(fs_val["brier"] - zs_tab["brier"])

            manifest_records.append({
                "Target_Cohort": tgt,
                "N_Subjects": len(unique_subjects),
                "Total_Resamples": n_bootstraps,
                "Valid_Resamples": valid_draws,
                "Invalid_Resamples": invalid_draws,
                "Invalid_Reasons_Sample": invalid_reasons[:5]
            })

            log_print(f"  [✓] Bootstrap finished: {valid_draws} valid, {invalid_draws} invalid resamples.")

            # Compute Point Estimates & Percentile 95% CIs for Models
            for cond_key in models_to_track:
                c_type, m_name, c_n = cond_key
                sub_orig = sub_target_df[
                    (sub_target_df["condition"] == c_type) &
                    (sub_target_df["model"] == m_name) &
                    (sub_target_df["context_n"] == c_n)
                ]
                if len(sub_orig) == 0:
                    continue
                y_orig = sub_orig["y_true"].values
                p_orig = sub_orig["y_prob"].values

                pt_auc = roc_auc_score(y_orig, p_orig)
                pt_pr = average_precision_score(y_orig, p_orig)
                pt_br = brier_score_loss(y_orig, p_orig)

                auc_dist = boot_metrics[cond_key]["auc"]
                pr_dist = boot_metrics[cond_key]["pr"]
                br_dist = boot_metrics[cond_key]["brier"]

                ci_auc = np.percentile(auc_dist, [2.5, 97.5]) if len(auc_dist) > 0 else (np.nan, np.nan)
                ci_pr = np.percentile(pr_dist, [2.5, 97.5]) if len(pr_dist) > 0 else (np.nan, np.nan)
                ci_br = np.percentile(br_dist, [2.5, 97.5]) if len(br_dist) > 0 else (np.nan, np.nan)

                m_label = f"{m_name} (N={c_n})" if c_n > 0 else m_name
                clustered_metrics_records.append({
                    "Target_Cohort": tgt,
                    "Evaluation_Scope": "Common Held-Out Eval Subset",
                    "Model": m_label,
                    "Condition": c_type,
                    "Context_N": c_n,
                    "N_Test_Subjects": len(unique_subjects),
                    "N_Test_Epochs": len(y_orig),
                    "Original_ROC_AUC": pt_auc,
                    "Clustered_ROC_AUC_CI_lower": ci_auc[0],
                    "Clustered_ROC_AUC_CI_upper": ci_auc[1],
                    "Original_PR_AUC": pt_pr,
                    "Clustered_PR_AUC_CI_lower": ci_pr[0],
                    "Clustered_PR_AUC_CI_upper": ci_pr[1],
                    "Original_Brier_Score": pt_br,
                    "Clustered_Brier_CI_lower": ci_br[0],
                    "Clustered_Brier_CI_upper": ci_br[1],
                    "Bootstrap_Valid_Iterations": valid_draws,
                    "Bootstrap_Invalid_Iterations": invalid_draws,
                    "Uncertainty_Status": f"Patient-clustered 95% CI ({len(unique_subjects)} clusters)"
                })
                log_print(f"  {m_label:22s} | AUC: {pt_auc:.4f} [{ci_auc[0]:.4f} - {ci_auc[1]:.4f}] | PR: {pt_pr:.4f} [{ci_pr[0]:.4f} - {ci_pr[1]:.4f}] | Br: {pt_br:.4f} [{ci_br[0]:.4f} - {ci_br[1]:.4f}]")

            # Paired Differences for Few-Shot vs Zero-Shot TabPFN
            orig_zs_tab = sub_target_df[
                (sub_target_df["condition"] == "zero_shot") &
                (sub_target_df["model"] == "TabPFN")
            ]
            y_zs = orig_zs_tab["y_true"].values
            p_zs = orig_zs_tab["y_prob"].values
            pt_auc_zs = roc_auc_score(y_zs, p_zs)
            pt_pr_zs = average_precision_score(y_zs, p_zs)
            pt_br_zs = brier_score_loss(y_zs, p_zs)

            for fs_n in [50, 100]:
                orig_fs_tab = sub_target_df[
                    (sub_target_df["condition"] == "few_shot") &
                    (sub_target_df["model"] == "TabPFN") &
                    (sub_target_df["context_n"] == fs_n)
                ]
                if len(orig_fs_tab) == 0:
                    continue
                y_fs = orig_fs_tab["y_true"].values
                p_fs_orig = orig_fs_tab["y_prob"].values

                pt_delta_auc = roc_auc_score(y_fs, p_fs_orig) - pt_auc_zs
                pt_delta_pr = average_precision_score(y_fs, p_fs_orig) - pt_pr_zs
                pt_delta_br = brier_score_loss(y_fs, p_fs_orig) - pt_br_zs

                d_auc_dist = boot_diffs[fs_n]["delta_auc"]
                d_pr_dist = boot_diffs[fs_n]["delta_pr"]
                d_br_dist = boot_diffs[fs_n]["delta_brier"]

                ci_d_auc = np.percentile(d_auc_dist, [2.5, 97.5])
                ci_d_pr = np.percentile(d_pr_dist, [2.5, 97.5])
                ci_d_br = np.percentile(d_br_dist, [2.5, 97.5])

                paired_diff_records.append({
                    "Target_Cohort": tgt,
                    "Evaluation_Scope": "Common Held-Out Eval Subset",
                    "Comparison": f"Few-Shot (N={fs_n}) minus Zero-Shot TabPFN",
                    "N_Test_Subjects": len(unique_subjects),
                    "Original_Delta_ROC_AUC": pt_delta_auc,
                    "Clustered_Delta_ROC_AUC_CI_lower": ci_d_auc[0],
                    "Clustered_Delta_ROC_AUC_CI_upper": ci_d_auc[1],
                    "Original_Delta_PR_AUC": pt_delta_pr,
                    "Clustered_Delta_PR_AUC_CI_lower": ci_d_pr[0],
                    "Clustered_Delta_PR_AUC_CI_upper": ci_d_pr[1],
                    "Original_Delta_Brier": pt_delta_br,
                    "Clustered_Delta_Brier_CI_lower": ci_d_br[0],
                    "Clustered_Delta_Brier_CI_upper": ci_d_br[1],
                    "Uncertainty_Status": f"Paired patient-clustered 95% CI ({len(unique_subjects)} clusters)"
                })
                log_print(f"  PAIRED DELTA (N={fs_n:3d}) | ΔAUC: {pt_delta_auc:+.4f} [{ci_d_auc[0]:+.4f} to {ci_d_auc[1]:+.4f}] | ΔPR: {pt_delta_pr:+.4f} [{ci_d_pr[0]:+.4f} to {ci_d_pr[1]:+.4f}] | ΔBr: {pt_delta_br:+.4f} [{ci_d_br[0]:+.4f} to {ci_d_br[1]:+.4f}]")

            # Multi-comparison test: each model vs TabPFN zero-shot on identical
            # patient-clustered bootstrap resamples, Holm-corrected.
            zs_key = ("zero_shot", "TabPFN", 0)
            zs_auc_dist = boot_metrics[zs_key]["auc"]
            if len(zs_auc_dist) > 0:
                zs_sub = sub_target_df[
                    (sub_target_df["condition"] == "zero_shot") &
                    (sub_target_df["model"] == "TabPFN")
                ]
                zs_point_auc = roc_auc_score(zs_sub["y_true"].values, zs_sub["y_prob"].values)
                comp_records = []
                for cond_key in models_to_track:
                    c_type, m_name, c_n = cond_key
                    if cond_key == zs_key:
                        continue
                    m_dist = boot_metrics[cond_key]["auc"]
                    if len(m_dist) != len(zs_auc_dist) or len(m_dist) == 0:
                        continue
                    deltas = np.array(zs_auc_dist) - np.array(m_dist)
                    p_raw = 2.0 * min(np.mean(deltas > 0), np.mean(deltas < 0))
                    p_raw = min(max(p_raw, 1.0 / max(len(deltas), 1)), 1.0)
                    sub_orig_m = sub_target_df[
                        (sub_target_df["condition"] == c_type) &
                        (sub_target_df["model"] == m_name) &
                        (sub_target_df["context_n"] == c_n)
                    ]
                    delta_point = zs_point_auc - roc_auc_score(
                        sub_orig_m["y_true"].values, sub_orig_m["y_prob"].values
                    )
                    comp_records.append({
                        "Model": f"{m_name} (N={c_n})" if c_n > 0 else m_name,
                        "TabPFN_minus_Model_ROC_AUC": delta_point,
                        "p_raw": p_raw,
                    })
                comp_records.sort(key=lambda r: r["p_raw"])
                m_tests = len(comp_records)
                running_max = 0.0
                for rank, rec in enumerate(comp_records):
                    # Holm step-down: p-values are enforced non-decreasing
                    running_max = max(running_max, min(1.0, (m_tests - rank) * rec["p_raw"]))
                    rec["p_holm"] = running_max
                    rec["significant_at_0.05"] = bool(rec["p_holm"] < 0.05)
                    rec["Target_Cohort"] = tgt
                    rec["N_Test_Subjects"] = len(unique_subjects)
                    rec["N_Bootstrap_Paired_Resamples"] = len(zs_auc_dist)
                    mc_paired_records.append(rec)
                    log_print(f"  MC {rec['Model']:22s} | TabPFN-Model dAUC: {rec['TabPFN_minus_Model_ROC_AUC']:+.4f} | p_holm: {rec['p_holm']:.3f} | sig: {rec['significant_at_0.05']}")

        # Save all metrics
        pd.DataFrame(clustered_metrics_records).to_csv(out_clustered_csv, index=False)
        pd.DataFrame(paired_diff_records).to_csv(out_paired_diff_csv, index=False)
        pd.DataFrame(mc_paired_records).to_csv(out_mc_csv, index=False)
        with open(out_boot_manifest, "w") as f:
            json.dump(manifest_records, f, indent=2)

        log_print(f"\n[✓] All outputs saved successfully!")
        log_print(f"  - Metrics: {out_clustered_csv}")
        log_print(f"  - Paired Differences: {out_paired_diff_csv}")
        log_print(f"  - Multiple Comparison (Holm): {out_mc_csv}")
        log_print(f"  - Bootstrap Manifest: {out_boot_manifest}")

if __name__ == "__main__":
    run()
