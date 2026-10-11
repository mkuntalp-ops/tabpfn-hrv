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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.train_and_evaluate import get_baseline_models, get_tabpfn_model

warnings.filterwarnings("ignore")

SAMPLE_SIZES = [20, 50, 100, 200, 500]

def run_label_efficiency_benchmark(
    data_path: str = "data/processed/hrv_features_apnea_ecg_full.parquet",
    output_dir: str = "results/label_efficiency",
    sample_sizes: list = SAMPLE_SIZES,
    n_seeds: int = 5
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading dataset: {data_path}")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.columns if c.startswith("HRV_") or c == "num_r_peaks"]
    
    X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan).values
    y = df["apnea_label"].values.astype(int)
    groups = df["record_id"].values

    # Strict Patient-Independent Split (using Fold 1 of a 5-fold CV)
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X_raw, y, groups=groups))
    
    # Verify zero leakage
    train_patients = set(groups[train_idx])
    test_patients = set(groups[test_idx])
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage detected!"

    print(f"[+] Total Train Pool: {len(train_idx)} epochs ({len(train_patients)} patients)")
    print(f"[+] Fixed Held-out Test Set: {len(test_idx)} epochs ({len(test_patients)} patients)")

    # ZERO LEAKAGE: Fit Imputer ONLY on Train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_raw[train_idx])
    X_test_imp = imputer.transform(X_raw[test_idx])

    # ZERO LEAKAGE: Fit Scaler ONLY on Train
    scaler = StandardScaler()
    X_pool_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    y_pool = y[train_idx]
    y_test = y[test_idx]

    records = []

    for n in sample_sizes:
        print(f"\n" + "="*50)
        print(f"[*] Evaluating Sample Size N = {n} ({n_seeds} random subsamples)")
        print("="*50)

        for seed in range(n_seeds):
            # Sample n examples from the training pool completely independently
            np.random.seed(seed * 101 + n)
            pos_idx = np.where(y_pool == 1)[0]
            neg_idx = np.where(y_pool == 0)[0]
            
            n_pos = max(1, int(n * (len(pos_idx)/len(y_pool))))
            n_neg = n - n_pos
            
            chosen_pos = np.random.choice(pos_idx, n_pos, replace=False)
            chosen_neg = np.random.choice(neg_idx, n_neg, replace=False)
            
            sub_train_idx = np.concatenate([chosen_pos, chosen_neg])
            np.random.shuffle(sub_train_idx)

            X_sub = X_pool_scaled[sub_train_idx]
            y_sub = y_pool[sub_train_idx]

            baselines = get_baseline_models()

            for name, clf in baselines.items():
                clf.fit(X_sub, y_sub)
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_test_scaled)[:, 1]
                else:
                    # Only rank-based metrics (ROC-AUC) are valid for decision-function scores;
                    # Brier scores from min-max normalized scores are not calibrated probabilities
                    raw = clf.decision_function(X_test_scaled)
                    probs = (raw - raw.min()) / (raw.max() - raw.min())
                    records.append({
                        "Sample_Size": n,
                        "Seed": seed,
                        "Model": name,
                        "ROC-AUC": roc_auc_score(y_test, probs),
                        "BrierScore": np.nan,
                        "Brier_Valid": False
                    })
                    continue

                auc = roc_auc_score(y_test, probs)
                brier = brier_score_loss(y_test, probs)

                records.append({
                    "Sample_Size": n,
                    "Seed": seed,
                    "Model": name,
                    "ROC-AUC": auc,
                    "BrierScore": brier,
                    "Brier_Valid": True
                })

            try:
                t_model = get_tabpfn_model()
                t_model.fit(X_sub, y_sub)
                t_probs = t_model.predict_proba(X_test_scaled)[:, 1]

                auc = roc_auc_score(y_test, t_probs)
                brier = brier_score_loss(y_test, t_probs)

                records.append({
                    "Sample_Size": n,
                    "Seed": seed,
                    "Model": "TabPFN",
                    "ROC-AUC": auc,
                    "BrierScore": brier
                })
            except Exception as e:
                print(f"[-] TabPFN eval failed at N={n}: {e}")

        df_res = pd.DataFrame(records)
        df_sub = df_res[df_res["Sample_Size"] == n].groupby("Model")[["ROC-AUC", "BrierScore"]].mean()
        print(df_sub)

    # Plot
    df_all = pd.DataFrame(records)
    plt.figure(figsize=(10, 6))
    
    palette = {m: "lightgray" for m in df_all["Model"].unique()}
    palette["TabPFN"] = "red"
    palette["LightGBM"] = "blue"
    palette["XGBoost"] = "green"
    
    sns.lineplot(
        data=df_all, x="Sample_Size", y="ROC-AUC", hue="Model", 
        marker="o", palette=palette, linewidth=2.5, err_style="band"
    )
    plt.title("Epoch-Level Foundation Model Label Efficiency\n(Single Patient-Independent Hold-out Split; Exploratory)", fontweight="bold")
    plt.xlabel("Number of Training Epochs ($N$)")
    plt.ylabel("Epoch-Level ROC-AUC on Held-out Patients")
    plt.xscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    chart_file = out_dir / "label_efficiency_curves.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()
    print(f"[+] Saved label efficiency plot to: {chart_file}")

if __name__ == "__main__":
    run_label_efficiency_benchmark()
