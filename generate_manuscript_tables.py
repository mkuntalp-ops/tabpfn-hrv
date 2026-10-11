"""
Generate manuscript tables from the regenerated study outputs.

Reads the outputs of reproduce_all_results.sh and writes publication-ready
CSV/markdown tables to results/manuscript_tables/:

  Table 1  Development cohort: patient-independent GroupKFold performance
           (results/full_development_study/group_cv_summary.csv)
  Table 2  Development cohort: subject-level diagnostic performance
           (results/full_development_study/subject_level_metrics.csv)
  Table 3  Tri-center LODO: external performance with patient-clustered 95% CIs
           (results/runs/run_20261006_lodo_provenance_fix/metrics/
            lodo_patient_clustered_metrics.csv)
  Table 4  Paired few-shot minus zero-shot TabPFN deltas
           (.../lodo_paired_differences.csv)
  Table 5  Holm-corrected model comparisons vs TabPFN zero-shot
           (.../lodo_multiple_comparison_holm.csv)

Usage:
    python generate_manuscript_tables.py \
        [--results-root results] \
        [--out-dir results/manuscript_tables]

Missing inputs are reported (not fatal) so the script can be run after any
subset of the studies completes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("results")


def _fmt_ci(point, lo, hi, digits=3):
    if pd.isna(lo) or pd.isna(hi):
        return f"{point:.{digits}f} (CI unestimable)"
    return f"{point:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def _read(path: Path, multiindex: bool = False):
    if not path.exists():
        print(f"[skip] {path} not found (run the corresponding study first)")
        return None
    if multiindex:
        return pd.read_csv(path, header=[0, 1], index_col=0)
    return pd.read_csv(path)


def _save(df: pd.DataFrame, out_dir: Path, stem: str, title: str):
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    with open(out_dir / f"{stem}.md", "w") as f:
        f.write(f"### {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"[ok]   {stem}: {len(df)} rows")


def table1_dev_cv(root: Path, out_dir: Path):
    p = root / "full_development_study/group_cv_summary.csv"
    df = _read(p, multiindex=True)
    if df is None:
        return
    rows = []
    for model in df.index:
        r = df.loc[model]
        rows.append({
            "Model": model,
            "ROC-AUC (mean +/- SD)": f"{r[('ROC-AUC', 'mean')]:.3f} +/- {r[('ROC-AUC', 'std')]:.3f}",
            "PR-AUC (mean +/- SD)": f"{r[('PR-AUC', 'mean')]:.3f} +/- {r[('PR-AUC', 'std')]:.3f}",
            "Brier (mean +/- SD)": f"{r[('BrierScore', 'mean')]:.3f} +/- {r[('BrierScore', 'std')]:.3f}",
            "ECE (mean +/- SD)": f"{r[('ECE', 'mean')]:.3f} +/- {r[('ECE', 'std')]:.3f}",
            "Accuracy (mean +/- SD)": f"{r[('Accuracy', 'mean')]:.3f} +/- {r[('Accuracy', 'std')]:.3f}",
            "F1 (mean +/- SD)": f"{r[('F1-Score', 'mean')]:.3f} +/- {r[('F1-Score', 'std')]:.3f}",
        })
    _save(pd.DataFrame(rows), out_dir, "table1_development_cv",
          "Table 1. Development cohort: patient-independent GroupKFold performance (mean +/- SD across folds)")


def table2_subject_level(root: Path, out_dir: Path):
    p = root / "full_development_study/subject_level_metrics.csv"
    df = _read(p)
    if df is None:
        return
    df = df[[
        "Model", "Subject_Accuracy", "Subject_Sensitivity", "Subject_Specificity",
        "Pearson_r_Apnea_Minutes", "Pearson_r_Apnea_Burden_per_Hour",
        "TP", "TN", "FP", "FN",
    ]].copy()
    for c in ["Subject_Accuracy", "Subject_Sensitivity", "Subject_Specificity"]:
        df[c] = df[c].map(lambda v: f"{v:.3f}")
    for c in ["Pearson_r_Apnea_Minutes", "Pearson_r_Apnea_Burden_per_Hour"]:
        df[c] = df[c].map(lambda v: "NA" if pd.isna(v) else f"{v:.3f}")
    _save(df, out_dir, "table2_subject_level",
          "Table 2. Development cohort: subject-level diagnostic performance "
          "(LOSO-selected Youden thresholds; OSA rule: >=5 apnea min per recorded hour)")


def table3_lodo_clustered(root: Path, out_dir: Path):
    p = root / "runs/run_20261006_lodo_provenance_fix/metrics/lodo_patient_clustered_metrics.csv"
    df = _read(p)
    if df is None:
        return
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Target cohort": r["Target_Cohort"],
            "Model": r["Model"],
            "N test subjects": r["N_Test_Subjects"],
            "N test epochs": r["N_Test_Epochs"],
            "ROC-AUC (95% CI)": _fmt_ci(r["Original_ROC_AUC"],
                                        r["Clustered_ROC_AUC_CI_lower"],
                                        r["Clustered_ROC_AUC_CI_upper"]),
            "PR-AUC (95% CI)": _fmt_ci(r["Original_PR_AUC"],
                                       r["Clustered_PR_AUC_CI_lower"],
                                       r["Clustered_PR_AUC_CI_upper"]),
            "Brier (95% CI)": _fmt_ci(r["Original_Brier_Score"],
                                       r["Clustered_Brier_CI_lower"],
                                       r["Clustered_Brier_CI_upper"]),
            "Uncertainty": r["Uncertainty_Status"],
        })
    _save(pd.DataFrame(rows), out_dir, "table3_lodo_clustered",
          "Table 3. Tri-center LODO: external performance with patient-clustered 95% CIs")


def table4_paired_few_shot(root: Path, out_dir: Path):
    p = root / "runs/run_20261006_lodo_provenance_fix/metrics/lodo_paired_differences.csv"
    df = _read(p)
    if df is None:
        return
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Target cohort": r["Target_Cohort"],
            "Comparison": r["Comparison"],
            "N test subjects": r["N_Test_Subjects"],
            "Delta ROC-AUC (95% CI)": _fmt_ci(r["Original_Delta_ROC_AUC"],
                                              r["Clustered_Delta_ROC_AUC_CI_lower"],
                                              r["Clustered_Delta_ROC_AUC_CI_upper"]),
            "Delta PR-AUC (95% CI)": _fmt_ci(r["Original_Delta_PR_AUC"],
                                             r["Clustered_Delta_PR_AUC_CI_lower"],
                                             r["Clustered_Delta_PR_AUC_CI_upper"]),
            "Delta Brier (95% CI)": _fmt_ci(r["Original_Delta_Brier"],
                                            r["Clustered_Delta_Brier_CI_lower"],
                                            r["Clustered_Delta_Brier_CI_upper"]),
            "Uncertainty": r["Uncertainty_Status"],
        })
    _save(pd.DataFrame(rows), out_dir, "table4_paired_few_shot",
          "Table 4. Paired few-shot minus zero-shot TabPFN deltas (patient-clustered 95% CIs)")


def table5_holm(root: Path, out_dir: Path):
    p = root / "runs/run_20261006_lodo_provenance_fix/metrics/lodo_multiple_comparison_holm.csv"
    df = _read(p)
    if df is None:
        return
    df = df[[
        "Target_Cohort", "Model", "TabPFN_minus_Model_ROC_AUC",
        "p_raw", "p_holm", "significant_at_0.05",
        "N_Test_Subjects", "N_Bootstrap_Paired_Resamples",
    ]].copy()
    df["TabPFN_minus_Model_ROC_AUC"] = df["TabPFN_minus_Model_ROC_AUC"].map(lambda v: f"{v:+.3f}")
    df["p_raw"] = df["p_raw"].map(lambda v: f"{v:.3f}")
    df["p_holm"] = df["p_holm"].map(lambda v: f"{v:.3f}")
    _save(df, out_dir, "table5_holm_comparisons",
          "Table 5. Holm-corrected comparisons vs TabPFN zero-shot "
          "(identical patient-clustered bootstrap resamples)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    root = args.results_root
    out_dir = args.out_dir or (root / "manuscript_tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    table1_dev_cv(root, out_dir)
    table2_subject_level(root, out_dir)
    table3_lodo_clustered(root, out_dir)
    table4_paired_few_shot(root, out_dir)
    table5_holm(root, out_dir)
    print(f"\nManuscript tables written to: {out_dir}")


if __name__ == "__main__":
    main()
