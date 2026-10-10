# Revision Notes — Framing the Manuscript for Resubmission

This document translates the code review findings into concrete manuscript
changes. Use it as a checklist when revising the rejected submission.

## 1. Claims that the current data cannot support

The committed results (LODO patient-clustered bootstrap) show:

- **Few-shot adaptation provides no significant benefit.** Every paired
  few-shot minus zero-shot TabPFN delta has a 95% CI crossing zero
  (`lodo_paired_differences.csv`), in every cohort.
- **External test sets are underpowered.** UCDDB held-out evaluation uses
  6 subjects (ROC-AUC CI spans roughly 0.50–0.85); SLPDB has a single
  held-out subject, making inter-subject CIs mathematically unestimable.
- Consequently, any claim of the form "TabPFN few-shot adaptation improves
  external performance" or "multi-center validation demonstrates
  generalization" is not supported.

**Required rewording (Abstract / Results / Discussion):**
- Replace superiority claims with neutral statements of point estimates
  with their clustered CIs.
- Add an explicit *Exploratory / hypothesis-generating* framing sentence in
  the Abstract and at the start of the Discussion.
- In the Limitations section, state: single effectively-powered external
  cohort (Apnea-ECG, 33 held-out subjects), 6-subject UCDDB subset,
  1-subject SLPDB subset, and that no few-shot improvement reached
  significance.

## 2. Corrections that change previously reported numbers

The following defects were fixed in the analysis code; **all previously
reported numbers must be regenerated** with `bash reproduce_all_results.sh`
before resubmission:

1. **Preprocessing leakage** (affects all previously reported metrics):
   `SimpleImputer` and `StandardScaler` were fit on the pooled dataset
   (including test folds / test cohorts). They are now fit on training data
   only. Expect slightly lower, more honest point estimates.
2. **Subject-level rule**: the OSA-positive rule was "≥5 apnea minutes per
   night"; it is now duration-normalized (≥5 apnea minutes per recorded
   hour, an AHI≥5 analogue). Subject-level sensitivity/specificity numbers
   will change.
3. **Youden threshold optimism**: thresholds were previously optimized and
   evaluated on the same 35 subjects. They are now selected
   leave-one-subject-out. Subject-level accuracy will decrease.
4. **DCA**: previously computed on a single fold; now pooled across all CV
   folds with dispersion.
5. **ECE**: the last calibration bin previously dropped predictions with
   p = 1.0; corrected. ECE values may shift slightly.

If the journal requires a change log, report these as errata with old vs
new values side by side (extend `metrics_errata_log.md`).

## 3. Statistics section additions

- Add the **Holm-corrected multi-comparison results**
  (`lodo_multiple_comparison_holm.csv`): each model vs TabPFN zero-shot on
  identical patient-clustered bootstrap resamples, within each cohort.
- Keep the existing patient-clustered bootstrap CIs and the paired few-shot
  delta CIs; report them explicitly as the primary uncertainty quantification.
- State that no multiple-comparison correction was applied in the original
  submission, and that the Holm-corrected analysis is now the primary one.

## 4. Methods / reproducibility section additions

- Reference the exact primary feature list (`configs/protocol.py`), the
  fixed random seed, and the TabPFN checkpoint version/size.
- Add the zero-leakage preprocessing statement: "imputation and feature
  scaling were fit exclusively on training data within each
  cross-validation fold / on source cohorts in each leave-one-database-out
  fold".
- State that SLPDB records sharing a subject prefix belong to one subject
  (`SLPDB_SUBJECT_MAP`), and that subject isolation was verified by
  assertion at runtime.
- Add the TRIPOD+AI checklist mapping if the target journal requires it
  (the DCA, calibration, and subject-level analyses already cover the
  key items).

## 5. Submission checklist

- [ ] Regenerate all numbers: `bash reproduce_all_results.sh`
- [ ] Align `configs/protocol.py::PRIMARY_FEATURES` with the manuscript
      feature table (the script now self-checks this against the data
      schema and aborts on mismatch)
- [ ] Update every table/figure with regenerated values
- [ ] Rewrite claims per Section 1 (exploratory framing)
- [ ] Add limitations text per Section 1
- [ ] Add statistics additions per Section 3
- [ ] Add methods additions per Section 4
- [ ] Update `metrics_errata_log.md` with old-vs-new values if requested
- [ ] Confirm the cover letter mentions the post-review analysis fixes
      (leakage, subject-level rule, multiplicity correction)
