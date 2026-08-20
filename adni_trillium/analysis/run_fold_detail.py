"""Per-fold training and test performance, and out-of-fold risk scores.

run_spec_models.py records only the mean and standard deviation of the test
C-index. Two figures need more than that: a box plot of the fold-level
distribution including the train-minus-test gap, which is where overfitting
shows up, and Kaplan-Meier curves stratified by predicted risk, which need a
risk score for every subject from a model that did not see them.

Both come from the same loop, so they are computed together here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_spec_models import SEED, cindex, feature_sets, fit_predict, preprocess, surv_y

# (label, model, feature set). Chosen to span the paper's argument: clinical
# covariates, each imaging family on its own, and the fullest combined model.
CONFIGS = [
    ("Cov. (Weibull)", "aft_weibull", "F0_covariates"),
    ("Cov. (Cox L2)", "cox_l2", "F0_covariates"),
    ("BSC global (RSF)", "rsf", "F2_bsc_slopes"),
    ("BSC regional (RSF)", "rsf", "F3_regional_slopes"),
    ("Thickness+hipp. (RSF)", "rsf", "F5_std_mri_slopes"),
    ("Cov.+BSC+MRI (XGB)", "xgb_aft", "F9b_cov_regional_stdmri"),
]

# Models whose out-of-fold risk scores feed the Kaplan-Meier figure.
KM_CONFIGS = [
    ("Clinical covariates", "aft_weibull", "F0_covariates"),
    ("Regional BSC slopes", "rsf", "F3_regional_slopes"),
]


def run(df, model, cols):
    """Per-fold train and test C-index, plus out-of-fold risk for each subject."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds, oof = [], []
    for k, (a, b) in enumerate(skf.split(df, df["event"])):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        ytr, yte = surv_y(tr), surv_y(te)
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        rtr, rte = rtr * flip, rte * flip
        folds.append({"fold": k, "train": cindex(ytr, rtr), "test": cindex(yte, rte)})
        oof.append(pd.DataFrame({"subject": te["subject"].to_numpy(),
                                 "event": te["event"].to_numpy(),
                                 "time_years": te["time_years"].to_numpy(),
                                 "risk": rte, "fold": k}))
    return pd.DataFrame(folds), pd.concat(oof, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/spec_v3_harmonized"))
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)
    sets = feature_sets(df)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, model, fname in CONFIGS:
        cols = [c for c in sets[fname] if c in df.columns]
        f, _ = run(df, model, cols)
        f["config"] = label.replace("\n", " ")
        f["label"] = label
        f["model"], f["feature_set"], f["n_features"] = model, fname, len(cols)
        f["gap"] = f["train"] - f["test"]
        rows.append(f)
        print(f"  {label.replace(chr(10), ' '):32s} train {f.train.mean():.3f}  "
              f"test {f.test.mean():.3f}  gap {f.gap.mean():+.3f}")
    pd.concat(rows, ignore_index=True).to_csv(out / "fold_detail.csv", index=False)

    km = []
    for label, model, fname in KM_CONFIGS:
        cols = [c for c in sets[fname] if c in df.columns]
        _, o = run(df, model, cols)
        o["config"] = label
        km.append(o)
        print(f"  out-of-fold risk written for {label} ({len(o)} subjects)")
    pd.concat(km, ignore_index=True).to_csv(out / "oof_risk.csv", index=False)
    print(f"wrote {out/'fold_detail.csv'} and {out/'oof_risk.csv'}")


if __name__ == "__main__":
    main()
