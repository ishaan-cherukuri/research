"""Design comparison: the submitted follow-up definition against the corrected one.

This is the central result of the revision. The two cohorts are built from the
same scans, the same features and the same models. Only the follow-up definition
differs:

  legacy     features from every scan including any acquired after the dementia
             diagnosis, clock starting at the MCI baseline scan
  corrected  features only from scans acquired while the subject was still MCI,
             clock starting at the last such scan

Restricting to the subjects present in both isolates the design effect from the
change in cohort composition, so the comparison cannot be dismissed as one
cohort simply being easier than the other. If BSC slopes lose discrimination
under the corrected design while clinical covariates hold steady, the prognostic
value was coming from the design rather than from the biomarker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_spec_models import SEED, cindex, feature_sets, fit_predict, preprocess, surv_y


def cv_cindex(df, cols, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    out = []
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        ytr, yte = surv_y(tr), surv_y(te)
        try:
            rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        except Exception as e:
            print(f"  [fail] {model}: {e}")
            continue
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        out.append(cindex(yte, rte * flip))
    return (float(np.mean(out)), float(np.std(out, ddof=1))) if out else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--legacy", default=str(root / "results/spec_v3_legacy/spec_cohort.csv"))
    ap.add_argument("--corrected", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/spec_v3_harmonized"))
    ap.add_argument("--models", default="rsf,xgb_aft,cox_l2")
    args = ap.parse_args()

    frames = {}
    for tag, path in (("legacy", args.legacy), ("corrected", args.corrected)):
        d = pd.read_csv(path)
        fs = pd.to_numeric(d.get("field_strength_bl"), errors="coerce")
        d["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)
        frames[tag] = d

    shared = sorted(set(frames["legacy"].subject) & set(frames["corrected"].subject))
    print(f"legacy {len(frames['legacy'])} subjects, corrected {len(frames['corrected'])}, "
          f"shared {len(shared)}")

    rows = []
    for tag in ("legacy", "corrected"):
        d = frames[tag][frames[tag].subject.isin(shared)].reset_index(drop=True)
        sets = feature_sets(d)
        ev = int(d.event.sum())
        print(f"\n{tag}: {len(d)} subjects, {ev} events, "
              f"median follow-up {d.time_years.median():.2f} y")
        for fname in ("F0_covariates", "F2_bsc_slopes", "F3_regional_slopes",
                      "F5_std_mri_slopes"):
            cols = [c for c in sets.get(fname, []) if c in d.columns]
            if not cols:
                continue
            for m in args.models.split(","):
                mean, sd = cv_cindex(d, cols, m)
                rows.append({"design": tag, "feature_set": fname, "model": m,
                             "n_subjects": len(d), "n_events": ev,
                             "cv_cindex_mean": mean, "cv_cindex_sd": sd})
                print(f"  {fname:22s} {m:8s} {mean:.3f} +- {sd:.3f}")

    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["feature_set", "model"], columns="design",
                          values="cv_cindex_mean")
    piv["delta"] = piv["corrected"] - piv["legacy"]
    print("\n=== design effect (corrected minus legacy) ===")
    print(piv.round(3).to_string())

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.out_dir) / "design_comparison.csv", index=False)
    piv.round(4).to_csv(Path(args.out_dir) / "design_comparison_pivot.csv")
    with open(Path(args.out_dir) / "design_comparison.json", "w") as f:
        json.dump({"n_shared": len(shared), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
