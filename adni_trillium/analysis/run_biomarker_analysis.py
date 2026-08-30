"""Biomarker analyses added in response to the Brain Communications decision.

The editor's objection was that conversion in this cohort is a clinical label
with no biological confirmation, so a null result cannot be read as a null about
Alzheimer's disease. This script answers that in four parts, and every part is
additive: the full 417-subject cohort and its existing results are untouched.

  1. Composition. Amyloid status by outcome, for Table 1.
  2. Ceiling. Whether amyloid status adds to the clinical covariate model, and
     whether the BSC increments change once it is in the baseline block. All 417
     subjects are kept, with unknown status carried as a missing indicator.
  3. Sensitivity. The same incremental tests inside the amyloid-positive
     subgroup, where every subject has confirmed Alzheimer's pathology.
  4. Construct validity. Whether BSC slopes track amyloid or tau burden at all.
     Section 4.4 of the manuscript already shows BSC tracks image quality and not
     age; this asks the same question of pathology.

Outputs go to results/biomarker_v1/. Nothing here overwrites an existing result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from run_spec_models import (SEED, COVARIATES, cindex, feature_sets,
                             fit_predict, preprocess, surv_y)
from run_spec_increment import PAIRS, cox_lr_test, paired_folds

# Missing-indicator coding: A+ versus everyone, plus a flag for untested.
# Two columns keep all 417 subjects in the model without imputing a status.
BIO_COLS = ["amyloid_pos_flag", "amyloid_unknown"]

CONSTRUCT_TARGETS = [
    ("pet_CENTILOIDS", "amyloid PET centiloids"),
    ("csf_ABETA42", "CSF Abeta42"),
    ("csf_PTAU", "CSF phosphorylated tau"),
    ("csf_ptau_abeta42_ratio", "CSF p-tau / Abeta42 ratio"),
    ("taupet_META_TEMPORAL_SUVR", "tau PET meta-temporal SUVR"),
]


def add_bio(sets: dict[str, list[str]]) -> dict[str, list[str]]:
    """Every covariate-containing feature set, with amyloid status appended."""
    cov = set(sets["F0_covariates"])
    out = {}
    for name, cols in sets.items():
        if cov.issubset(set(cols)):
            out[name + "_bio"] = cols + BIO_COLS
    return out


def cv_cindex(df, cols, model, n_splits=5):
    """Plain stratified CV C-index, the same protocol as run_spec_models."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        ytr, yte = surv_y(tr), surv_y(te)
        Xtr, Xte = preprocess(tr, te, cols)
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        scores.append(cindex(yte, rte * flip))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def composition(df: pd.DataFrame) -> pd.DataFrame:
    """Amyloid status crossed with outcome, plus per-stratum conversion rate."""
    rows = []
    for lab in ["A+", "A-", "unknown", "all"]:
        s = df if lab == "all" else df[df["amyloid_status_label"] == lab]
        if not len(s):
            continue
        rows.append({
            "amyloid_status": lab,
            "n": len(s),
            "pct_of_cohort": 100.0 * len(s) / len(df),
            "converters": int(s["event"].sum()),
            "stable": int((~s["event"].astype(bool)).sum()),
            "conversion_rate_pct": 100.0 * float(s["event"].mean()),
            "median_followup_years": float(s["time_years"].median()),
            # Era matters: untested subjects are concentrated in ADNI1, before
            # amyloid PET was routine, which is why that stratum looks unusual.
            "median_landmark_year": float(pd.to_datetime(
                s["landmark_date"], errors="coerce").dt.year.median()),
            "pct_3T": 100.0 * float(
                (pd.to_numeric(s["field_strength_bl"], errors="coerce") > 2.25).mean()),
        })
    tab = pd.DataFrame(rows)
    known = df[df["amyloid_status_label"].isin(["A+", "A-"])]
    ct = pd.crosstab(known["amyloid_status_label"], known["event"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    tab.attrs["chi2_p"] = float(p)
    return tab


def increments(df, sets, model, n_boot, rng, pairs, tag):
    rows = []
    for a_name, b_name, label in pairs:
        if a_name not in sets or b_name not in sets:
            continue
        ca, cb, deltas, _ = paired_folds(df, sets[a_name], sets[b_name],
                                         model, n_boot, rng)
        lo, hi = (np.percentile(deltas, [2.5, 97.5]) if len(deltas)
                  else (np.nan, np.nan))
        lr = cox_lr_test(df, sets[a_name], sets[b_name])
        rows.append({
            "analysis": tag, "comparison": label,
            "baseline": a_name, "augmented": b_name, "model": model,
            "n_subjects": len(df), "n_events": int(df["event"].sum()),
            "cv_baseline_mean": float(ca.mean()),
            "cv_augmented_mean": float(cb.mean()),
            "fold_matched_delta": float((cb - ca).mean()),
            "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
            "cox_lr_p": float(lr.get("p", np.nan)),
        })
        print(f"  [{tag}] {label}: {ca.mean():.3f} -> {cb.mean():.3f}  "
              f"delta={rows[-1]['fold_matched_delta']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  cox LR p={lr.get('p', float('nan')):.3g}")
    return rows


def construct_validity(df: pd.DataFrame, slope_cols: list[str]) -> pd.DataFrame:
    """Spearman correlation of each BSC slope with each pathology measure.

    Spearman rather than Pearson because centiloids and the CSF ratio are
    strongly skewed. Benjamini-Hochberg is applied within each target.
    """
    rows = []
    for target, pretty in CONSTRUCT_TARGETS:
        if target not in df.columns:
            continue
        y = pd.to_numeric(df[target], errors="coerce")
        if y.notna().sum() < 50:
            continue
        for c in slope_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() < 50 or x[ok].std() == 0:
                continue
            r, p = stats.spearmanr(x[ok], y[ok])
            rows.append({"target": target, "target_label": pretty,
                         "feature": c, "n": int(ok.sum()),
                         "rho": float(r), "p": float(p)})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    parts = []
    for _, g in out.groupby("target"):
        g = g.sort_values("p").reset_index(drop=True)
        m = len(g)
        raw = g["p"] * m / (g.index + 1)
        # Step-up: the running minimum sweeps from the largest p-value down.
        g["q"] = raw[::-1].cummin()[::-1].clip(upper=1.0)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["target", "p"])


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(root / "results/biomarker_v1/cohort_with_biomarkers.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/biomarker_v1"))
    ap.add_argument("--model", default="xgb_aft")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)

    sets = feature_sets(df)
    sets.update(add_bio(sets))
    rng = np.random.default_rng(SEED)

    # ---- 1. composition ----------------------------------------------------
    print("\n=== 1. Amyloid composition of the cohort ===")
    comp = composition(df)
    print(comp.to_string(index=False))
    print(f"A+ vs A- conversion, chi-square p = {comp.attrs['chi2_p']:.3g}")
    comp.to_csv(out_dir / "table1_amyloid.csv", index=False)

    # ---- 2. does amyloid raise the clinical ceiling ------------------------
    print("\n=== 2. Clinical ceiling with and without amyloid status (all 417) ===")
    ceiling = []
    for name in ["F0_covariates", "F0_covariates_bio", "F8_cov_stdmri",
                 "F8_cov_stdmri_bio"]:
        if name not in sets:
            continue
        m, s = cv_cindex(df, sets[name], args.model)
        ceiling.append({"feature_set": name, "n_features": len(sets[name]),
                        "cv_cindex_mean": m, "cv_cindex_sd": s,
                        "model": args.model, "n_subjects": len(df)})
        print(f"  {name:28s} {m:.3f} +/- {s:.3f}")
    pd.DataFrame(ceiling).to_csv(out_dir / "ceiling_with_biomarkers.csv", index=False)

    # ---- 3. increments -----------------------------------------------------
    all_rows = []

    print("\n=== 3a. BSC increments over covariates + amyloid (all 417) ===")
    bio_pairs = [(a + "_bio", b + "_bio", lab + ", amyloid in baseline")
                 for a, b, lab in PAIRS]
    all_rows += increments(df, sets, args.model, args.n_boot, rng, bio_pairs,
                           "full_cohort_amyloid_adjusted")

    ap_df = df[df["amyloid_status_label"] == "A+"].reset_index(drop=True)
    print(f"\n=== 3b. Sensitivity: amyloid-positive subgroup "
          f"(n={len(ap_df)}, events={int(ap_df['event'].sum())}) ===")
    all_rows += increments(ap_df, sets, args.model, args.n_boot, rng, PAIRS,
                           "amyloid_positive_subgroup")

    inc = pd.DataFrame(all_rows)
    inc.to_csv(out_dir / "increment_tests_biomarker.csv", index=False)

    # ---- 4. construct validity --------------------------------------------
    print("\n=== 4. Do BSC slopes track pathology? ===")
    slope_cols = (sets.get("F2_bsc_slopes", []) + sets.get("F3_regional_slopes", [])
                  + sets.get("F4_adsig_slope", []))
    cv = construct_validity(df, slope_cols)
    cv.to_csv(out_dir / "construct_validity_biomarkers.csv", index=False)
    if len(cv):
        for target, g in cv.groupby("target"):
            n_sig = int((g["q"] < 0.05).sum())
            best = g.iloc[0]
            print(f"  {best['target_label']:28s} n={best['n']:4d}  "
                  f"{n_sig} of {len(g)} slopes survive FDR;  strongest "
                  f"{best['feature']} rho={best['rho']:+.3f} p={best['p']:.3g}")

    summary = {
        "model": args.model,
        "n_subjects": int(len(df)),
        "n_events": int(df["event"].sum()),
        "n_amyloid_positive": int(len(ap_df)),
        "n_events_amyloid_positive": int(ap_df["event"].sum()),
        "composition": comp.to_dict(orient="records"),
        "composition_chi2_p": comp.attrs["chi2_p"],
        "ceiling": ceiling,
        "increments": all_rows,
        "construct_validity_n_fdr_sig": {
            t: int((g["q"] < 0.05).sum()) for t, g in cv.groupby("target")
        } if len(cv) else {},
    }
    with open(out_dir / "biomarker_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote results to {out_dir}")


if __name__ == "__main__":
    main()
