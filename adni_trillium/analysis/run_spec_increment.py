"""Fold-matched incremental-value tests for the V3 revision spec.

Spec section 8.3 names F8 versus F9 as the key comparison for the paper's
central claim, and section 10.5 asks for the delta with a 95 percent interval.
Comparing the two marginal CV means is not a valid paired test, so this script
refits both feature sets on identical folds, takes the per-fold difference in
C-index, and bootstraps subjects to get an interval.

Also runs a likelihood-ratio test of the nested Cox models, and an FDR-corrected
univariate screen of the individual BSC slopes, so the conclusion does not rest
on a single metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_spec_models import (SEED, cindex, feature_sets, fit_predict,
                             preprocess, surv_y)

# Each pair is (baseline set, augmented set, what the delta is testing).
PAIRS = [
    ("F0_covariates", "F6_cov_bsc", "BSC slopes over covariates"),
    ("F8_cov_stdmri", "F9_cov_bsc_stdmri", "BSC slopes over covariates + standard MRI"),
    ("F0_covariates", "F8_cov_stdmri", "standard MRI slopes over covariates"),
    ("F0_covariates", "F7_cov_regional", "regional BSC over covariates"),
    ("F8_cov_stdmri", "F9b_cov_regional_stdmri", "regional BSC over covariates + standard MRI"),
    ("F0_covariates", "F4b_cov_adsig", "AD-signature composite over covariates"),
    ("F8_cov_stdmri", "F9c_cov_adsig_stdmri", "AD-signature over covariates + standard MRI"),
]


def paired_folds(df, cols_a, cols_b, model, n_boot, rng, seed=None):
    """Per-fold C-index for both feature sets on identical splits."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED if seed is None else seed)
    ca, cb, per_subject = [], [], []
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        ytr, yte = surv_y(tr), surv_y(te)
        risks = {}
        for tag, cols in (("a", cols_a), ("b", cols_b)):
            Xtr, Xte = preprocess(tr, te, cols)
            rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
            flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
            risks[tag] = rte * flip
        ca.append(cindex(yte, risks["a"]))
        cb.append(cindex(yte, risks["b"]))
        per_subject.append(pd.DataFrame({
            "event": te["event"].to_numpy(), "time": te["time_years"].to_numpy(),
            "risk_a": risks["a"], "risk_b": risks["b"]}))

    # Bootstrap over pooled out-of-fold predictions, resampling subjects.
    oof = pd.concat(per_subject, ignore_index=True)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(oof), len(oof))
        s = oof.iloc[idx]
        if s["event"].sum() < 5:
            continue
        y = np.array([(bool(e), float(t)) for e, t in zip(s["event"], s["time"])],
                     dtype=[("event", "?"), ("time", "<f8")])
        try:
            deltas.append(cindex(y, s["risk_b"].to_numpy())
                          - cindex(y, s["risk_a"].to_numpy()))
        except Exception:
            continue
    return np.array(ca), np.array(cb), np.array(deltas), oof


def cox_lr_test(df, cols_a, cols_b):
    """Likelihood-ratio test of the nested Cox models on the full cohort."""
    from lifelines import CoxPHFitter
    from scipy import stats

    extra = [c for c in cols_b if c not in cols_a]
    if not extra:
        return {}
    Xa, _ = preprocess(df, df, cols_a)
    Xb, _ = preprocess(df, df, cols_b)

    def ll(X, cols):
        d = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        d["T"], d["E"] = df["time_years"].to_numpy(), df["event"].to_numpy()
        f = CoxPHFitter(penalizer=0.1).fit(d, duration_col="T", event_col="E")
        return f.log_likelihood_

    lla, llb = ll(Xa, cols_a), ll(Xb, cols_b)
    stat = 2 * (llb - lla)
    dof = len(extra)
    return {"lr_stat": float(stat), "df": dof,
            "p": float(stats.chi2.sf(stat, dof)) if stat > 0 else 1.0}


def univariate_screen(df, cols):
    """Per-slope Cox screen with Benjamini-Hochberg correction."""
    from lifelines import CoxPHFitter
    rows = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() < 50 or v.std(skipna=True) == 0:
            continue
        d = pd.DataFrame({"x": ((v - v.mean()) / v.std()).fillna(0.0),
                          "T": df["time_years"], "E": df["event"]})
        try:
            f = CoxPHFitter().fit(d, duration_col="T", event_col="E")
            rows.append({"feature": c, "hr": float(np.exp(f.params_["x"])),
                         "p": float(f.summary.loc["x", "p"])})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(out)
    # Benjamini-Hochberg is a step-up procedure: the running minimum sweeps from
    # the largest p-value down, so a feature can never be called more
    # significant than any less-significant feature above it.
    raw = out["p"] * m / (out.index + 1)
    out["q"] = raw[::-1].cummin()[::-1].clip(upper=1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3/spec_cohort.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/spec_v3"))
    ap.add_argument("--model", default="cox_lasso")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None,
                    help="override the CV fold seed; a result that only holds for "
                         "one split is an artefact of that split, not a finding")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)
    sets = feature_sets(df)
    rng = np.random.default_rng(SEED)

    results = []
    for a_name, b_name, label in PAIRS:
        ca, cb, deltas, oof = paired_folds(df, sets[a_name], sets[b_name],
                                           args.model, args.n_boot, rng, args.seed)
        lo, hi = (np.percentile(deltas, [2.5, 97.5]) if len(deltas)
                  else (np.nan, np.nan))
        lr = cox_lr_test(df, sets[a_name], sets[b_name])
        rec = {"comparison": label, "baseline": a_name, "augmented": b_name,
               "model": args.model,
               "cv_baseline_mean": float(ca.mean()), "cv_baseline_sd": float(ca.std(ddof=1)),
               "cv_augmented_mean": float(cb.mean()), "cv_augmented_sd": float(cb.std(ddof=1)),
               "fold_matched_delta": float((cb - ca).mean()),
               "delta_ci_lo": float(lo), "delta_ci_hi": float(hi),
               "n_boot": int(len(deltas))}
        rec.update({f"cox_{k}": v for k, v in lr.items()})
        results.append(rec)
        print(f"{label}\n  {a_name} {ca.mean():.3f} -> {b_name} {cb.mean():.3f}"
              f"  delta={rec['fold_matched_delta']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  cox LR p={lr.get('p', float('nan')):.3g}")

    screen_cols = sets["F2_bsc_slopes"] + sets.get("F3_regional_slopes", []) \
                  + sets.get("F4_adsig_slope", [])
    screen = univariate_screen(df, screen_cols)
    n_sig = int((screen["q"] < 0.05).sum()) if len(screen) else 0
    print(f"\nunivariate BSC slope screen: {n_sig} of {len(screen)} survive FDR q<0.05")
    if len(screen):
        print(screen.head(5).to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "increment_tests.csv", index=False)
    screen.to_csv(out_dir / "bsc_slope_screen.csv", index=False)
    with open(out_dir / "increment_tests.json", "w") as f:
        json.dump({"results": results, "n_slopes_tested": len(screen),
                   "n_slopes_fdr_sig": n_sig}, f, indent=2)


if __name__ == "__main__":
    main()
