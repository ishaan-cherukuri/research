"""Construct validity of BSC, and how much of it is acquisition signal.

Section 4.4 of the manuscript reports six cross-sectional correlations showing
that BSC tracks image quality and not age. A cross-sectional correlation across
subjects cannot separate a property of the measure from a property of the people
being measured, so this script asks the same question three further ways, all on
the frozen 417-subject cohort.

  A. Within-subject. For every subject with three or more scans, does BSC move
     with image quality inside the same brain over time? Each variable is
     centred within subject, which removes every time-invariant confound the
     subject carries, and a mixed model with a subject random intercept is fitted
     to the centred data. A between-subject regression on the subject means is
     fitted alongside for contrast.

  B. Partial association with the outcome. The association between BSC slopes and
     conversion, before and after the image-quality slopes are partialled out. If
     BSC's outcome association survives, the quality sensitivity is a nuisance;
     if it shrinks, the association was running through acquisition.

  C. Scanner signal. Two tests. First, whether a model given nothing but
     acquisition structure (site, vendor, field strength) reaches a non-trivial
     C-index, which measures how much outcome information leaks through where and
     on what a subject was scanned. Second, the regional-BSC C-index before and
     after each regional feature is residualized on scanner identity inside the
     training folds.

Outputs go to results/nireports/. Nothing here refits a model reported elsewhere.
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

QUALITY = [("qc_snr", "Signal-to-noise ratio"),
           ("qc_brain_std", "Brain intensity SD"),
           ("qc_brain_bg_ratio", "Brain-to-background ratio")]
BIOLOGY = [("age", "Age at scan"), ("seg_bpf", "Brain parenchymal fraction")]

# Subject-level slope columns, matched to the per-scan quality metrics above.
QUALITY_SLOPES = ["qc_snr_slope_yr", "qc_brain_bg_ratio_slope_yr"]


def per_scan_table(cluster: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per scan: BSC, image quality, biology and age at scan."""
    bsc = pd.read_csv(cluster / "bsc_simple_features_merged.csv")
    t1 = pd.read_csv(cluster / "t1_scan_features.csv")
    rd = pd.read_csv(cluster / "manifest_realdates.csv")
    rd["image_id"] = (rd.subject + "_" + rd.visit_code + "_"
                      + pd.to_datetime(rd.acq_date).dt.strftime("%Y-%m-%d"))
    rd["dt"] = pd.to_datetime(rd.real_date.fillna(rd.acq_date), errors="coerce")
    rd = rd.merge(cohort[["subject", "age_at_landmark", "landmark_date"]],
                  on="subject", how="inner")
    rd["age"] = (rd.age_at_landmark
                 + (rd.dt - pd.to_datetime(rd.landmark_date)).dt.days / 365.25)

    d = (bsc[["image_id", "subject", "bsc_dir_mean", "bsc_mag_mean"]]
         .merge(rd[["image_id", "age", "dt"]], on="image_id")
         .merge(t1[["image_id", "qc_snr", "qc_brain_std", "qc_brain_bg_ratio",
                    "seg_bpf"]], on="image_id"))
    return d.dropna(subset=["bsc_dir_mean"] + [c for c, _ in QUALITY + BIOLOGY])


def cross_sectional(d: pd.DataFrame) -> list[dict]:
    from scipy import stats
    rows = []
    for col, label in BIOLOGY + QUALITY:
        r, p = stats.pearsonr(d[col], d["bsc_dir_mean"])
        rows.append({"predictor": col, "label": label, "n_scans": len(d),
                     "r": float(r), "p": float(p),
                     "kind": "biological" if (col, label) in BIOLOGY else "quality"})
    return rows


def _pack(res, names, suffix=""):
    out = {}
    ci = res.conf_int()
    for n in names:
        key = n + suffix
        if key in res.params.index:
            out[n] = {"coef": float(res.params[key]), "se": float(res.bse[key]),
                      "p": float(res.pvalues[key]),
                      "ci_lo": float(ci.loc[key, 0]), "ci_hi": float(ci.loc[key, 1])}
    return out


def within_subject(d: pd.DataFrame, min_scans=3) -> dict:
    """Does BSC move with image quality inside the same brain over time?

    Three estimators are reported because they answer slightly different
    questions and it is worth seeing that they agree.

      * A mixed model with a subject random intercept, fitted to the scans as
        they are. This is the specification asked for, and it partially pools
        within- and between-subject information.
      * A within-subject (fixed-effects) regression, obtained by centring every
        variable within subject and fitting by least squares with standard
        errors clustered on subject. This discards all between-subject variation
        and is the strict within-brain estimate. Centring removes the subject
        intercept by construction, so a random intercept is not added on top of
        it: the variance component would sit on the boundary at zero.
      * A between-subject regression on the subject means, for contrast.

    Quality alone is fitted first, then age and brain parenchymal fraction are
    added, because within a subject age is simply elapsed study time and is
    therefore confounded with the scanner upgrades ADNI performed over the same
    years.
    """
    import statsmodels.formula.api as smf

    qual = [c for c, _ in QUALITY]
    cols = ["bsc_dir_mean"] + qual + ["age", "seg_bpf"]
    k = d.groupby("subject")["bsc_dir_mean"].transform("size")
    s = d[k >= min_scans].copy()

    # Standardize on the pooled scale, so every coefficient is per SD.
    for c in cols:
        s[c] = (s[c] - s[c].mean()) / s[c].std()

    rhs_q = " + ".join(qual)
    rhs_full = rhs_q + " + age + seg_bpf"

    mixed = {}
    for tag, rhs in (("quality_only", rhs_q), ("full", rhs_full)):
        m = smf.mixedlm(f"bsc_dir_mean ~ {rhs}", s, groups=s["subject"]).fit(reml=True)
        mixed[tag] = {"coefs": _pack(m, qual + ["age", "seg_bpf"]),
                      "converged": bool(m.converged),
                      "group_var": float(m.cov_re.iloc[0, 0])}

    means = s.groupby("subject")[cols].transform("mean")
    w = s[["subject"]].copy()
    for c in cols:
        w[c] = s[c] - means[c]
    within = {}
    for tag, rhs in (("quality_only", rhs_q), ("full", rhs_full)):
        fe = smf.ols(f"bsc_dir_mean ~ {rhs}", data=w).fit(
            cov_type="cluster", cov_kwds={"groups": w["subject"]})
        within[tag] = {"coefs": _pack(fe, qual + ["age", "seg_bpf"]),
                       "r2": float(fe.rsquared)}

    b = s.groupby("subject")[cols].mean()
    bm = smf.ols(f"bsc_dir_mean ~ {rhs_full}", data=b).fit()

    # Within subject, elapsed time and signal-to-noise move together, so the
    # correlation between them is reported rather than left implicit.
    corr = {c: float(w["age"].corr(w[c])) for c in qual}

    # Single-predictor versions, so the within, between and cross-sectional
    # estimates in Figure 7A are all on the same footing: a standardized slope
    # from a regression on one metric.
    unadj = {}
    for c in qual:
        fe = smf.ols(f"bsc_dir_mean ~ {c}", data=w).fit(
            cov_type="cluster", cov_kwds={"groups": w["subject"]})
        bo = smf.ols(f"bsc_dir_mean ~ {c}", data=b).fit()
        unadj[c] = {"within": _pack(fe, [c])[c], "between": _pack(bo, [c])[c]}

    return {
        "unadjusted": unadj,
        "n_subjects": int(w["subject"].nunique()),
        "n_scans": int(len(w)),
        "min_scans": min_scans,
        "mixed_random_intercept": mixed,
        "within_fixed_effects": within,
        "between": _pack(bm, qual + ["age", "seg_bpf"]),
        "within_corr_age_vs_quality": corr,
    }


def partial_outcome(cohort: pd.DataFrame) -> list[dict]:
    """Cox HR per SD for BSC slopes, raw and adjusted for image-quality slopes."""
    from lifelines import CoxPHFitter

    q = [c for c in QUALITY_SLOPES if c in cohort.columns]
    rows = []
    for feat in ["bsc_dir_mean_slope", "bsc_mag_mean_slope"]:
        if feat not in cohort.columns:
            continue
        use = [feat] + q
        d = cohort[use + ["time_years", "event"]].apply(pd.to_numeric,
                                                        errors="coerce")
        d = d.dropna()
        for c in use:
            d[c] = np.sign(d[c]) * np.log1p(np.abs(d[c]))
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        for tag, cols in (("raw", [feat]), ("quality-adjusted", use)):
            f = CoxPHFitter(penalizer=0.01).fit(
                d[cols + ["time_years", "event"]],
                duration_col="time_years", event_col="event")
            s = f.summary.loc[feat]
            rows.append({"feature": feat, "model": tag, "n": len(d),
                         "hr": float(np.exp(s["coef"])),
                         "hr_lo": float(np.exp(s["coef lower 95%"])),
                         "hr_hi": float(np.exp(s["coef upper 95%"])),
                         "p": float(s["p"]),
                         "adjusted_for": ", ".join(q) if tag != "raw" else ""})
    return rows


def oof_risk(df, cols, model) -> np.ndarray:
    """Pooled out-of-fold risk score, in the original row order."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    r = np.full(len(df), np.nan)
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        ytr = surv_y(tr)
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        r[b] = rte * flip
    return r


def partial_risk_score(df, reg_cols, model) -> list[dict]:
    """The regional-BSC risk score against conversion, raw and quality-adjusted.

    The single global slope tested above has no outcome association to begin
    with, so there is nothing for image quality to explain away. The regional
    feature set is the one that reaches a C-index of 0.649, so the same question
    is put to its out-of-fold risk score, which is what that number describes.
    """
    from lifelines import CoxPHFitter

    q = [c for c in QUALITY_SLOPES if c in df.columns]
    d = df[["time_years", "event"] + q].apply(pd.to_numeric, errors="coerce").copy()
    d["bsc_regional_score"] = oof_risk(df, reg_cols, model)
    d = d.dropna()
    for c in ["bsc_regional_score"] + q:
        d[c] = (d[c] - d[c].mean()) / d[c].std()

    rows = []
    for tag, cols in (("raw", ["bsc_regional_score"]),
                      ("quality-adjusted", ["bsc_regional_score"] + q)):
        f = CoxPHFitter(penalizer=0.01).fit(
            d[cols + ["time_years", "event"]],
            duration_col="time_years", event_col="event")
        s = f.summary.loc["bsc_regional_score"]
        rows.append({"feature": "regional BSC risk score", "model": tag,
                     "n": len(d), "hr": float(np.exp(s["coef"])),
                     "hr_lo": float(np.exp(s["coef lower 95%"])),
                     "hr_hi": float(np.exp(s["coef upper 95%"])),
                     "p": float(s["p"]),
                     "adjusted_for": ", ".join(q) if tag != "raw" else ""})
    return rows


def scanner_matrix(df: pd.DataFrame, min_n=10) -> pd.DataFrame:
    """Acquisition structure as predictors: site, vendor and field strength.

    Sites contributing fewer than min_n subjects are pooled, since a dummy fitted
    to three subjects encodes those subjects rather than their scanner.
    """
    site = df["site"].astype(str)
    keep = site.value_counts()
    site = site.where(site.isin(keep[keep >= min_n].index), "other")
    X = pd.get_dummies(site, prefix="site").astype(float)
    X = X.join(pd.get_dummies(df["manufacturer"].astype(str),
                              prefix="vendor").astype(float))
    X["field_strength_bin"] = df["field_strength_bin"].to_numpy()
    X.index = df.index
    return X


def cv_cindex_matrix(X: pd.DataFrame, df: pd.DataFrame, model: str):
    """5-fold CV C-index for an already-built design matrix."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for a, b in skf.split(df, df["event"]):
        Xtr, Xte = X.iloc[a].to_numpy(), X.iloc[b].to_numpy()
        ytr, yte = surv_y(df.iloc[a]), surv_y(df.iloc[b])
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        folds.append(cindex(yte, rte * flip))
    return float(np.mean(folds)), float(np.std(folds, ddof=1))


def cv_cindex_residualized(df, cols, model, resid_on: pd.DataFrame | None):
    """CV C-index, optionally residualizing every feature on scanner identity.

    The residualizing regression is fitted on the training rows only, so no test
    scanner information reaches the training fold.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        if resid_on is not None:
            Str = np.c_[np.ones(len(a)), resid_on.iloc[a].to_numpy()]
            Ste = np.c_[np.ones(len(b)), resid_on.iloc[b].to_numpy()]
            beta, *_ = np.linalg.lstsq(Str, Xtr, rcond=None)
            Xtr = Xtr - Str @ beta
            Xte = Xte - Ste @ beta
        ytr, yte = surv_y(tr), surv_y(te)
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        folds.append(cindex(yte, rte * flip))
    return float(np.mean(folds)), float(np.std(folds, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--cluster", default="/Users/ishu/research/adni_trillium/from_cluster")
    ap.add_argument("--out_dir", default=str(root / "results/nireports"))
    ap.add_argument("--model", default="rsf")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scans = per_scan_table(Path(args.cluster), df)
    xs = cross_sectional(scans)
    pd.DataFrame(xs).to_csv(out / "construct_cross_sectional.csv", index=False)
    print(f"cross-sectional, {len(scans)} scans")
    for r in xs:
        print(f"  {r['label']:30s} r={r['r']:+.3f}  p={r['p']:.2g}")

    ws = within_subject(scans)
    print(f"\nwithin-subject models, {ws['n_subjects']} subjects, "
          f"{ws['n_scans']} scans")
    for tag in ("quality_only", "full"):
        print(f"  [{tag}]")
        for k, v in ws["within_fixed_effects"][tag]["coefs"].items():
            print(f"    within  {k:20s} {v['coef']:+.3f} "
                  f"({v['ci_lo']:+.3f}, {v['ci_hi']:+.3f})  p={v['p']:.2g}")
        for k, v in ws["mixed_random_intercept"][tag]["coefs"].items():
            print(f"    mixed   {k:20s} {v['coef']:+.3f} p={v['p']:.2g}")
    for k, v in ws["between"].items():
        print(f"  between {k:20s} {v['coef']:+.3f} p={v['p']:.2g}")
    print(f"  within-subject corr(age, quality): {ws['within_corr_age_vs_quality']}")

    sets = feature_sets(df)
    reg = [c for c in sets["F3_regional_slopes"] if c in df.columns]

    po = partial_outcome(df) + partial_risk_score(df, reg, args.model)
    pd.DataFrame(po).to_csv(out / "construct_partial_outcome.csv", index=False)
    print("\npartial association with conversion")
    for r in po:
        print(f"  {r['feature']:24s} {r['model']:17s} "
              f"HR={r['hr']:.3f} ({r['hr_lo']:.3f}, {r['hr_hi']:.3f}) p={r['p']:.3f}")

    S = scanner_matrix(df)
    sc_m, sc_s = cv_cindex_matrix(S, df, args.model)
    raw_m, raw_s = cv_cindex_residualized(df, reg, args.model, None)
    res_m, res_s = cv_cindex_residualized(df, reg, args.model, S)
    scanner = {
        "model": args.model,
        "n_scanner_predictors": int(S.shape[1]),
        "n_sites_retained": int(sum(c.startswith("site_") and c != "site_other"
                                    for c in S.columns)),
        "cindex_scanner_only": sc_m, "cindex_scanner_only_sd": sc_s,
        "cindex_regional_bsc": raw_m, "cindex_regional_bsc_sd": raw_s,
        "cindex_regional_bsc_residualized": res_m,
        "cindex_regional_bsc_residualized_sd": res_s,
        "drop_after_residualizing": raw_m - res_m,
    }
    print(f"\nscanner-only model         C={sc_m:.3f}+-{sc_s:.3f}")
    print(f"regional BSC               C={raw_m:.3f}+-{raw_s:.3f}")
    print(f"regional BSC, residualized C={res_m:.3f}+-{res_s:.3f}  "
          f"drop={raw_m - res_m:+.3f}")

    with open(out / "construct_validity.json", "w") as f:
        json.dump({"seed": SEED, "cross_sectional": xs,
                   "within_subject": ws, "partial_outcome": po,
                   "scanner_signal": scanner}, f, indent=2)
    pd.DataFrame([scanner]).to_csv(out / "scanner_signal.csv", index=False)
    print(f"\nwrote {out/'construct_validity.json'}")


if __name__ == "__main__":
    main()
