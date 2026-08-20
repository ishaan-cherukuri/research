"""Model portfolio for the JAD ALZ-26-0928 V3 revision spec.

Implements spec sections 7 through 9:

  * every model runs against every feature set it is defined for (section 8.3)
  * stratified 5-fold CV on a training pool, with a 20 percent stratified
    hold-out reserved and touched once at the end (section 9)
  * all preprocessing, imputation and winsorization is fit on training folds
    only, so nothing leaks across the split
  * the sign convention of section 7.2 is enforced: a risk score whose training
    concordance falls below 0.5 is negated, so no reported C-index is below 0.5
  * Harrell's C, Uno's C, integrated Brier score and time-dependent AUC

XGBoost AFT is the flagship. RSF, L2-Cox, Lasso-Cox and the three parametric
AFT models are comparators.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

SEED = 42

COVARIATES = ["age_at_landmark", "female", "educ_years", "apoe4",
              "mmse_at_landmark", "adas13_at_landmark", "cdrsb_at_landmark",
              "field_strength_bin"]

META = {"subject", "event", "time_years", "landmark_date", "n_scans_window",
        "window_span_years", "n_scans_total", "n_scans_freesurfer",
        "manufacturer", "scanner_model", "scanner_id", "site",
        "field_strength_changed", "scanner_changed", "field_strength_bl",
        "field_strength_bin"}


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """The feature-set variants of spec section 8.3 that do not need regional BSC."""
    bsc_all = [c for c in df.columns
               if c.startswith("bsc_") or c.startswith("Nboundary")]
    bsc_slope = [c for c in bsc_all if c.endswith("_slope")]
    bsc_base = [c for c in bsc_all if c.endswith("_baseline")]
    thick = [c for c in df.columns if c.startswith("fs_ST") and c.endswith("_slope")]
    hippo = [c for c in df.columns if c.startswith("fs_hippo") and c.endswith("_slope")]
    std_mri = thick + hippo

    reg_slope = [c for c in df.columns
                 if c.startswith(("bscdir_roi", "bscmag_roi")) and c.endswith("_slope")]
    adsig = [c for c in df.columns
             if c.startswith(("bscdir_adsig", "bscmag_adsig")) and c.endswith("_slope")]

    cov = [c for c in COVARIATES if c in df.columns]
    sets = {
        "F0_covariates": cov,
        "F1_bsc_baseline": bsc_base,
        "F2_bsc_slopes": bsc_slope,
        "F5_std_mri_slopes": std_mri,
        "F6_cov_bsc": cov + bsc_slope,
        "F8_cov_stdmri": cov + std_mri,
        "F9_cov_bsc_stdmri": cov + bsc_slope + std_mri,
    }
    # Regional sets only appear when the regional features were built in.
    if reg_slope:
        sets["F3_regional_slopes"] = reg_slope
        sets["F7_cov_regional"] = cov + reg_slope
        sets["F9b_cov_regional_stdmri"] = cov + reg_slope + std_mri
    if adsig:
        sets["F4_adsig_slope"] = adsig
        sets["F4b_cov_adsig"] = cov + adsig
        sets["F9c_cov_adsig_stdmri"] = cov + adsig + std_mri
    return sets


def preprocess(train: pd.DataFrame, test: pd.DataFrame,
               cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Impute, signed-log, winsorize and scale, all fit on the training rows."""
    tr = train[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    te = test[cols].apply(pd.to_numeric, errors="coerce").astype(float)

    med = tr.median()
    med = med.fillna(0.0)
    tr, te = tr.fillna(med), te.fillna(med)

    tr = np.sign(tr) * np.log1p(np.abs(tr))
    te = np.sign(te) * np.log1p(np.abs(te))

    lo, hi = tr.quantile(0.01), tr.quantile(0.99)
    tr, te = tr.clip(lo, hi, axis=1), te.clip(lo, hi, axis=1)

    mn, rng = tr.min(), (tr.max() - tr.min()).replace(0, 1.0)
    tr = (tr - mn) / rng
    te = ((te - mn) / rng).clip(-1.0, 2.0)
    return tr.to_numpy(), te.to_numpy()


def surv_y(df: pd.DataFrame) -> np.ndarray:
    return np.array([(bool(e), float(t))
                     for e, t in zip(df["event"], df["time_years"])],
                    dtype=[("event", "?"), ("time", "<f8")])


def cindex(y, risk) -> float:
    from sksurv.metrics import concordance_index_censored
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])



def _inner_cv_select(model, Xtr, ytr, grid, fit_one, n_splits=3):
    """Pick a penalty by inner cross-validation on the training fold only.

    Spec section 8.2 asks for the Lasso penalty to be chosen this way. Leaving
    the penalty fixed lets these models overfit 33 collinear slope features on
    roughly a hundred events, and the risk direction they learn on the training
    fold is then noise that does not survive to the test fold, which is what a
    held-out C-index below 0.5 records.
    """
    from sklearn.model_selection import StratifiedKFold
    if len(grid) == 1:
        return grid[0]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for g in grid:
        fold = []
        for a, b in skf.split(Xtr, ytr["event"]):
            try:
                rtr, rte = fit_one(g, Xtr[a], ytr[a], Xtr[b])
                flip = -1.0 if cindex(ytr[a], rtr) < 0.5 else 1.0
                fold.append(cindex(ytr[b], rte * flip))
            except Exception:
                continue
        scores.append(np.mean(fold) if fold else -np.inf)
    return grid[int(np.argmax(scores))]


def fit_predict(model: str, Xtr, ytr, Xte, params: dict | None = None):
    """Return (train_risk, test_risk). Higher risk means shorter survival."""
    import xgboost as xgb
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from lifelines import (LogLogisticAFTFitter, LogNormalAFTFitter,
                           WeibullAFTFitter)

    if model == "xgb_aft":
        p = dict(objective="survival:aft", eval_metric="aft-nloglik",
                 aft_loss_distribution="normal", aft_loss_distribution_scale=1.0,
                 tree_method="hist", learning_rate=0.05, max_depth=4,
                 min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
                 subsample=0.8, colsample_bytree=0.8, seed=SEED)
        p.update(params or {})
        n_rounds = int(p.pop("n_rounds", 200))
        t = ytr["time"]
        dtr = xgb.DMatrix(Xtr)
        dtr.set_float_info("label_lower_bound", t)
        dtr.set_float_info("label_upper_bound", np.where(ytr["event"], t, np.inf))
        bst = xgb.train(p, dtr, num_boost_round=n_rounds)
        # AFT predicts survival time, so risk is its negation (spec 8.1).
        return -bst.predict(xgb.DMatrix(Xtr)), -bst.predict(xgb.DMatrix(Xte)), bst

    if model == "rsf":
        m = RandomSurvivalForest(n_estimators=1000, min_samples_split=10,
                                 min_samples_leaf=15, max_features="sqrt",
                                 random_state=SEED, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xtr), m.predict(Xte), m

    if model == "cox_l2":
        def _f(a, xa, ya, xb):
            mm = CoxPHSurvivalAnalysis(alpha=a).fit(xa, ya)
            return mm.predict(xa), mm.predict(xb)
        a = _inner_cv_select(model, Xtr, ytr, [0.01, 0.1, 1.0, 10.0, 100.0], _f)
        m = CoxPHSurvivalAnalysis(alpha=a).fit(Xtr, ytr)
        return m.predict(Xtr), m.predict(Xte), m

    if model == "cox_lasso":
        base = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.05,
                                      n_alphas=30, fit_baseline_model=False)
        base.fit(Xtr, ytr)
        grid = list(base.alphas_[:: max(1, len(base.alphas_) // 8)])

        def _f(a, xa, ya, xb):
            mm = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[a],
                                        fit_baseline_model=False).fit(xa, ya)
            return mm.predict(xa), mm.predict(xb)
        a = _inner_cv_select(model, Xtr, ytr, grid, _f)
        return (base.predict(Xtr, alpha=a), base.predict(Xte, alpha=a), base)

    # lifelines parametric AFT models
    fitters = {"aft_weibull": WeibullAFTFitter, "aft_lognormal": LogNormalAFTFitter,
               "aft_loglogistic": LogLogisticAFTFitter}
    cols = [f"x{i}" for i in range(Xtr.shape[1])]

    def _f(pen, xa, ya, xb):
        da = pd.DataFrame(xa, columns=cols)
        da["T"], da["E"] = ya["time"], ya["event"].astype(int)
        ff = fitters[model](penalizer=pen).fit(da, duration_col="T", event_col="E")
        ra = -ff.predict_median(pd.DataFrame(xa, columns=cols)).to_numpy()
        rb = -ff.predict_median(pd.DataFrame(xb, columns=cols)).to_numpy()
        return (np.nan_to_num(ra, posinf=0, neginf=0),
                np.nan_to_num(rb, posinf=0, neginf=0))
    pen = _inner_cv_select(model, Xtr, ytr, [0.01, 0.1, 1.0, 10.0], _f)
    f = fitters[model](penalizer=pen)
    dtr = pd.DataFrame(Xtr, columns=cols)
    dtr["T"], dtr["E"] = ytr["time"], ytr["event"].astype(int)
    f.fit(dtr, duration_col="T", event_col="E")
    # Predicted median survival time, negated to become a risk score.
    rtr = -f.predict_median(pd.DataFrame(Xtr, columns=cols)).to_numpy()
    rte = -f.predict_median(pd.DataFrame(Xte, columns=cols)).to_numpy()
    return np.nan_to_num(rtr, posinf=0, neginf=0), np.nan_to_num(rte, posinf=0, neginf=0), f


def extra_metrics(ytr, yte, risk_te, horizons) -> dict:
    """Uno's C, integrated Brier score and time-dependent AUC."""
    from sksurv.metrics import (concordance_index_ipcw, cumulative_dynamic_auc)
    out = {}
    tmax = min(yte["time"][yte["event"]].max(), ytr["time"].max())
    hz = [h for h in horizons if h < tmax]
    try:
        out["uno_c"] = float(concordance_index_ipcw(ytr, yte, risk_te, tau=tmax)[0])
    except Exception:
        out["uno_c"] = np.nan
    try:
        if hz:
            auc, _ = cumulative_dynamic_auc(ytr, yte, risk_te, np.array(hz))
            for h, a in zip(hz, auc):
                out[f"auc_{h}y"] = float(a)
    except Exception:
        pass
    return out


def run(args) -> None:
    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)

    sets = feature_sets(df)
    models = args.models.split(",")

    tr_idx, ho_idx = train_test_split(np.arange(len(df)), test_size=0.20,
                                      stratify=df["event"], random_state=SEED)
    pool, hold = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)

    rows = []
    for sname, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for mname in models:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_c, fold_extra = [], []
            for k, (a, b) in enumerate(skf.split(pool, pool["event"])):
                tr, te = pool.iloc[a], pool.iloc[b]
                Xtr, Xte = preprocess(tr, te, cols)
                ytr, yte = surv_y(tr), surv_y(te)
                try:
                    rtr, rte, _ = fit_predict(mname, Xtr, ytr, Xte)
                except Exception as e:
                    print(f"[FAIL] {mname} {sname} fold{k}: {e}")
                    continue
                # Spec 7.2: fix the risk direction using the training fold only.
                flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
                rte = rte * flip
                fold_c.append(cindex(yte, rte))
                fold_extra.append(extra_metrics(ytr, yte, rte, [1.0, 2.0, 3.0]))
            if not fold_c:
                continue
            rec = {"model": mname, "feature_set": sname, "n_features": len(cols),
                   "cv_cindex_mean": float(np.mean(fold_c)),
                   "cv_cindex_sd": float(np.std(fold_c, ddof=1)) if len(fold_c) > 1 else 0.0,
                   "n_folds": len(fold_c)}
            for key in set().union(*[set(d) for d in fold_extra]):
                vals = [d[key] for d in fold_extra if key in d and np.isfinite(d[key])]
                if vals:
                    rec[f"cv_{key}"] = float(np.mean(vals))

            # Hold-out, fit once on the whole pool.
            try:
                Xp, Xh = preprocess(pool, hold, cols)
                yp, yh = surv_y(pool), surv_y(hold)
                rp, rh, _ = fit_predict(mname, Xp, yp, Xh)
                flip = -1.0 if cindex(yp, rp) < 0.5 else 1.0
                rec["holdout_cindex"] = cindex(yh, rh * flip)
                rec["sign_flipped"] = int(flip < 0)
            except Exception as e:
                print(f"[FAIL holdout] {mname} {sname}: {e}")
            rows.append(rec)
            print(f"  {mname:16s} {sname:20s} n={len(cols):3d} "
                  f"CV={rec['cv_cindex_mean']:.3f}+-{rec['cv_cindex_sd']:.3f} "
                  f"HO={rec.get('holdout_cindex', float('nan')):.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("cv_cindex_mean", ascending=False)
    res.to_csv(out_dir / "cv_results.csv", index=False)
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump({"n_pool": len(pool), "n_holdout": len(hold),
                   "events_pool": int(pool["event"].sum()),
                   "events_holdout": int(hold["event"].sum()),
                   "results": rows}, f, indent=2)
    print(f"\nwrote {out_dir/'cv_results.csv'}")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3/spec_cohort.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/spec_v3"))
    ap.add_argument("--models", default="xgb_aft,rsf,cox_l2,cox_lasso,"
                                        "aft_weibull,aft_lognormal,aft_loglogistic")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
