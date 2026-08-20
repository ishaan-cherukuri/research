"""Cross-validated model ladder on the landmark cohort (ALZ-26-0928 revision).

Runs on the output of build_landmark_cohort.py, so every feature is derived from
scans acquired while the subject was still MCI and follow-up starts at the last
of those scans.

Feature blocks
    cov      age at landmark, sex, education, MMSE at landmark
    bsc      annualized BSC slopes, top-k selected inside each training fold
    t1x      cross-sectional T1 morphometry at the first scan in the window
    t1long   T1 morphometry trajectories over the window (brain volume, BPF,
             tissue volumes, intensity, SNR)

Model families: penalized Cox, three parametric AFT distributions, Random
Survival Forest, XGBoost AFT. Feature selection, imputation, and scaling are fit
on training folds only. Risk direction is fixed in fit_predict(), so larger
always means "converts sooner" and a C-index below 0.5 is a real failure rather
than a flipped sign.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import (LogLogisticAFTFitter, LogNormalAFTFitter,
                       WeibullAFTFitter)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")

COV_COLS = ["age_at_landmark", "female", "educ_years", "mmse_at_landmark"]
T1_TRAJ_BASES = ["seg_brain_mm3", "seg_bpf", "seg_gm_total_mm3", "seg_wm_total_mm3",
                 "seg_csf_mm3", "qc_brain_mean", "qc_snr", "qc_brain_bg_ratio"]
T1_TRAJ_SUFFIXES = ["last", "delta", "pctchg", "slope_yr", "mean", "std"]
T1_CROSS_EXTRA = ["seg_tiv_mm3_bl", "qc_brain_mask_vol_mm3_bl", "qc_brain_std_bl",
                  "meta_field_strength_t_bl"]


def get_blocks(df: pd.DataFrame) -> dict[str, list[str]]:
    bsc = [c for c in df.columns if c.endswith("_slope") and df[c].notna().mean() > 0.8]
    t1x = [f"{b}_bl" for b in T1_TRAJ_BASES] + T1_CROSS_EXTRA
    t1long = [f"{b}_{s}" for b in T1_TRAJ_BASES for s in T1_TRAJ_SUFFIXES]
    keep = lambda cols: [c for c in cols if c in df.columns and df[c].notna().any()]
    return {"cov": keep(COV_COLS), "bsc": bsc, "t1x": keep(t1x), "t1long": keep(t1long)}


def signed_log1p(X: pd.DataFrame) -> pd.DataFrame:
    return np.sign(X) * np.log1p(np.abs(X))


def select_slopes(X_train: pd.DataFrame, cols: list[str], top_k: int,
                  penalize: str = "nboundary", penalty: float = 0.10) -> list[str]:
    Xl = signed_log1p(X_train[cols])
    Xl = Xl.clip(lower=Xl.quantile(0.01), upper=Xl.quantile(0.99), axis=1)
    score = Xl.var()
    mask = score.index.str.contains(penalize, case=False, regex=True)
    score[mask] = score[mask] * penalty
    return list(score.sort_values(ascending=False).head(top_k).index)


def prep_fold(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    usable = [c for c in X_tr.columns if X_tr[c].notna().any()]
    X_tr, X_te = X_tr[usable], X_te[usable]
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = np.clip(sc.fit_transform(imp.fit_transform(X_tr)), -5, 5)
    B = np.clip(sc.transform(imp.transform(X_te)), -5, 5)
    return (pd.DataFrame(A, columns=usable, index=X_tr.index),
            pd.DataFrame(B, columns=usable, index=X_te.index))


def cindex(event, time, risk) -> float:
    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


def fit_predict(model, Xtr, ytr_e, ytr_t, Xte, seed):
    """Return (risk_train, risk_test); larger risk means earlier conversion."""
    t = np.clip(ytr_t, 1e-3, None)

    if model in ("weibull", "lognormal", "loglogistic"):
        cls = {"weibull": WeibullAFTFitter, "lognormal": LogNormalAFTFitter,
               "loglogistic": LogLogisticAFTFitter}[model]
        f = cls(penalizer=0.1, l1_ratio=0.0)
        d = Xtr.copy()
        d["_t"], d["_e"] = t, ytr_e
        f.fit(d, duration_col="_t", event_col="_e")

        def risk(X):
            m = f.predict_median(X).to_numpy(dtype=float)
            if not np.isfinite(m).all():
                cap = np.nanmax(m[np.isfinite(m)]) * 10 if np.isfinite(m).any() else 1e3
                m = np.where(np.isfinite(m), m, cap)
            return -m
        return risk(Xtr), risk(Xte)

    if model == "coxnet":
        y = Surv.from_arrays(event=ytr_e.astype(bool), time=t)
        f = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.05, n_alphas=50)
        f.fit(Xtr.to_numpy(), y)
        a = f.alphas_[len(f.alphas_) // 2]
        return f.predict(Xtr.to_numpy(), alpha=a), f.predict(Xte.to_numpy(), alpha=a)

    if model == "rsf":
        y = Surv.from_arrays(event=ytr_e.astype(bool), time=t)
        f = RandomSurvivalForest(n_estimators=500, min_samples_split=10,
                                 min_samples_leaf=15, max_features="sqrt",
                                 n_jobs=-1, random_state=seed)
        f.fit(Xtr.to_numpy(), y)
        return f.predict(Xtr.to_numpy()), f.predict(Xte.to_numpy())

    if model == "xgb":
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr.to_numpy())
        dtr.set_float_info("label_lower_bound", t)
        dtr.set_float_info("label_upper_bound", np.where(ytr_e == 1, t, np.inf))
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
                  "aft_loss_distribution": "normal",
                  "aft_loss_distribution_scale": 1.20, "tree_method": "hist",
                  "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10,
                  "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
                  "reg_lambda": 1.0, "seed": seed}
        bst = xgb.train(params, dtr, num_boost_round=500)
        return -bst.predict(dtr), -bst.predict(xgb.DMatrix(Xte.to_numpy()))

    raise ValueError(model)


def fold_columns(df, blocks, feature_set, tr_idx, top_k):
    cols = []
    for b in feature_set:
        cols += (select_slopes(df.iloc[tr_idx], blocks["bsc"], top_k)
                 if b == "bsc" else blocks[b])
    return cols


def run_cv(df, blocks, feature_set, model, top_k, n_splits, seed) -> dict:
    y_e, y_t = df["event"].to_numpy(), df["time_years"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_s, te_s, n_f = [], [], []
    for tr_idx, te_idx in skf.split(df, y_e):
        cols = fold_columns(df, blocks, feature_set, tr_idx, top_k)
        Xtr, Xte = prep_fold(df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
        rtr, rte = fit_predict(model, Xtr, y_e[tr_idx], y_t[tr_idx], Xte, seed)
        tr_s.append(cindex(y_e[tr_idx], y_t[tr_idx], rtr))
        te_s.append(cindex(y_e[te_idx], y_t[te_idx], rte))
        n_f.append(len(cols))
    return {"model": model, "features": "+".join(feature_set),
            "n_features": int(np.median(n_f)),
            "train_mean": round(float(np.mean(tr_s)), 3),
            "train_sd": round(float(np.std(tr_s)), 3),
            "test_mean": round(float(np.mean(te_s)), 3),
            "test_sd": round(float(np.std(te_s)), 3),
            "gap": round(float(np.mean(tr_s) - np.mean(te_s)), 3),
            "test_folds": [round(x, 3) for x in te_s]}


CONFIGS = [
    (["cov"], ["coxnet", "weibull", "lognormal", "loglogistic", "rsf", "xgb"]),
    (["bsc"], ["coxnet", "weibull", "lognormal", "loglogistic", "rsf", "xgb"]),
    (["t1x"], ["coxnet", "rsf", "xgb"]),
    (["t1long"], ["coxnet", "rsf", "xgb"]),
    (["t1x", "t1long"], ["coxnet", "rsf", "xgb"]),
    (["bsc", "t1x", "t1long"], ["coxnet", "rsf", "xgb"]),
    (["cov", "bsc"], ["coxnet", "weibull", "rsf", "xgb"]),
    (["cov", "t1x", "t1long"], ["coxnet", "weibull", "rsf", "xgb"]),
    (["cov", "bsc", "t1x", "t1long"], ["coxnet", "weibull", "lognormal",
                                       "loglogistic", "rsf", "xgb"]),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="analysis/results/landmark24/landmark_cohort.csv")
    p.add_argument("--out_dir", default="analysis/results/landmark24")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.cohort)
    blocks = get_blocks(df)
    print(f"Cohort: {len(df)} subjects, {int(df['event'].sum())} events")
    print("Blocks: " + ", ".join(f"{k}={len(v)}" for k, v in blocks.items()))

    rows = []
    for fs, models in CONFIGS:
        for m in models:
            r = run_cv(df, blocks, fs, m, args.top_k, args.folds, args.seed)
            rows.append(r)
            print(f"  {r['features']:<24} {m:<12} train {r['train_mean']:.3f}  "
                  f"test {r['test_mean']:.3f}±{r['test_sd']:.3f}  (p={r['n_features']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "cv_model_comparison.csv", index=False)
    with open(out_dir / "cv_model_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_dir/'cv_model_comparison.csv'}")


if __name__ == "__main__":
    main()
