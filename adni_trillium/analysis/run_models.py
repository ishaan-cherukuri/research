"""
Fit the full model ladder on the ADNI-Trillium cohort (657 subjects, 251 converters)
and write 5-fold cross-validated results plus held-out predictions.

Model ladder (matches the structure of the 450-subject analysis):
    Weibull AFT      - baseline BSC (cross-sectional anchor)
    Weibull AFT      - 20 BSC slopes
    LogLogistic AFT  - 20 BSC slopes
    LogNormal AFT    - 20 BSC slopes
    RSF              - 20 BSC slopes
    RSF              - 20 BSC slopes + 37 T1 morphometry
    XGBoost AFT      - 20 BSC slopes + 37 T1 morphometry   (primary)

All preprocessing (feature selection, winsor limits, scaling) is refit inside each
fold on the training portion only.

Outputs (--out_dir):
    all_models_cv.json        per-model fold-level train/test/gap C-index
    xgb_predictions.csv       XGBoost held-out predictions (70/30 split)
    xgb_train_predictions.csv XGBoost training-set predictions (for KM panel)
    xgb_gain_importance.csv   gain importance of the fitted XGBoost model
    cohort_table.csv          demographic table rows
    summary.txt               human-readable digest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

SEED = 42

# Explicit T1 whitelist, identical to the 37 features the Trillium cluster model was
# trained on. This must stay a whitelist rather than "every column except X":
# features_t1_all.csv also carries n_visits_used, which correlates with survival time
# at r = 0.82 because follow-up length determines both. Including it leaks the outcome.
T1_WHITELIST = [
    "seg_csf_mm3_bl", "seg_gm_total_mm3_bl", "seg_wm_total_mm3_bl", "seg_brain_mm3_bl",
    "seg_tiv_mm3_bl", "seg_bpf_bl",
    "long_brain_vol_last", "long_brain_vol_delta", "long_brain_vol_pctchg",
    "long_brain_vol_slope_yr", "long_brain_vol_mean", "long_brain_vol_std",
    "long_brain_intensity_last", "long_brain_intensity_delta",
    "long_brain_intensity_pctchg", "long_brain_intensity_slope_yr",
    "long_brain_intensity_mean", "long_brain_intensity_std",
    "long_snr_last", "long_snr_delta", "long_snr_pctchg", "long_snr_slope_yr",
    "long_snr_mean", "long_snr_std",
    "long_brain_bg_ratio_last", "long_brain_bg_ratio_delta",
    "long_brain_bg_ratio_pctchg", "long_brain_bg_ratio_slope_yr",
    "long_brain_bg_ratio_mean", "long_brain_bg_ratio_std",
    "qc_brain_mask_vol_mm3_bl", "qc_brain_mean_bl", "qc_brain_std_bl", "qc_snr_bl",
    "qc_brain_bg_ratio_bl", "field_strength_mode_t", "meta_field_strength_t_bl",
]

XGB_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.20,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": SEED,
}
N_ESTIMATORS = 500


# ---------------------------------------------------------------- preprocessing
def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def fit_limits(X, lo, hi):
    return {"lower": X.quantile(lo), "upper": X.quantile(hi)}


def apply_limits(X, lim):
    return X.clip(lower=lim["lower"], upper=lim["upper"], axis=1)


def select_by_penalized_variance(X_train, top_k, penalize="nboundary",
                                 factor=0.10, lo=0.01, hi=0.99):
    Xr = apply_limits(signed_log1p(X_train), fit_limits(signed_log1p(X_train), lo, hi))
    score = Xr.var()
    mask = score.index.str.contains(penalize, case=False, regex=True)
    score[mask] = score[mask] * factor
    return score.nlargest(top_k).index.tolist()


def preprocess(X_tr, X_te, lo=0.01, hi=0.99):
    tr_log, te_log = signed_log1p(X_tr), signed_log1p(X_te)
    lim = fit_limits(tr_log, lo, hi)
    tr, te = apply_limits(tr_log, lim), apply_limits(te_log, lim)
    sc = MinMaxScaler()
    return (pd.DataFrame(sc.fit_transform(tr), columns=tr.columns, index=tr.index),
            pd.DataFrame(sc.transform(te), columns=te.columns, index=te.index))


# ---------------------------------------------------------------------- models
def c_index_risk(y, risk):
    return float(concordance_index_censored(
        y["event"].values.astype(bool), y["time_years"].values, risk)[0])


def fit_aft(kind, X_tr, X_te, y_tr, y_te):
    F = {"weibull": WeibullAFTFitter, "loglogistic": LogLogisticAFTFitter,
         "lognormal": LogNormalAFTFitter}[kind]
    m = F(penalizer=0.1)
    df = X_tr.copy()
    df["time_years"], df["event"] = y_tr["time_years"].values, y_tr["event"].values
    m.fit(df, "time_years", "event")
    # predict_expectation returns expected survival time: higher == lower risk
    return (concordance_index(y_tr["time_years"], m.predict_expectation(X_tr), y_tr["event"]),
            concordance_index(y_te["time_years"], m.predict_expectation(X_te), y_te["event"]))


def fit_rsf(X_tr, X_te, y_tr, y_te, n_estimators=1000):
    m = RandomSurvivalForest(n_estimators=n_estimators, min_samples_split=10,
                             min_samples_leaf=15, max_features="sqrt",
                             n_jobs=-1, random_state=SEED)
    m.fit(X_tr, Surv.from_arrays(y_tr["event"].astype(bool), y_tr["time_years"]))
    return (c_index_risk(y_tr, m.predict(X_tr)), c_index_risk(y_te, m.predict(X_te)))


def fit_xgb(X_tr, X_te, y_tr, y_te, return_model=False):
    def dm(X, y):
        d = xgb.DMatrix(X)
        d.set_float_info("label_lower_bound", y["time_years"].values.astype(float))
        d.set_float_info("label_upper_bound",
                         np.where(y["event"].values == 1, y["time_years"].values,
                                  np.inf).astype(float))
        return d
    dtr, dte = dm(X_tr, y_tr), dm(X_te, y_te)
    m = xgb.train(XGB_PARAMS, dtr, num_boost_round=N_ESTIMATORS, verbose_eval=False)
    ptr, pte = m.predict(dtr), m.predict(dte)
    # AFT predicts log survival TIME -> negate for risk
    res = (c_index_risk(y_tr, -ptr), c_index_risk(y_te, -pte))
    return (res, m, ptr, pte) if return_model else res


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=".", help="dir with the pulled Trillium CSVs")
    p.add_argument("--ptdemog", default="/Users/ishu/research/PTDEMOG_06Aug2026.csv")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()

    D, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sv = pd.read_csv(D / "survival_labels.csv", parse_dates=["baseline_date"])
    sl = pd.read_csv(D / "bsc_longitudinal_slopes.csv")
    t1 = pd.read_csv(D / "features_t1_all.csv").rename(columns={"subject_id": "subject"})

    df = sv.merge(sl, on="subject").merge(t1, on="subject", how="left")
    df = df.reset_index(drop=True)
    y = df[["time_years", "event"]].copy()
    print(f"cohort: {len(df)} subjects | {int(y.event.sum())} converters")

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    base_cols = [c for c in df.columns if c.endswith("_baseline")]
    t1_cols = [c for c in T1_WHITELIST if c in t1.columns]
    missing = [c for c in T1_WHITELIST if c not in t1.columns]
    if missing:
        raise SystemExit(f"T1 whitelist columns absent from {D/'features_t1_all.csv'}: {missing}")
    leaky = [c for c in t1.columns if "n_visits" in c or "n_scans" in c]
    print(f"excluded from T1 block (outcome-related): {leaky}")
    print(f"features available: {len(slope_cols)} slopes | {len(base_cols)} baseline | "
          f"{len(t1_cols)} T1")

    X_slope = df[slope_cols].fillna(df[slope_cols].median())
    X_base = df[base_cols].fillna(df[base_cols].median())
    X_t1 = df[t1_cols].fillna(df[t1_cols].median())

    configs = {
        "Weibull-BL\n(baseline BSC)": ("weibull", "base"),
        "Weibull-20":                 ("weibull", "slope"),
        "LogLogistic-20":             ("loglogistic", "slope"),
        "LogNormal-20":               ("lognormal", "slope"),
        "RSF-20\n(BSC only)":         ("rsf", "slope"),
        "RSF-57\n(BSC+T1)":           ("rsf", "combined"),
        "XGB-57\n(BSC+T1)":           ("xgb", "combined"),
    }

    skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=SEED)
    all_cv = {k: {"train": [], "test": [], "gap": []} for k in configs}

    for fold, (tr, te) in enumerate(skf.split(df, y["event"]), 1):
        print(f"\n--- fold {fold}/{a.folds} ---")
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        sel_s = select_by_penalized_variance(X_slope.iloc[tr], a.top_k)
        sel_b = select_by_penalized_variance(X_base.iloc[tr], a.top_k)

        blocks = {
            "slope": (X_slope[sel_s].iloc[tr], X_slope[sel_s].iloc[te]),
            "base":  (X_base[sel_b].iloc[tr],  X_base[sel_b].iloc[te]),
            "combined": (pd.concat([X_slope[sel_s], X_t1], axis=1).iloc[tr],
                         pd.concat([X_slope[sel_s], X_t1], axis=1).iloc[te]),
        }
        prep = {k: preprocess(v[0], v[1]) for k, v in blocks.items()}

        for name, (kind, block) in configs.items():
            X_tr, X_te = prep[block]
            if kind == "rsf":
                tr_c, te_c = fit_rsf(X_tr, X_te, y_tr, y_te)
            elif kind == "xgb":
                tr_c, te_c = fit_xgb(X_tr, X_te, y_tr, y_te)
            else:
                tr_c, te_c = fit_aft(kind, X_tr, X_te, y_tr, y_te)
            all_cv[name]["train"].append(round(float(tr_c), 4))
            all_cv[name]["test"].append(round(float(te_c), 4))
            all_cv[name]["gap"].append(round(float(tr_c - te_c), 4))
            print(f"  {name.replace(chr(10),' '):26s} train={tr_c:.4f} test={te_c:.4f}")

    json.dump(all_cv, open(out / "all_models_cv.json", "w"), indent=2)

    # ---- single 70/30 split for the KM panels + gain importance ----
    print("\n--- 70/30 split: XGBoost predictions for KM ---")
    itr, ite = train_test_split(df.index, test_size=0.3, random_state=SEED,
                                stratify=y["event"])
    sel_s = select_by_penalized_variance(X_slope.loc[itr], a.top_k)
    Xc = pd.concat([X_slope[sel_s], X_t1], axis=1)
    X_tr, X_te = preprocess(Xc.loc[itr], Xc.loc[ite])
    (tr_c, te_c), model, ptr, pte = fit_xgb(X_tr, X_te, y.loc[itr], y.loc[ite],
                                            return_model=True)
    print(f"  train C={tr_c:.4f}  test C={te_c:.4f}")

    for tag, idx, pred in [("train", itr, ptr), ("", ite, pte)]:
        fn = "xgb_train_predictions.csv" if tag else "xgb_predictions.csv"
        pd.DataFrame({"subject": df.loc[idx, "subject"].values,
                      "true_time": y.loc[idx, "time_years"].values,
                      "event": y.loc[idx, "event"].values,
                      "predicted_risk": pred}).to_csv(out / fn, index=False)

    # gain importance
    lr = json.loads(model.save_raw("json").decode())["learner"]
    fn_, trees = lr["feature_names"], lr["gradient_booster"]["model"]["trees"]
    gain = {}
    for t in trees:
        for i, l in enumerate(t["left_children"]):
            if l == -1:
                continue
            f = fn_[t["split_indices"][i]]
            gain[f] = gain.get(f, 0) + t["loss_changes"][i]
    tot = sum(gain.values())
    gi = (pd.DataFrame({"feature": list(gain), "gain": list(gain.values())})
          .assign(pct=lambda d: 100 * d.gain / tot,
                  block=lambda d: np.where(d.feature.str.endswith("_slope"), "BSC", "T1"))
          .sort_values("gain", ascending=False))
    gi.to_csv(out / "xgb_gain_importance.csv", index=False)
    bsc_pct = gi[gi.block == "BSC"].pct.sum()

    # ---- demographics ----
    dm = pd.read_csv(a.ptdemog, low_memory=False)
    dm = dm[dm.PTGENDER.isin([1, 2])].dropna(subset=["PTDOB"])
    dm = (dm.sort_values("VISDATE").groupby("PTID").first().reset_index()
            [["PTID", "PTGENDER", "PTDOB", "PTEDUCAT"]])
    dm["dob"] = pd.to_datetime(dm.PTDOB, format="%m/%Y", errors="coerce")
    # n_visits / baseline_date appear in both source CSVs, so take them from sv
    c = (df[["subject", "event", "time_years"]]
         .merge(sv[["subject", "n_visits", "baseline_date"]], on="subject", how="left")
         .merge(dm, left_on="subject", right_on="PTID", how="left"))
    c["age"] = (c.baseline_date - c.dob).dt.days / 365.25
    c["female"] = c.PTGENDER == 2
    c["educ"] = pd.to_numeric(c.PTEDUCAT, errors="coerce").where(lambda s: s > 0)
    c.to_csv(out / "cohort_table.csv", index=False)

    def stat(col, f="{:.1f}"):
        g = [c[col], c[c.event == 1][col], c[c.event == 0][col]]
        return "  ".join(f"{f.format(x.mean())}±{f.format(x.std())}" for x in g)

    with open(out / "summary.txt", "w") as fh:
        fh.write("ADNI-Trillium cohort — model ladder (5-fold CV)\n" + "=" * 72 + "\n\n")
        fh.write(f"Subjects {len(df)} | converters {int(y.event.sum())} | "
                 f"censored {int((y.event == 0).sum())}\n\n")
        fh.write(f"{'Model':28s} {'train':>7s} {'test':>16s} {'gap':>7s}\n")
        for k, v in all_cv.items():
            t = np.array(v["test"])
            fh.write(f"{k.replace(chr(10),' '):28s} {np.mean(v['train']):7.4f} "
                     f"{t.mean():8.4f}±{t.std():.4f} {np.mean(v['gap']):7.4f}\n")
        fh.write(f"\n70/30 split XGBoost: train {tr_c:.4f}  test {te_c:.4f}\n")
        fh.write(f"\nGain: BSC slopes {bsc_pct:.1f}%  |  T1 {100-bsc_pct:.1f}%\n")
        fh.write("Top 10 features by gain:\n")
        for _, r in gi.head(10).iterrows():
            fh.write(f"  {r.feature:34s} {r.pct:5.2f}%  [{r.block}]\n")
        fh.write("\nDemographics (all / converters / stable)\n")
        fh.write(f"  Age (yr)        {stat('age')}\n")
        fh.write(f"  Education (yr)  {stat('educ')}\n")
        fh.write(f"  Female n(%)     {c.female.sum()} ({100*c.female.mean():.1f})  "
                 f"{c[c.event==1].female.sum()} ({100*c[c.event==1].female.mean():.1f})  "
                 f"{c[c.event==0].female.sum()} ({100*c[c.event==0].female.mean():.1f})\n")
        fh.write(f"  Follow-up (yr)  {stat('time_years','{:.2f}')}\n")
        fh.write(f"  Visits          {stat('n_visits','{:.2f}')}\n")

    print(open(out / "summary.txt").read())
    print(f"Wrote -> {out.resolve()}")


if __name__ == "__main__":
    main()
