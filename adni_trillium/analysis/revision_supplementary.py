"""Supplementary analyses for the ALZ-26-0928 revision (Trillium cohort).

Answers four reviewer questions the CV grid alone does not:

1. Does the BSC block add discrimination once covariates and T1 morphometry are
   already in the model? Fold-matched differences with a bootstrap interval.
2. What are the adjusted hazard ratios, and does the BSC block improve the
   partial likelihood over covariates alone?
3. Is BSC driven by acquisition rather than by biology? Field strength, SNR, and
   a subgroup whose field strength never changes during follow-up.
4. Does the BSC slope differ between converters and non-converters at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from run_revision_models import (cindex, fit_predict, fold_columns, get_blocks,
                                 prep_fold, select_slopes)


def paired_increment(df, blocks, base_set, full_set, model, top_k, folds, seed,
                     n_boot=2000):
    y_e, y_t = df["event"].to_numpy(), df["time_years"].to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def score(fs, tr_idx, te_idx):
        cols = fold_columns(df, blocks, fs, tr_idx, top_k)
        Xtr, Xte = prep_fold(df.iloc[tr_idx][cols], df.iloc[te_idx][cols])
        _, rte = fit_predict(model, Xtr, y_e[tr_idx], y_t[tr_idx], Xte, seed)
        return cindex(y_e[te_idx], y_t[te_idx], rte)

    diffs = np.array([score(full_set, tr, te) - score(base_set, tr, te)
                      for tr, te in skf.split(df, y_e)])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n_boot)]
    _, p_val = stats.ttest_1samp(diffs, 0.0)
    return {"model": model, "base": "+".join(base_set), "full": "+".join(full_set),
            "fold_diffs": [round(float(d), 4) for d in diffs],
            "mean_diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                     round(float(np.percentile(boot, 97.5)), 4)],
            "paired_t_p": round(float(p_val), 4)}


def adjusted_cox(df, blocks, top_k):
    slopes = select_slopes(df, blocks["bsc"], top_k)
    cov = blocks["cov"]
    d = df[cov + slopes + ["event", "time_years"]].copy()
    d[cov + slopes] = d[cov + slopes].fillna(d[cov + slopes].median())
    for c in cov + slopes:
        sd = d[c].std(ddof=0)
        d[c] = (d[c] - d[c].mean()) / (sd if sd > 0 else 1.0)
    d["time_years"] = d["time_years"].clip(lower=1e-3)

    full = CoxPHFitter(penalizer=0.1).fit(d, duration_col="time_years",
                                          event_col="event")
    cov_only = CoxPHFitter(penalizer=0.1).fit(d[cov + ["event", "time_years"]],
                                              duration_col="time_years",
                                              event_col="event")
    s = full.summary[["coef", "exp(coef)", "exp(coef) lower 95%",
                      "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["coef", "hr", "hr_lo", "hr_hi", "p"]
    s["block"] = ["covariate" if i in cov else "bsc_slope" for i in s.index]
    lr = 2 * (full.log_likelihood_ - cov_only.log_likelihood_)
    ph = proportional_hazard_test(full, d, time_transform="rank")
    return s.sort_values("p"), {
        "concordance_full": round(float(full.concordance_index_), 3),
        "concordance_cov_only": round(float(cov_only.concordance_index_), 3),
        "lr_stat_bsc_block": round(float(lr), 2),
        "lr_df": len(slopes),
        "lr_p_bsc_block": round(float(stats.chi2.sf(lr, df=len(slopes))), 4),
        "min_ph_assumption_p": round(float(ph.summary["p"].min()), 4)}


def acquisition(df):
    out = {}
    fs = pd.to_numeric(df.get("meta_field_strength_t_bl"), errors="coerce")
    if fs is not None:
        fs = fs.where(fs < 100, fs / 10000.0).round(1)
    for name, col in (("bsc_mag_mean_baseline", "bsc_mag_mean_baseline"),
                      ("bsc_mag_mean_slope", "bsc_mag_mean_slope")):
        v = pd.to_numeric(df.get(col), errors="coerce")
        if v is None or v.isna().all():
            continue
        if fs is not None:
            a, b = v[np.isclose(fs, 1.5)].dropna(), v[np.isclose(fs, 3.0)].dropna()
            if len(a) > 5 and len(b) > 5:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                out[f"{name}_by_field_strength"] = {
                    "n_1p5T": int(len(a)), "n_3T": int(len(b)),
                    "mean_1p5T": round(float(a.mean()), 4),
                    "mean_3T": round(float(b.mean()), 4),
                    "welch_p": float(f"{p:.3g}"),
                    "cohens_d": round(float((b.mean() - a.mean()) / pooled), 3)}
        snr = pd.to_numeric(df.get("qc_snr_bl"), errors="coerce")
        if snr is not None:
            ok = snr.notna() & v.notna()
            r, p = stats.pearsonr(snr[ok], v[ok])
            out[f"{name}_vs_snr"] = {"r": round(float(r), 3),
                                     "p": float(f"{p:.3g}"), "n": int(ok.sum())}

    if "field_strength_changed" in df.columns:
        ch = pd.to_numeric(df["field_strength_changed"], errors="coerce").fillna(0)
        out["field_strength_changed_during_window"] = {
            "n": int(ch.sum()), "pct": round(100.0 * float(ch.mean()), 1)}
    return out


def group_differences(df, blocks, top_k):
    rows = []
    for c in select_slopes(df, blocks["bsc"], top_k) + blocks["cov"]:
        a = pd.to_numeric(df.loc[df.event == 1, c], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df.event == 0, c], errors="coerce").dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"feature": c, "mean_converters": float(a.mean()),
                     "mean_stable": float(b.mean()),
                     "cohens_d": round(float((a.mean() - b.mean()) / pooled), 3)
                     if pooled > 0 else np.nan,
                     "p": float(p)})
    t = pd.DataFrame(rows).sort_values("p")
    # Benjamini-Hochberg across the tested features.
    m = len(t)
    scaled = t["p"].to_numpy() * m / np.arange(1, m + 1)
    t["q"] = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return t


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"n": len(df), "events": int(df["event"].sum())}

    print("Incremental value of the BSC block")
    inc = []
    for model in ("weibull", "coxnet", "rsf"):
        for base, full in ((["cov"], ["cov", "bsc"]),
                           (["cov", "t1x", "t1long"],
                            ["cov", "bsc", "t1x", "t1long"])):
            r = paired_increment(df, blocks, base, full, model, args.top_k,
                                 args.folds, args.seed)
            inc.append(r)
            print(f"  {model:<9} {r['base']:<18} -> +bsc  delta {r['mean_diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  p={r['paired_t_p']}")
    res["incremental"] = inc

    print("\nAdjusted Cox model")
    tbl, meta = adjusted_cox(df, blocks, args.top_k)
    tbl.to_csv(out_dir / "cox_adjusted.csv")
    res["cox"] = meta
    print("  " + json.dumps(meta))
    print(tbl.head(8).to_string())

    print("\nConverter vs stable differences")
    gd = group_differences(df, blocks, args.top_k)
    gd.to_csv(out_dir / "group_differences.csv", index=False)
    print(gd.head(8).to_string())
    res["n_bsc_slopes_q_below_0.05"] = int(
        ((gd["q"] < 0.05) & (gd["feature"].str.endswith("_slope"))).sum())

    print("\nAcquisition sensitivity")
    res["acquisition"] = acquisition(df)
    print(json.dumps(res["acquisition"], indent=2))

    with open(out_dir / "supplementary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {out_dir/'supplementary.json'}")


if __name__ == "__main__":
    main()
