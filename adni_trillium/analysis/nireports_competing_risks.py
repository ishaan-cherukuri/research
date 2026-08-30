"""Competing-risks sensitivity analysis for the NeuroImage: Reports revision.

The cohort has a mean age of 76.9 years and multi-year follow-up, so death is a
real competing event and treating it as plain censoring biases every C-index.
This script does three things and nothing more:

  1. Counts deaths in the frozen 417-subject cohort, separating deaths that
     precede a dementia diagnosis from deaths that follow one.
  2. Refits the covariates-only, regional-BSC-only and full configurations under
     a cause-specific framing, where a subject known to have died without a
     dementia diagnosis is followed to the death date and then censored there.
  3. Fits a Fine-Gray subdistribution-hazard model to the out-of-fold risk score
     of each configuration, so the same comparison is available on the
     subdistribution scale.

Death ascertainment. ADNI records death in two places and they do not overlap.
TREATDIS carries a withdrawal reason with an exact form date and covers ADNI1,
ADNIGO and ADNI2; the reason code for death is 2 in the ADNI1 dictionary and 1
in every later one, so the code is mapped per protocol. ADVERSE carries
AEHDTHDT and covers ADNI3 and ADNI4, but that field is de-identified to a year,
so those deaths are placed at the midpoint of the reported year. Both facts are
stated in the manuscript.

Nothing here rebuilds the cohort or refits a model reported elsewhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from sklearn.model_selection import StratifiedKFold

from run_spec_models import (SEED, cindex, feature_sets, fit_predict,
                             preprocess, surv_y)

CONFIGS = [
    ("F0_covariates", "Clinical covariates"),
    ("F3_regional_slopes", "Regional BSC slopes"),
    ("F9b_cov_regional_stdmri", "Covariates + regional BSC + standard MRI"),
]


def load_rda(data_dir: Path, name: str) -> pd.DataFrame:
    obj = pyreadr.read_r(str(data_dir / f"{name}.rda"))
    return obj[list(obj)[0]]


def death_records(adni_dir: Path) -> pd.DataFrame:
    """One row per subject with a death record, with the source recorded."""
    td = load_rda(adni_dir, "TREATDIS")
    # ADNI1 coded death as 2; ADNIGO onward code it as 1. WDREASON can carry
    # several colon-separated reasons, so the code is matched as a member.
    def is_death(row) -> bool:
        want = "2" if str(row.COLPROT).upper() == "ADNI1" else "1"
        return want in str(row.WDREASON).split(":")

    td = td[td.apply(is_death, axis=1)].copy()
    td["death_date"] = pd.to_datetime(td["EXAMDATE"], errors="coerce")
    td = td.dropna(subset=["death_date"])
    a = (td.groupby("PTID")["death_date"].min().reset_index()
           .assign(death_source="TREATDIS withdrawal form", death_precision="exact date"))

    ae = load_rda(adni_dir, "ADVERSE")
    ae = ae[ae["AEHDTHDT"].notna()].copy()
    # AEHDTHDT is de-identified to a year in ADNI3 and ADNI4. Midpoint of the
    # reported year is the least-committal placement.
    yr = pd.to_numeric(ae["AEHDTHDT"], errors="coerce")
    ae = ae[yr.between(1990, 2030)]
    ae["death_date"] = pd.to_datetime(
        yr[yr.between(1990, 2030)].astype(int).astype(str) + "-07-01")
    b = (ae.groupby("PTID")["death_date"].min().reset_index()
           .assign(death_source="ADVERSE AEHDTHDT", death_precision="year only"))

    both = pd.concat([a, b], ignore_index=True)
    both = both.sort_values("death_precision").drop_duplicates("PTID", keep="first")
    return both.rename(columns={"PTID": "subject"})


def attach_deaths(df: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    """Add competing-risk time and status columns to the frozen cohort."""
    out = df.merge(deaths, on="subject", how="left")
    lm = pd.to_datetime(out["landmark_date"], errors="coerce")
    out["t_death"] = (out["death_date"] - lm).dt.days / 365.25

    # A death recorded before the landmark scan is a data error, not an event.
    bad = out["t_death"].notna() & (out["t_death"] <= 0)
    out.loc[bad, ["death_date", "t_death"]] = pd.NaT, np.nan

    # Deaths are almost always recorded after the last clinical visit, so a
    # subject known to have died is followed to the death date. A year-only
    # death date that lands before the last visit is pushed to that visit.
    out["t_death"] = np.where(out["t_death"].notna(),
                              np.maximum(out["t_death"], out["time_years"]),
                              np.nan)

    conv = out["event"].astype(bool)
    died = out["t_death"].notna()
    # Death after a recorded conversion is not a competing event: the outcome
    # of interest already happened.
    out["cr_status"] = np.where(conv, 1, np.where(died, 2, 0))
    out["cr_time"] = np.where(conv, out["time_years"],
                              np.where(died, out["t_death"], out["time_years"]))
    return out


def cs_surv_y(df: pd.DataFrame) -> np.ndarray:
    """Cause-specific outcome: conversion is the event, death censors."""
    return np.array([(s == 1, float(t))
                     for s, t in zip(df["cr_status"], df["cr_time"])],
                    dtype=[("event", "?"), ("time", "<f8")])


def cv_risk(df, cols, model, y_fn):
    """5-fold CV C-index and pooled out-of-fold risk, on the given outcome."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds, oof = [], []
    strat = (df["cr_status"] == 1).astype(int)
    for a, b in skf.split(df, strat):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        ytr, yte = y_fn(tr), y_fn(te)
        rtr, rte, _ = fit_predict(model, Xtr, ytr, Xte)
        flip = -1.0 if cindex(ytr, rtr) < 0.5 else 1.0
        rte = rte * flip
        folds.append(cindex(yte, rte))
        oof.append(pd.DataFrame({"subject": te["subject"].to_numpy(),
                                 "risk": rte}))
    return (float(np.mean(folds)), float(np.std(folds, ddof=1)),
            pd.concat(oof, ignore_index=True))


def _fg_expand(time, status, x, n_grid=60):
    """Geskus's risk-set expansion for the subdistribution hazard.

    Subjects who experience the competing event stay in the risk set past their
    event time, carrying the time-varying weight G(t)/G(T_i) where G is the
    Kaplan-Meier estimate of the censoring distribution. With no competing
    events every weight is one and the expansion reduces to the ordinary
    cause-specific Cox model.
    """
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter().fit(time, (status == 0).astype(int))
    def G(t):
        return float(np.clip(np.asarray(kmf.predict(t)).item(), 1e-8, None))

    ev_times = np.unique(time[status == 1])
    if len(ev_times) > n_grid:
        ev_times = np.unique(np.quantile(ev_times, np.linspace(0, 1, n_grid)))
    tmax = float(time.max())

    rows = []
    for i in range(len(time)):
        if status[i] != 2:
            rows.append((i, 0.0, time[i], int(status[i] == 1), 1.0, x[i]))
            continue
        # Up to the competing event the subject is at risk with weight one.
        rows.append((i, 0.0, time[i], 0, 1.0, x[i]))
        gi = G(time[i])
        start = float(time[i])
        for stop in np.unique(np.concatenate([ev_times[ev_times > start], [tmax]])):
            if stop <= start:
                continue
            w = G(stop) / gi
            if w > 1e-6:
                rows.append((i, start, float(stop), 0, w, x[i]))
            start = float(stop)

    d = pd.DataFrame(rows, columns=["id", "start", "stop", "event", "w", "x"])
    return d[d["stop"] > d["start"]]


def _fg_coef(time, status, x):
    from lifelines import CoxTimeVaryingFitter
    d = _fg_expand(time, status, x)
    f = CoxTimeVaryingFitter(penalizer=0.01).fit(
        d, id_col="id", start_col="start", stop_col="stop",
        event_col="event", weights_col="w")
    return float(f.params_["x"])


def fine_gray(time, status, x, n_boot=400, seed=SEED):
    """Subdistribution hazard ratio per SD, with a subject-level bootstrap CI.

    The weighted partial likelihood gives the right point estimate but the wrong
    standard error, since the weights are estimated. Resampling subjects and
    refitting sidesteps that.
    """
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    x = np.asarray(x, float)

    beta = _fg_coef(time, status, x)
    rng = np.random.default_rng(seed)
    draws = []
    n = len(time)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if (status[idx] == 1).sum() < 20 or (status[idx] == 2).sum() < 3:
            continue
        try:
            draws.append(_fg_coef(time[idx], status[idx], x[idx]))
        except Exception:
            continue
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Two-sided bootstrap p-value for beta = 0.
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"shr": float(np.exp(beta)),
            "shr_lo": float(np.exp(lo)), "shr_hi": float(np.exp(hi)),
            "p": float(min(1.0, p)), "n_boot": int(len(draws))}


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--adni_dir", default="/Users/ishu/research/ADNIMERGE2/data")
    ap.add_argument("--out_dir", default=str(root / "results/nireports"))
    ap.add_argument("--model", default="xgb_aft")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)

    deaths = death_records(Path(args.adni_dir))
    df = attach_deaths(df, deaths)

    n_death_any = int(df["death_date"].notna().sum())
    n_competing = int((df["cr_status"] == 2).sum())
    n_after_conv = int(((df["event"] == 1) & df["death_date"].notna()).sum())
    n_censored = int((df["event"] == 0).sum())
    counts = {
        "n_subjects": len(df),
        "n_conversions": int((df["cr_status"] == 1).sum()),
        "n_with_death_record": n_death_any,
        "n_death_before_conversion": n_competing,
        "n_death_after_conversion": n_after_conv,
        "n_censored_original": n_censored,
        "pct_of_censored_who_died": 100.0 * n_competing / n_censored,
        "n_death_exact_date": int((df["death_precision"] == "exact date").sum()),
        "n_death_year_only": int((df["death_precision"] == "year only").sum()),
        "median_years_landmark_to_death": float(
            df.loc[df["cr_status"] == 2, "cr_time"].median()),
        "median_extra_followup_years": float(
            (df.loc[df["cr_status"] == 2, "cr_time"]
             - df.loc[df["cr_status"] == 2, "time_years"]).median()),
    }
    print(json.dumps(counts, indent=2))

    sets = feature_sets(df)
    rows = []
    for key, label in CONFIGS:
        cols = [c for c in sets[key] if c in df.columns]
        orig_m, orig_s, _ = cv_risk(df, cols, args.model, surv_y)
        cs_m, cs_s, oof = cv_risk(df, cols, args.model, cs_surv_y)
        merged = df[["subject", "cr_time", "cr_status"]].merge(oof, on="subject")
        z = (merged["risk"] - merged["risk"].mean()) / merged["risk"].std()
        fg = fine_gray(merged["cr_time"], merged["cr_status"], z)
        rec = {"config": key, "label": label, "n_features": len(cols),
               "model": args.model,
               "cindex_original": orig_m, "cindex_original_sd": orig_s,
               "cindex_cause_specific": cs_m, "cindex_cause_specific_sd": cs_s,
               "cindex_change": cs_m - orig_m}
        rec.update({f"fg_{k}": v for k, v in fg.items()})
        rows.append(rec)
        print(f"{label:44s} orig={orig_m:.3f} cause-specific={cs_m:.3f} "
              f"FG sHR={fg['shr']:.2f} ({fg['shr_lo']:.2f}-{fg['shr_hi']:.2f})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "competing_risks.csv", index=False)
    with open(out / "competing_risks.json", "w") as f:
        json.dump({"seed": SEED, "counts": counts, "results": rows}, f, indent=2)
    df[["subject", "event", "time_years", "cr_time", "cr_status",
        "death_source", "death_precision"]].to_csv(
        out / "competing_risks_subject_status.csv", index=False)
    print(f"\nwrote {out/'competing_risks.csv'}")


if __name__ == "__main__":
    main()
