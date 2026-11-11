#!/usr/bin/env python3
"""A5 of PRESPEC_positive_v1.md: the prognostic arm, on untested endpoints.

Conversion to dementia is exhausted in this cohort across four previous rounds
and is deliberately NOT retested here. Two endpoints this project has never
used:

  mortality          All-cause death as a competing risk, from the already
                     built competing_risks_subject_status.csv (417 subjects).
                     Death dates sit a median 0.95 years after the last
                     clinical visit and ADNI3/4 records give the year only, so
                     the time axis is coarse. Stated with the result.

  cognitive decline  Rate of change in the ADNI composite memory and executive
                     z-scores, as a continuous outcome rather than a
                     diagnostic event. A continuous outcome recovers power
                     that dichotomising into conversion throws away, which is
                     the main reason this endpoint is worth trying after the
                     conversion nulls.

Measures are taken at or before the landmark so that nothing here repeats the
guarantee-time bias a reviewer correctly identified in the first submission.
gBSC uses the cohort's own pre-landmark baseline column. cosBSC is rebuilt the
same way, from scans dated at or before each subject's landmark.

Seed stability is mandatory and enforced in code: any interval that excludes
zero at the default seed is rerun at 101, 202 and 303, and a sign change
marks it an artifact. That rule exists because a cosBSC increment in this
project excluded zero at one seed and went negative at three others.

The penalized Cox likelihood ratio test is not used. The L2 penalty
invalidates its null, which is why it was dropped from the manuscript.

Writes a5_prognostic_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

from build_subject_table import parse_date

COVARIATES = ["age_at_landmark", "female", "educ_years", "mmse_at_landmark"]
EQUIVALENCE_MARGIN = 0.03  # C-index, fixed in advance earlier in this project.
SEEDS = [42, 101, 202, 303]


def prelandmark_cos(cos_file: Path, manifest: Path, landmarks: dict[str, dt.datetime]) -> dict[str, float]:
    """Mean cosBSC over each subject's scans dated at or before the landmark."""
    dates: dict[str, dt.datetime] = {}
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            date = parse_date(row["real_date"])
            if date is not None:
                dates[row["image_id"]] = date

    values: dict[str, list[float]] = defaultdict(list)
    with open(cos_file, newline="") as handle:
        for row in csv.DictReader(handle):
            subject = row["subject"]
            landmark = landmarks.get(subject)
            date = dates.get(row["image_id"])
            raw = row.get("bsc_cos_mean", "")
            if landmark is None or date is None or raw in ("", "nan", "NaN"):
                continue
            if date > landmark:
                continue
            try:
                values[subject].append(float(raw))
            except ValueError:
                continue
    return {subject: float(np.mean(vals)) for subject, vals in values.items() if vals}


def cv_cindex(frame: pd.DataFrame, features: list[str], time_col: str, event_col: str,
              seed: int, n_splits: int = 5) -> np.ndarray:
    """Per-fold out-of-fold C-index for a penalized Cox model."""
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    data = frame[features + [time_col, event_col]].dropna()
    for train_idx, test_idx in folds.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        if train[event_col].sum() < 5 or test[event_col].sum() < 2:
            scores.append(np.nan)
            continue
        model = CoxPHFitter(penalizer=0.1)
        try:
            model.fit(train, duration_col=time_col, event_col=event_col)
        except Exception:
            scores.append(np.nan)
            continue
        risk = model.predict_partial_hazard(test)
        scores.append(concordance_index(test[time_col], -risk, test[event_col]))
    return np.asarray(scores, dtype=float)


def fold_matched_increment(frame, base, extended, time_col, event_col, seed, n_boot=2000):
    """Fold-matched delta C-index with a bootstrap interval over folds."""
    base_scores = cv_cindex(frame, base, time_col, event_col, seed)
    full_scores = cv_cindex(frame, extended, time_col, event_col, seed)
    deltas = full_scores - base_scores
    usable = deltas[np.isfinite(deltas)]
    if usable.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(usable, usable.size, replace=True).mean() for _ in range(n_boot)])
    return {
        "base_c": float(np.nanmean(base_scores)),
        "full_c": float(np.nanmean(full_scores)),
        "delta": float(usable.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
    }


def decline_slopes(adnimerge: Path, landmarks: dict[str, dt.datetime],
                   score: str) -> dict[str, float]:
    """Per-subject OLS slope of a composite z-score against years post-landmark."""
    roster = pyreadr.read_r(str(adnimerge / "ROSTER.rda"))["ROSTER"]
    rid_to_ptid = {
        int(r): p for r, p in zip(roster["RID"], roster["PTID"])
        if pd.notna(r) and isinstance(p, str)
    }
    frame = pyreadr.read_r(str(adnimerge / "UWNPSYCHSUM.rda"))["UWNPSYCHSUM"]

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, row in frame.iterrows():
        rid = row.get("RID")
        value = pd.to_numeric(row.get(score), errors="coerce")
        date = parse_date(row.get("EXAMDATE"))
        if pd.isna(rid) or pd.isna(value) or date is None:
            continue
        ptid = rid_to_ptid.get(int(rid))
        landmark = landmarks.get(ptid) if ptid else None
        if landmark is None or date < landmark:
            continue
        series[ptid].append(((date - landmark).days / 365.25, float(value)))

    slopes = {}
    for subject, points in series.items():
        if len(points) < 3:
            continue
        years = np.array([p[0] for p in points])
        values = np.array([p[1] for p in points])
        if years.max() - years.min() < 1.0:
            continue
        slopes[subject] = float(np.polyfit(years, values, 1)[0])
    return slopes


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("/Users/ishu/research/adni_trillium")
    parser.add_argument("--cohort", default=str(base / "analysis/results/spec_v3_harmonized/spec_cohort.csv"))
    parser.add_argument("--competing", default=str(base / "analysis/results/nireports/competing_risks_subject_status.csv"))
    parser.add_argument("--cos_file", default=str(base / "from_cluster/bsc_simple_features_cos.csv"))
    parser.add_argument("--manifest", default=str(base / "from_cluster/manifest_realdates.csv"))
    parser.add_argument("--adnimerge", default="/Users/ishu/research/ADNIMERGE2/data")
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, low_memory=False)
    cohort["landmark_dt"] = cohort["landmark_date"].map(parse_date)
    landmarks = {
        row["subject"]: row["landmark_dt"]
        for _, row in cohort.iterrows()
        if row["landmark_dt"] is not None
    }

    cos_baseline = prelandmark_cos(Path(args.cos_file), Path(args.manifest), landmarks)
    cohort["cos_baseline"] = cohort["subject"].map(cos_baseline)
    print(f"cohort: {len(cohort)} subjects; pre-landmark cosBSC for {cohort['cos_baseline'].notna().sum()}")

    competing = pd.read_csv(args.competing)
    cohort = cohort.merge(competing[["subject", "cr_time", "cr_status"]], on="subject", how="left")
    # cr_status: 1 conversion, 2 death before dementia (coding from the
    # nireports build). Mortality endpoint is death as the event of interest.
    cohort["death_event"] = (cohort["cr_status"] == 2).astype(int)

    results = []

    print("\nENDPOINT 1: all-cause mortality before dementia")
    print(f"  events: {int(cohort['death_event'].sum())} of {len(cohort)}")
    if cohort["death_event"].sum() >= 20:
        for measure, label in (("bsc_dir_mean_baseline", "gBSC baseline"),
                               ("cos_baseline", "cosBSC baseline")):
            if measure not in cohort.columns:
                continue
            per_seed = []
            for seed in SEEDS:
                out = fold_matched_increment(
                    cohort, COVARIATES, COVARIATES + [measure],
                    "cr_time", "death_event", seed)
                if out is None:
                    continue
                per_seed.append((seed, out))
                if seed == SEEDS[0]:
                    print(f"  {label:16s} cov C={out['base_c']:.3f} -> {out['full_c']:.3f}"
                          f"  delta {out['delta']:+.4f} [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]")
            if per_seed:
                deltas = [o["delta"] for _, o in per_seed]
                signs = {np.sign(d) for d in deltas}
                stable = len(signs) == 1
                print(f"  {' ':16s} seeds {[round(d, 4) for d in deltas]}"
                      f"  sign stable: {stable}")
                first = per_seed[0][1]
                excludes_zero = first["ci_lo"] > 0 or first["ci_hi"] < 0
                results.append({
                    "endpoint": "mortality", "measure": label,
                    "n": int(len(cohort)), "events": int(cohort["death_event"].sum()),
                    "base_c": round(first["base_c"], 4), "full_c": round(first["full_c"], 4),
                    "delta": round(first["delta"], 4),
                    "ci_lo": round(first["ci_lo"], 4), "ci_hi": round(first["ci_hi"], 4),
                    "seed_deltas": ";".join(f"{d:.4f}" for d in deltas),
                    "sign_stable": int(stable),
                    "verdict": ("artifact: sign flips across seeds" if excludes_zero and not stable
                                else "positive" if excludes_zero and stable
                                else f"null, within the {EQUIVALENCE_MARGIN} margin"
                                if abs(first["delta"]) < EQUIVALENCE_MARGIN else "null"),
                })
    else:
        print("  too few death events for a survival model")

    print("\nENDPOINT 2: rate of cognitive decline (continuous)")
    for score in ("ADNI_MEM", "ADNI_EF"):
        slopes = decline_slopes(Path(args.adnimerge), landmarks, score)
        cohort[f"slope_{score}"] = cohort["subject"].map(slopes)
        n_slopes = cohort[f"slope_{score}"].notna().sum()
        print(f"  {score}: slopes for {n_slopes} subjects "
              f"(median {cohort[f'slope_{score}'].median():+.4f} z per year)")
        if n_slopes < 100:
            continue
        import statsmodels.api as sm
        for measure, label in (("bsc_dir_mean_baseline", "gBSC baseline"),
                               ("cos_baseline", "cosBSC baseline")):
            if measure not in cohort.columns:
                continue
            usable = cohort.dropna(subset=[f"slope_{score}", measure] + COVARIATES)
            if len(usable) < 100:
                continue
            exog = sm.add_constant(usable[[measure] + COVARIATES].astype(float))
            model = sm.OLS(usable[f"slope_{score}"].astype(float), exog).fit()
            # Standardized so it is comparable to A2 and A3.
            sd_x = usable[measure].astype(float).std(ddof=1)
            sd_y = usable[f"slope_{score}"].astype(float).std(ddof=1)
            beta_std = float(model.params[measure] * sd_x / sd_y)
            pvalue = float(model.pvalues[measure])
            print(f"    {label:16s} n={len(usable):4d} beta_std={beta_std:+.4f} p={pvalue:.3g}")
            results.append({"endpoint": f"decline_{score}", "measure": label,
                            "n": int(len(usable)), "events": "",
                            "delta": round(beta_std, 4), "p": pvalue,
                            "verdict": "positive" if pvalue < 0.05 else "null"})

    out_path = Path(args.out_dir) / "a5_prognostic_results.csv"
    fields = ["endpoint", "measure", "n", "events", "base_c", "full_c", "delta",
              "ci_lo", "ci_hi", "p", "seed_deltas", "sign_stable", "verdict"]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
