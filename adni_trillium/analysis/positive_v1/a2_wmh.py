#!/usr/bin/env python3
"""A2 of PRESPEC_positive_v1.md: does boundary sharpness track WMH burden?

Rationale and sources are in the pre-spec. In short, the gray/white contrast
literature reports the positives this project keeps failing to find, and the
mechanism those papers give for age-related contrast loss is rising white
matter water content pulling white matter signal toward gray. White matter
hyperintensity burden measures that process directly, so if gBSC indexes
tissue contrast at all, this is where it should show.

Primary test, fixed in advance: one linear model, cortex-wide mean gBSC
regressed on log WMH volume, adjusted for age, sex, intracranial volume, field
strength and scanner site. One measure, one coefficient, two-sided p < 0.05.

WMH comes from ADNI's UCD lab on FLAIR, a different modality processed by a
different group, so there is no shared-pipeline circularity with our Atropos
gradient. The join is same-session for most subjects (median gap 0 days).

Three guards, all pre-specified rather than added afterwards:
  3T-only repeat      Site, vendor and field strength are confounded with case
                      mix in ADNI, and this project's own batch sweep showed
                      only site-aware batches move the scanner effect. A
                      coefficient that does not survive within the 3T stratum
                      is reported as scanner, not biology.
  split-half          200 stratified splits, coefficient fit in one half and
                      re-estimated in the other. The full-sample estimate is
                      not the headline if it does not hold out.
  attenuation note    A1 put the same-scanner ICC of gBSC at 0.843, so an
                      observed correlation is attenuated by about sqrt(0.843).
                      The disattenuated bound is printed alongside.

Secondaries run only if the primary is significant, in the pre-spec order.
Benjamini-Hochberg is step-up, computed with the reversed-cummin idiom,
because a forward cummin in this project once reported 158 of 159 features
surviving when the answer was 2.

Writes a2_wmh_results.csv and a2_wmh_regional.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PRIMARY_MEASURE = "bsc_dir_mean"
SECONDARY_MEASURES = ["bsc_cos_mean", "bscdir_adsig", "cos_adsig", "bsc_mag_mean"]
# Reported for contrast, not as a BSC claim: a measure from the same pipeline
# that is expected to track WMH, so a null on every row would suggest the
# covariate block or the join is at fault rather than the measure.
REFERENCE_MEASURES = ["seg_bpf"]


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Step-up BH. The direction matters: cummin runs from the LARGEST p down."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def design(frame: pd.DataFrame, predictor: str, covariates: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix, dropping single-level categoricals."""
    parts = [frame[[predictor]].astype(float)]
    for name in covariates:
        if name == "site":
            dummies = pd.get_dummies(frame["site"], prefix="site", drop_first=True, dtype=float)
            if dummies.shape[1]:
                parts.append(dummies)
        elif name == "field_strength":
            dummies = pd.get_dummies(
                frame["field_strength"], prefix="fs", drop_first=True, dtype=float
            )
            if dummies.shape[1]:
                parts.append(dummies)
        else:
            parts.append(frame[[name]].astype(float))
    matrix = pd.concat(parts, axis=1)
    return sm.add_constant(matrix, has_constant="add"), matrix.columns


def fit_one(frame: pd.DataFrame, outcome: str, predictor: str, covariates: list[str]):
    exog, _ = design(frame, predictor, covariates)
    endog = frame[outcome].astype(float)
    model = sm.OLS(endog, exog, missing="drop").fit()
    return model


def standardized_beta(model, frame: pd.DataFrame, outcome: str, predictor: str) -> float:
    """Coefficient in outcome SD units per 1 SD of the predictor."""
    sd_x = frame[predictor].astype(float).std(ddof=1)
    sd_y = frame[outcome].astype(float).std(ddof=1)
    if not sd_y:
        return float("nan")
    return float(model.params[predictor] * sd_x / sd_y)


def run_test(frame: pd.DataFrame, outcome: str, predictor: str, covariates: list[str], label: str):
    usable = frame.dropna(subset=[outcome, predictor] + [c for c in covariates if c not in ("site", "field_strength")])
    if len(usable) < 30:
        return {"analysis": label, "measure": outcome, "n": len(usable), "note": "too few subjects"}
    model = fit_one(usable, outcome, predictor, covariates)
    beta = float(model.params[predictor])
    se = float(model.bse[predictor])
    return {
        "analysis": label,
        "measure": outcome,
        "n": int(model.nobs),
        "beta": round(beta, 6),
        "se": round(se, 6),
        "ci_lo": round(beta - 1.96 * se, 6),
        "ci_hi": round(beta + 1.96 * se, 6),
        "beta_std": round(standardized_beta(model, usable, outcome, predictor), 4),
        "p": float(model.pvalues[predictor]),
        "r2": round(float(model.rsquared), 4),
        "note": "",
    }


def split_half(frame: pd.DataFrame, outcome: str, predictor: str, covariates: list[str],
               n_splits: int, seed: int):
    """Fit in one random half, re-estimate in the other. Reports held-out betas."""
    rng = np.random.default_rng(seed)
    usable = frame.dropna(subset=[outcome, predictor, "age", "female", "seg_tiv_mm3"])
    index = np.asarray(usable.index)
    train_betas, held_betas, held_p = [], [], []
    for _ in range(n_splits):
        shuffled = rng.permutation(index)
        half = len(shuffled) // 2
        for first, second in ((shuffled[:half], shuffled[half:]),):
            try:
                model_a = fit_one(usable.loc[first], outcome, predictor, covariates)
                model_b = fit_one(usable.loc[second], outcome, predictor, covariates)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if predictor not in model_a.params or predictor not in model_b.params:
                continue
            train_betas.append(float(model_a.params[predictor]))
            held_betas.append(float(model_b.params[predictor]))
            held_p.append(float(model_b.pvalues[predictor]))
    if not train_betas:
        return None
    train_betas = np.asarray(train_betas)
    held_betas = np.asarray(held_betas)
    held_p = np.asarray(held_p)
    same_sign = float(np.mean(np.sign(held_betas) == np.sign(train_betas)))
    return {
        "n_splits": len(train_betas),
        "train_beta_mean": round(float(train_betas.mean()), 6),
        "held_beta_mean": round(float(held_betas.mean()), 6),
        "held_beta_lo": round(float(np.percentile(held_betas, 2.5)), 6),
        "held_beta_hi": round(float(np.percentile(held_betas, 97.5)), 6),
        "shrinkage": round(float(held_betas.mean() / train_betas.mean()), 4)
        if train_betas.mean() else float("nan"),
        "frac_same_sign": round(same_sign, 3),
        "frac_held_significant": round(float(np.mean(held_p < 0.05)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(Path(__file__).parent / "subject_table.csv"))
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    parser.add_argument("--regional_file",
                        default="/Users/ishu/research/adni_trillium/from_cluster/per_scan_regional.csv")
    parser.add_argument("--max_gap_days", type=int, default=365)
    parser.add_argument("--n_splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--icc", type=float, default=0.843,
                        help="same-scanner ICC from A1, used for the attenuation bound")
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    frame = frame[frame["total_wmh"].notna()]
    frame = frame[frame["wmh_gap_days"] <= args.max_gap_days]
    # Right-skewed with true zeros, so log1p rather than log.
    frame["log_wmh"] = np.log1p(frame["total_wmh"].astype(float))
    frame["field_strength"] = frame["field_strength"].fillna("unknown").astype(str)
    frame["site"] = frame["site"].astype(str)

    covariates = ["age", "female", "seg_tiv_mm3", "field_strength", "site"]
    print(f"A2 cohort: {len(frame)} subjects with WMH within {args.max_gap_days}d")
    print(f"  WMH mL: median {frame['total_wmh'].median():.2f}, "
          f"IQR {frame['total_wmh'].quantile(.25):.2f} to {frame['total_wmh'].quantile(.75):.2f}")
    print(f"  field strength: {dict(frame['field_strength'].value_counts())}")
    print(f"  sites: {frame['site'].nunique()}")
    print()

    results = []

    print("PRIMARY (pre-specified): cortex-wide gBSC on log WMH, adjusted")
    primary = run_test(frame, PRIMARY_MEASURE, "log_wmh", covariates, "primary_full")
    results.append(primary)
    print(f"  n={primary['n']}  beta={primary['beta']:.5g}  "
          f"beta_std={primary['beta_std']:+.4f} SD per SD  p={primary['p']:.3g}")

    significant = primary.get("p", 1.0) < 0.05
    disattenuated = primary["beta_std"] / np.sqrt(args.icc) if significant else None
    if disattenuated is not None:
        print(f"  disattenuated for ICC {args.icc}: {disattenuated:+.4f} SD per SD")

    print("\nGUARD 1, 3T-only repeat")
    three_t = frame[frame["field_strength"].str.startswith("3")]
    guard = run_test(three_t, PRIMARY_MEASURE, "log_wmh",
                     ["age", "female", "seg_tiv_mm3", "site"], "primary_3T_only")
    results.append(guard)
    if "beta" in guard:
        print(f"  n={guard['n']}  beta_std={guard['beta_std']:+.4f}  p={guard['p']:.3g}")
    else:
        print(f"  {guard['note']}")

    print("\nGUARD 2, split-half stability")
    stability = split_half(frame, PRIMARY_MEASURE, "log_wmh", covariates,
                           args.n_splits, args.seed)
    if stability:
        print(f"  {stability['n_splits']} splits  train beta {stability['train_beta_mean']:.5g}"
              f"  held-out {stability['held_beta_mean']:.5g}"
              f" [{stability['held_beta_lo']:.5g}, {stability['held_beta_hi']:.5g}]")
        print(f"  shrinkage {stability['shrinkage']}  same sign {stability['frac_same_sign']}"
              f"  held-out significant {stability['frac_held_significant']}")
        results.append({"analysis": "split_half", "measure": PRIMARY_MEASURE,
                        "n": stability["n_splits"], "beta": stability["held_beta_mean"],
                        "ci_lo": stability["held_beta_lo"], "ci_hi": stability["held_beta_hi"],
                        "note": f"shrinkage={stability['shrinkage']}, "
                                f"same_sign={stability['frac_same_sign']}, "
                                f"held_sig={stability['frac_held_significant']}"})

    print("\nSECONDARY and reference measures")
    for measure in SECONDARY_MEASURES + REFERENCE_MEASURES:
        if measure not in frame.columns:
            continue
        tag = "secondary" if measure in SECONDARY_MEASURES else "reference"
        if tag == "secondary" and not significant:
            results.append({"analysis": "secondary_not_run", "measure": measure,
                            "note": "primary not significant; pre-spec forbids running this"})
            print(f"  {measure:16s} not run, primary was not significant")
            continue
        row = run_test(frame, measure, "log_wmh", covariates, tag)
        results.append(row)
        if "beta" in row:
            print(f"  {measure:16s} n={row['n']:4d} beta_std={row['beta_std']:+.4f} p={row['p']:.3g}")

    # Regional screen, only when the primary clears.
    regional_rows = []
    if significant:
        print("\nREGIONAL screen (68 parcels, BH step-up)")
        regional = pd.read_csv(args.regional_file)
        merged = frame.merge(regional, on="image_id", how="inner", suffixes=("", "_reg"))
        parcels = [c for c in regional.columns if c.startswith("bscdir_roi")]
        rows = []
        for parcel in parcels:
            sub = merged.dropna(subset=[parcel, "log_wmh", "age", "female", "seg_tiv_mm3"])
            if len(sub) < 100:
                continue
            model = fit_one(sub, parcel, "log_wmh", covariates)
            rows.append({"parcel": parcel, "n": int(model.nobs),
                         "beta_std": standardized_beta(model, sub, parcel, "log_wmh"),
                         "p": float(model.pvalues["log_wmh"])})
        if rows:
            qvalues = benjamini_hochberg(np.array([r["p"] for r in rows]))
            for row, q in zip(rows, qvalues):
                row["q"] = q
            rows.sort(key=lambda r: r["p"])
            regional_rows = rows
            survived = sum(1 for r in rows if r["q"] < 0.05)
            print(f"  {len(rows)} parcels tested, {survived} survive BH at q<0.05")
            for row in rows[:8]:
                print(f"    {row['parcel']:18s} beta_std={row['beta_std']:+.4f} "
                      f"p={row['p']:.3g} q={row['q']:.3g}")
    else:
        print("\nREGIONAL screen not run: pre-spec allows it only after a significant primary.")

    out_dir = Path(args.out_dir)
    keys = ["analysis", "measure", "n", "beta", "se", "ci_lo", "ci_hi", "beta_std", "p", "r2", "note"]
    with open(out_dir / "a2_wmh_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    if regional_rows:
        with open(out_dir / "a2_wmh_regional.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["parcel", "n", "beta_std", "p", "q"])
            writer.writeheader()
            writer.writerows(regional_rows)
    print(f"\nWrote {out_dir / 'a2_wmh_results.csv'}")


if __name__ == "__main__":
    main()
