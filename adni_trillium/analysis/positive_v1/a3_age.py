#!/usr/bin/env python3
"""A3 of PRESPEC_positive_v1.md: the age anchor.

This is not a discovery test. Loss of gray/white contrast with age is the most
reproducible finding in this literature: Salat et al. 2009 (Neurobiology of
Aging, PMC2750073) report it across most of the cortex independently of
cortical thinning, a 2025 Scientific Reports paper (s41598-025-34355-y)
reproduces the age-related decline, and the Parkinson's work in PMC5881246
describes gray/white contrast as a very sensitive measure for detecting aging
effects. The mechanism proposed is rising white matter water content pulling
white matter signal toward gray.

So a measure that claims to index gray/white tissue contrast and does NOT
decline with age is not measuring tissue contrast. That makes this the
sharpest available test of what gBSC actually is, and its result is reportable
in either direction:

  an age effect      supports the tissue-contrast reading, and the prognostic
                     nulls then say something about AD specifically rather
                     than about the measure
  no age effect      is evidence for the measurement-positive claim (a), that
                     gBSC is dominated by acquisition variance. Combined with
                     the A1 result that it is highly reliable WITHIN a scanner
                     and collapses ACROSS field strengths, that is a coherent
                     and publishable characterisation of the measure.

Either way it is reported, not buried. A1 established the attenuation bound:
the same-scanner ICC is 0.843, so a true correlation is observed at roughly
sqrt(0.843) of its size, which is not enough to hide an effect of the
magnitude the literature reports.

Reference rows are included deliberately. Brain parenchymal fraction and SNR
come from the same pipeline and the same scans, so if gBSC shows no age effect
while BPF shows a large one, the covariate block is not the explanation.

Writes a3_age_results.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from a2_wmh import fit_one, standardized_beta

MEASURES = [
    ("bsc_dir_mean", "gBSC directional (cortex mean)", "measure"),
    ("bsc_mag_mean", "gBSC magnitude (cortex mean)", "measure"),
    ("bsc_cos_mean", "cosBSC (cortex mean)", "measure"),
    ("bscdir_adsig", "gBSC AD-signature composite", "measure"),
    ("cos_adsig", "cosBSC AD-signature composite", "measure"),
    ("seg_bpf", "REFERENCE brain parenchymal fraction", "reference"),
    ("seg_tiv_mm3", "REFERENCE intracranial volume", "reference"),
    ("qc_snr", "REFERENCE SNR", "reference"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(Path(__file__).parent / "subject_table.csv"))
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    parser.add_argument("--icc", type=float, default=0.843)
    args = parser.parse_args()

    frame = pd.read_csv(args.table)
    frame = frame[frame["age"].notna()]
    frame["field_strength"] = frame["field_strength"].fillna("unknown").astype(str)
    frame["site"] = frame["site"].astype(str)

    # ICV is a covariate for the measures but would be circular as a covariate
    # for itself, so it is dropped for that one row.
    covariates = ["female", "field_strength", "site"]

    print(f"A3 cohort: {len(frame)} subjects with age")
    print(f"  age: median {frame['age'].median():.1f}, "
          f"range {frame['age'].min():.1f} to {frame['age'].max():.1f}")
    print(f"  field strength: {dict(frame['field_strength'].value_counts())}")
    print()

    header = f"{'measure':38s} {'n':>5s} {'beta/decade':>12s} {'beta_std':>9s} {'p':>10s}"
    print(header)
    results = []
    for column, label, kind in MEASURES:
        if column not in frame.columns:
            continue
        usable = frame.dropna(subset=[column, "age", "female"])
        if len(usable) < 50:
            continue
        model = fit_one(usable, column, "age", covariates)
        beta = float(model.params["age"])
        se = float(model.bse["age"])
        beta_std = standardized_beta(model, usable, column, "age")
        pvalue = float(model.pvalues["age"])
        # Per decade is the unit this literature reports.
        results.append(
            {
                "measure": label,
                "column": column,
                "kind": kind,
                "n": int(model.nobs),
                "beta_per_year": round(beta, 8),
                "beta_per_decade": round(beta * 10, 6),
                "se_per_year": round(se, 8),
                "beta_std": round(beta_std, 4),
                "beta_std_disattenuated": round(beta_std / np.sqrt(args.icc), 4)
                if kind == "measure" else "",
                "p": pvalue,
                "r2": round(float(model.rsquared), 4),
            }
        )
        print(f"{label:38s} {int(model.nobs):5d} {beta * 10:12.5g} {beta_std:+9.4f} {pvalue:10.3g}")

    out_path = Path(args.out_dir) / "a3_age_results.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Any measure that clears p<0.05 gets the same scrutiny A2's null received.
    # A positive must not escape guards that a null was held to.
    positives = [r for r in results if r["kind"] == "measure" and r["p"] < 0.05]
    guard_rows = []
    if positives:
        print("\nGUARDS on every measure that cleared p<0.05")
        three_t = frame[frame["field_strength"].str.startswith("3")]
        for row in positives:
            column = row["column"]
            print(f"\n  {row['measure']}")
            checks = [
                ("3T only", three_t, covariates),
                ("+ SNR adjusted", frame, covariates + ["qc_snr"]),
                ("+ ICV adjusted", frame, covariates + ["seg_tiv_mm3"]),
                ("+ diagnosis adjusted", frame, covariates + ["diagnosis"]),
                ("+ SNR, ICV, diagnosis", frame, covariates + ["qc_snr", "seg_tiv_mm3", "diagnosis"]),
            ]
            for label, subset, covs in checks:
                needed = [column, "age"] + [c for c in covs if c not in ("site", "field_strength")]
                usable = subset.dropna(subset=needed)
                if len(usable) < 50:
                    print(f"    {label:24s} too few subjects ({len(usable)})")
                    continue
                model = fit_one(usable, column, "age", covs)
                beta_std = standardized_beta(model, usable, column, "age")
                pvalue = float(model.pvalues["age"])
                flag = "" if pvalue < 0.05 else "   <-- does not survive"
                print(f"    {label:24s} n={int(model.nobs):5d} beta_std={beta_std:+.4f} "
                      f"p={pvalue:.3g}{flag}")
                guard_rows.append({"measure": row["measure"], "column": column,
                                   "guard": label, "n": int(model.nobs),
                                   "beta_std": round(beta_std, 4), "p": pvalue})

            # Split-half stability, 200 splits, as required for any positive.
            rng = np.random.default_rng(42)
            usable = frame.dropna(subset=[column, "age", "female"])
            index = np.asarray(usable.index)
            train, held, held_p = [], [], []
            for _ in range(200):
                shuffled = rng.permutation(index)
                half = len(shuffled) // 2
                try:
                    model_a = fit_one(usable.loc[shuffled[:half]], column, "age", covariates)
                    model_b = fit_one(usable.loc[shuffled[half:]], column, "age", covariates)
                except Exception:
                    continue
                train.append(float(model_a.params["age"]))
                held.append(float(model_b.params["age"]))
                held_p.append(float(model_b.pvalues["age"]))
            if train:
                train_arr, held_arr = np.asarray(train), np.asarray(held)
                same_sign = float(np.mean(np.sign(held_arr) == np.sign(train_arr)))
                print(f"    {'split-half (200)':24s} held-out beta {held_arr.mean():.5g} "
                      f"[{np.percentile(held_arr, 2.5):.5g}, {np.percentile(held_arr, 97.5):.5g}]"
                      f"  same sign {same_sign:.2f}"
                      f"  held sig {np.mean(np.asarray(held_p) < 0.05):.2f}")
                guard_rows.append({"measure": row["measure"], "column": column,
                                   "guard": "split_half_200", "n": len(train),
                                   "beta_std": "", "p": "",
                                   "note": f"held_beta={held_arr.mean():.5g}, "
                                           f"same_sign={same_sign:.2f}, "
                                           f"held_sig={np.mean(np.asarray(held_p) < 0.05):.2f}"})
    if guard_rows:
        guard_path = Path(args.out_dir) / "a3_age_guards.csv"
        fields = ["measure", "column", "guard", "n", "beta_std", "p", "note"]
        with open(guard_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(guard_rows)
        print(f"\nWrote {guard_path}")

    primary = next((r for r in results if r["column"] == "bsc_dir_mean"), None)
    reference = next((r for r in results if r["column"] == "seg_bpf"), None)
    print("\nPre-specified reading (A3):")
    if primary and reference:
        print(f"  gBSC vs age:  beta_std {primary['beta_std']:+.4f}, p={primary['p']:.3g}")
        print(f"  BPF vs age:   beta_std {reference['beta_std']:+.4f}, p={reference['p']:.3g}")
        if primary["p"] >= 0.05 and reference["p"] < 0.05:
            print("  The anchor FAILS for gBSC while holding for a reference measure from the")
            print("  same pipeline and the same scans. Per the pre-spec this is evidence for")
            print("  the measurement-positive claim: gBSC does not behave like a tissue")
            print("  contrast measure, and the covariate block is not the explanation.")
        elif primary["p"] < 0.05:
            print("  The anchor HOLDS. gBSC recovers the age effect this literature reports,")
            print("  so the prognostic nulls constrain AD specifically rather than the measure.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
