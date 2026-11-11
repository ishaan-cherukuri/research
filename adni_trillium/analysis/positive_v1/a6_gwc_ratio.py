#!/usr/bin/env python3
"""A6: the gray/white tissue intensity ratio, the measure the literature uses.

This arm was pre-specified in PRESPEC_positive_v1.md as blocked rather than
abandoned, because the file lives on Trillium and the cluster refused the SSH
key when A1 to A5 were run. It is now unblocked and run with no change to what
was declared.

Why it is the highest-value remaining test. Every positive finding in this
measure class belongs to the gray/white intensity RATIO, not to a sigmoid
growth rate and not to an intensity gradient: age-related decline across the
cortex independently of thinning (Salat PMC2750073; Sci Rep 2025
s41598-025-34355-y), reduction in schizophrenia (Jorgensen PLoS ONE 2012),
sensitivity in Parkinson's (PMC5881246), and change preceding amyloid
positivity in non-demented older adults (Pieperhoff, Alzheimer's and Dementia
2026, doi:10.1002/alz.71609). A3 then found that our scale-invariant cosine
variant recovers the age decline while the published gradient does not, which
points directly here: the ratio is scale-free by construction.

Two tests, both already fixed in advance:
  A3-style age anchor, the single most reproducible finding in this
  literature, as the test of whether the measure carries tissue contrast.
  A2-style WMH association, the mechanism those papers propose for
  age-related contrast loss.

Both get the guards A3's positive got, not the lighter treatment a null got:
3T-only, SNR, ICV and diagnosis adjustment separately and jointly, and 200
split-halves.

The column prefix is detected rather than hardcoded. Two pipelines in this
project write a ratio under different names (`bscsigratio_*` from
fs_sigmoid_features.py, `bscratio_*` from the CIVET
extract_sigmoid_features.py) and `bscratio_*` ALSO means the cosine measure in
the per_scan_cos files. The detector therefore checks the value range: a
WM/GM intensity ratio in a T1w image exceeds 1, while the cosine measure is
bounded in [0,1]. If the candidate columns look bounded, the script refuses to
run rather than silently analyse the wrong quantity.

Writes a6_ratio_results.csv and a6_ratio_guards.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from a2_wmh import benjamini_hochberg, fit_one, standardized_beta

CANDIDATE_PREFIXES = ["bscsigratio", "bscratio"]

# All three families in the file, because the pulled file turned out to carry
# the actual published measure as well as the ratio. Testing only the ratio
# would have left the headline question unanswered: the sigmoid growth rate is
# the quantity Olafson 2021 defines and every null in this project to date was
# obtained on a gradient proxy instead, so this is the first time the published
# measure itself meets the age anchor.
MEASURE_FAMILIES = {
    "bscsig": "published sigmoid BSC, log growth rate, reference bounds",
    "bscsigfree": "sigmoid BSC, a unbounded below, sensitivity variant",
    "bscsigratio": "WM/GM tissue intensity ratio, the GWC analogue",
}
# Cortical regions of the AD signature, matched as substrings against the
# FreeSurfer-style parcel names in this file. CIVET has no entorhinal parcel,
# so a composite built here is not interchangeable with the volumetric one.
AD_SIGNATURE = [
    "entorhinal", "fusiform", "inferiortemporal", "middletemporal",
    "parahippocampal", "posteriorcingulate", "precuneus",
]


def detect_ratio_columns(frame: pd.DataFrame) -> tuple[str, list[str]]:
    """Find the tissue intensity ratio columns and verify they are not cosines."""
    for prefix in CANDIDATE_PREFIXES:
        columns = [c for c in frame.columns if c.startswith(prefix + "_")]
        if not columns:
            continue
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        finite = values.values[np.isfinite(values.values)]
        if finite.size == 0:
            continue
        maximum = float(np.nanmax(finite))
        print(f"  candidate prefix {prefix!r}: {len(columns)} columns, max value {maximum:.4f}")
        if maximum <= 1.0001:
            raise SystemExit(
                f"REFUSING TO RUN. Columns with prefix {prefix!r} have a maximum of "
                f"{maximum:.4f}, which is bounded in [0,1]. That is the signature of the "
                "cosine measure, not a WM/GM tissue intensity ratio, which exceeds 1 in a "
                "T1w image. See the naming trap in PRESPEC_positive_v1.md. Analysing these "
                "as a ratio would repeat exactly the mistake this project already made once."
            )
        return prefix, columns
    raise SystemExit(
        f"No ratio columns found. Tried prefixes {CANDIDATE_PREFIXES}. "
        f"Available prefixes: {sorted({c.split('_')[0] for c in frame.columns})}"
    )


def build_measures(ratio: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Cortex-wide mean and AD-signature mean for every family in the file.

    Coverage is checked per family, not assumed. The shipped per_scan_sigmoid.csv
    was once unusable because five of seven AD-signature parcels were empty for
    every scan while the composite column still existed and looked plausible,
    so an empty-parcel count is printed for each family.
    """
    out = pd.DataFrame({"image_id": ratio["image_id"]})
    parcels_by_family: dict[str, list[str]] = {}

    for family, description in MEASURE_FAMILIES.items():
        columns = [c for c in ratio.columns if c.startswith(family + "_")]
        if not columns:
            continue
        parcels = [c for c in columns if "roi" in c]
        parcels_by_family[family] = parcels
        print(f"  {family}: {len(columns)} columns, {len(parcels)} parcels  ({description})")

        summary = f"{family}_mean"
        if summary in ratio.columns:
            out[f"{family}_cortex"] = pd.to_numeric(ratio[summary], errors="coerce")
        elif parcels:
            out[f"{family}_cortex"] = ratio[parcels].apply(
                pd.to_numeric, errors="coerce").mean(axis=1)

        composite = f"{family}_adsig"
        if composite in ratio.columns:
            out[f"{family}_adsig"] = pd.to_numeric(ratio[composite], errors="coerce")

        signature = [c for c in parcels if any(r in c.lower() for r in AD_SIGNATURE)]
        if signature:
            values = ratio[signature].apply(pd.to_numeric, errors="coerce")
            coverage = values.notna().mean()
            empty = [c for c in signature if coverage[c] < 0.5]
            print(f"    AD-signature parcels: {len(signature)} matched, "
                  f"{len(empty)} under 50% coverage")
            if empty:
                print(f"    WARNING low coverage: {empty[:6]}")
            # Recompute from the parcels so the composite is auditable rather
            # than inherited from a column that may predate the NaN fix.
            out[f"{family}_adsig_recomputed"] = values.mean(axis=1)

    return out, parcels_by_family


def guard_table(frame: pd.DataFrame, column: str, predictor: str, base_covs: list[str],
                label: str) -> list[dict]:
    """The same guards A3 applied to its positive."""
    rows = []
    three_t = frame[frame["field_strength"].astype(str).str.startswith("3")]
    checks = [
        ("3T only", three_t, base_covs),
        ("+ SNR adjusted", frame, base_covs + ["qc_snr"]),
        ("+ ICV adjusted", frame, base_covs + ["seg_tiv_mm3"]),
        ("+ diagnosis adjusted", frame, base_covs + ["diagnosis"]),
        ("+ SNR, ICV, diagnosis", frame, base_covs + ["qc_snr", "seg_tiv_mm3", "diagnosis"]),
    ]
    for name, subset, covs in checks:
        needed = [column, predictor] + [c for c in covs if c not in ("site", "field_strength")]
        usable = subset.dropna(subset=needed)
        if len(usable) < 50:
            print(f"    {name:24s} too few subjects ({len(usable)})")
            continue
        model = fit_one(usable, column, predictor, covs)
        beta_std = standardized_beta(model, usable, column, predictor)
        pvalue = float(model.pvalues[predictor])
        flag = "" if pvalue < 0.05 else "   <-- does not survive"
        print(f"    {name:24s} n={int(model.nobs):5d} beta_std={beta_std:+.4f} p={pvalue:.3g}{flag}")
        rows.append({"analysis": label, "guard": name, "measure": column,
                     "n": int(model.nobs), "beta_std": round(beta_std, 4), "p": pvalue})

    rng = np.random.default_rng(42)
    usable = frame.dropna(subset=[column, predictor] + [c for c in base_covs if c != "site"])
    index = np.asarray(usable.index)
    train, held, held_p = [], [], []
    for _ in range(200):
        shuffled = rng.permutation(index)
        half = len(shuffled) // 2
        try:
            model_a = fit_one(usable.loc[shuffled[:half]], column, predictor, base_covs)
            model_b = fit_one(usable.loc[shuffled[half:]], column, predictor, base_covs)
        except Exception:
            continue
        train.append(float(model_a.params[predictor]))
        held.append(float(model_b.params[predictor]))
        held_p.append(float(model_b.pvalues[predictor]))
    if train:
        train_arr, held_arr, p_arr = map(np.asarray, (train, held, held_p))
        same_sign = float(np.mean(np.sign(held_arr) == np.sign(train_arr)))
        frac_sig = float(np.mean(p_arr < 0.05))
        print(f"    {'split-half (200)':24s} held-out {held_arr.mean():.5g} "
              f"[{np.percentile(held_arr, 2.5):.5g}, {np.percentile(held_arr, 97.5):.5g}]"
              f"  same sign {same_sign:.2f}  held sig {frac_sig:.2f}")
        rows.append({"analysis": label, "guard": "split_half_200", "measure": column,
                     "n": len(train), "beta_std": "", "p": "",
                     "note": f"held_beta={held_arr.mean():.5g}, same_sign={same_sign:.2f}, "
                             f"held_sig={frac_sig:.2f}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path("/Users/ishu/research/adni_trillium")
    parser.add_argument("--ratio_file", default=str(base / "from_cluster/per_scan_sigmoid_nn.csv"))
    parser.add_argument("--table", default=str(Path(__file__).parent / "subject_table.csv"))
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    parser.add_argument("--max_wmh_gap_days", type=int, default=365)
    args = parser.parse_args()

    print("Detecting the ratio columns")
    ratio = pd.read_csv(args.ratio_file, low_memory=False)
    print(f"  file: {len(ratio)} scans, {ratio.shape[1]} columns, "
          f"{ratio['subject'].nunique()} subjects")
    # Verifies the ratio family is a real intensity ratio and not a cosine.
    detect_ratio_columns(ratio)
    measures, parcels_by_family = build_measures(ratio)
    parcels = parcels_by_family.get("bscsigratio", [])

    frame = pd.read_csv(args.table)
    merged = frame.merge(measures, on="image_id", how="inner")
    merged["field_strength"] = merged["field_strength"].fillna("unknown").astype(str)
    merged["site"] = merged["site"].astype(str)
    print(f"\nMerged onto the subject table: {len(merged)} subjects "
          f"(the sigmoid pipeline covers the spec cohort, not all 1601)")

    measure_columns = [c for c in measures.columns if c != "image_id" and c in merged.columns]
    results, guards = [], []

    print("\n=== A6a: AGE ANCHOR (pre-specified in A3) ===")
    covariates = ["female", "field_strength", "site"]
    for column in measure_columns:
        usable = merged.dropna(subset=[column, "age", "female"])
        if len(usable) < 50:
            print(f"  {column}: n={len(usable)} too few")
            continue
        model = fit_one(usable, column, "age", covariates)
        beta_std = standardized_beta(model, usable, column, "age")
        pvalue = float(model.pvalues["age"])
        print(f"  {column:14s} n={int(model.nobs):4d} beta/decade={model.params['age'] * 10:+.5g} "
              f"beta_std={beta_std:+.4f} p={pvalue:.3g}")
        results.append({"analysis": "age", "measure": column, "n": int(model.nobs),
                        "beta_std": round(beta_std, 4), "p": pvalue,
                        "verdict": "positive" if pvalue < 0.05 else "null"})
        if pvalue < 0.05:
            print(f"  guards for {column}")
            guards += guard_table(merged, column, "age", covariates, "age")

    print("\n=== A6b: WMH BURDEN (pre-specified in A2) ===")
    wmh = merged[merged["total_wmh"].notna() & (merged["wmh_gap_days"] <= args.max_wmh_gap_days)].copy()
    wmh["log_wmh"] = np.log1p(wmh["total_wmh"].astype(float))
    print(f"  {len(wmh)} subjects with WMH within {args.max_wmh_gap_days}d")
    wmh_covs = ["age", "female", "seg_tiv_mm3", "field_strength", "site"]
    for column in measure_columns:
        usable = wmh.dropna(subset=[column, "log_wmh", "age", "female", "seg_tiv_mm3"])
        if len(usable) < 50:
            print(f"  {column}: n={len(usable)} too few")
            continue
        model = fit_one(usable, column, "log_wmh", wmh_covs)
        beta_std = standardized_beta(model, usable, column, "log_wmh")
        pvalue = float(model.pvalues["log_wmh"])
        print(f"  {column:14s} n={int(model.nobs):4d} beta_std={beta_std:+.4f} p={pvalue:.3g}")
        results.append({"analysis": "wmh", "measure": column, "n": int(model.nobs),
                        "beta_std": round(beta_std, 4), "p": pvalue,
                        "verdict": "positive" if pvalue < 0.05 else "null"})
        if pvalue < 0.05:
            print(f"  guards for {column}")
            guards += guard_table(wmh, column, "log_wmh", wmh_covs, "wmh")

    # Regional screen on age, gated the same way A2's was.
    age_positive = any(r["analysis"] == "age" and r["verdict"] == "positive" for r in results)
    regional = []
    if age_positive and parcels:
        print(f"\n=== A6c: REGIONAL age screen, {len(parcels)} parcels, BH step-up ===")
        wide = merged.merge(
            ratio[["image_id"] + parcels].apply(pd.to_numeric, errors="ignore"),
            on="image_id", how="inner")
        rows = []
        for parcel in parcels:
            usable = wide.dropna(subset=[parcel, "age", "female"])
            if len(usable) < 100:
                continue
            model = fit_one(usable, parcel, "age", covariates)
            rows.append({"parcel": parcel, "n": int(model.nobs),
                         "beta_std": standardized_beta(model, usable, parcel, "age"),
                         "p": float(model.pvalues["age"])})
        if rows:
            qvalues = benjamini_hochberg(np.array([r["p"] for r in rows]))
            for row, q in zip(rows, qvalues):
                row["q"] = q
            rows.sort(key=lambda r: r["p"])
            regional = rows
            print(f"  {len(rows)} parcels tested, "
                  f"{sum(1 for r in rows if r['q'] < 0.05)} survive BH q<0.05")
            for row in rows[:10]:
                print(f"    {row['parcel']:44s} beta_std={row['beta_std']:+.4f} "
                      f"p={row['p']:.3g} q={row['q']:.3g}")
    else:
        print("\n=== A6c: regional screen not run (age anchor was not positive) ===")

    out_dir = Path(args.out_dir)
    with open(out_dir / "a6_ratio_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["analysis", "measure", "n", "beta_std", "p", "verdict"])
        writer.writeheader()
        writer.writerows(results)
    if guards:
        with open(out_dir / "a6_ratio_guards.csv", "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["analysis", "guard", "measure", "n", "beta_std", "p", "note"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(guards)
    if regional:
        with open(out_dir / "a6_ratio_regional_age.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["parcel", "n", "beta_std", "p", "q"])
            writer.writeheader()
            writer.writerows(regional)
    print(f"\nWrote {out_dir / 'a6_ratio_results.csv'}")


if __name__ == "__main__":
    main()
