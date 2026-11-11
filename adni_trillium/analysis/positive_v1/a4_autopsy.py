#!/usr/bin/env python3
"""A4 of PRESPEC_positive_v1.md: autopsy-confirmed neuropathology.

121 ADNI subjects have complete NPBRAAK, NPTHAL, NPADNC and NPNEUR, and 95 of
them have gBSC. This is the only outcome available in this project that is not
a clinical proxy, which makes it the hardest result to argue with and the one
most worth doing properly.

Primary test, fixed in advance: one ordinal regression of Braak stage on the
cortex-wide mean of the measure, adjusted for age at death, sex and field
strength, using each subject's last available scan.

Hard constraint carried over from the pre-spec, written before the result was
seen: at n = 95 this is powered only for a large effect. If the global test is
null there is NO regional follow-up and NO substitution of another
neuropathology variable. The secondary pathology outcomes below are reported
only when the primary clears, and that gate is enforced in code rather than
left to judgment.

Known limitation, stated up front: this autopsy sample is enriched for
advanced disease. 79 of 121 subjects are Braak 5 or 6 and 77 of 121 are ADNC
3, so the usable range of the outcome is compressed and the test is weaker
than n = 95 suggests.

Writes a4_autopsy_results.csv.
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
from statsmodels.miscmodels.ordinal_model import OrderedModel

from build_subject_table import MEASURE_COLUMNS, load_measure, load_demographics, parse_date

PRIMARY_OUTCOME = "NPBRAAK"
SECONDARY_OUTCOMES = ["NPTHAL", "NPADNC", "NPNEUR"]
MEASURES = ["bsc_dir_mean", "bsc_mag_mean", "bsc_cos_mean", "bscdir_adsig", "cos_adsig"]


def last_scan_per_subject(root: Path, measures: dict) -> dict[str, dict]:
    by_subject: dict[str, list] = defaultdict(list)
    with open(root / "manifest_realdates.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            date = parse_date(row["real_date"])
            if date is None:
                continue
            row["_date"] = date
            by_subject[row["subject"]].append(row)

    out = {}
    for subject, scans in by_subject.items():
        scans = [s for s in scans if s["image_id"] in measures["bsc_dir_mean"]]
        if not scans:
            continue
        chosen = max(scans, key=lambda s: s["_date"])
        record = {
            "image_id": chosen["image_id"],
            "scan_date": chosen["_date"],
            "field_strength": chosen.get("field_strength_ida", "").strip() or "unknown",
        }
        for name in MEASURE_COLUMNS:
            record[name] = measures[name].get(chosen["image_id"], np.nan)
        out[subject] = record
    return out


def fit_ordinal(frame: pd.DataFrame, outcome: str, measure: str):
    exog = frame[[measure, "age_at_death", "female"]].astype(float)
    dummies = pd.get_dummies(frame["field_strength"], prefix="fs", drop_first=True, dtype=float)
    if dummies.shape[1]:
        exog = pd.concat([exog, dummies], axis=1)
    # Standardize the measure so the coefficient reads per SD, comparable to
    # the effect sizes reported elsewhere in this folder.
    exog[measure] = (exog[measure] - exog[measure].mean()) / exog[measure].std(ddof=1)
    endog = frame[outcome].astype(int)
    model = OrderedModel(endog, exog, distr="logit").fit(method="bfgs", disp=False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/Users/ishu/research/adni_trillium/from_cluster")
    parser.add_argument("--adnimerge", default="/Users/ishu/research/ADNIMERGE2/data")
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    args = parser.parse_args()

    root = Path(args.data_root)
    measures = {
        name: load_measure(root, filename, column)
        for name, (filename, column) in MEASURE_COLUMNS.items()
    }
    scans = last_scan_per_subject(root, measures)
    demographics = load_demographics(Path(args.adnimerge))

    neuropath = pyreadr.read_r(str(Path(args.adnimerge) / "NEUROPATH.rda"))["NEUROPATH"]
    rows = []
    for _, record in neuropath.iterrows():
        ptid = record.get("PTID")
        if not isinstance(ptid, str) or ptid not in scans:
            continue
        age_at_death = pd.to_numeric(record.get("NPDAGE"), errors="coerce")
        if pd.isna(age_at_death):
            continue
        entry = {
            "subject": ptid,
            "age_at_death": float(age_at_death),
            "female": {"Female": 1, "Male": 0}.get(demographics.get(ptid, {}).get("sex"), np.nan),
            "field_strength": scans[ptid]["field_strength"],
            "scan_date": scans[ptid]["scan_date"].date().isoformat(),
        }
        for name in MEASURE_COLUMNS:
            entry[name] = scans[ptid][name]
        for outcome in [PRIMARY_OUTCOME] + SECONDARY_OUTCOMES:
            entry[outcome] = pd.to_numeric(record.get(outcome), errors="coerce")
        rows.append(entry)

    frame = pd.DataFrame(rows)
    print(f"A4 cohort: {len(frame)} subjects with autopsy and gBSC")
    if len(frame):
        print(f"  age at death: median {frame['age_at_death'].median():.1f}")
        print(f"  Braak: {dict(frame[PRIMARY_OUTCOME].value_counts().sort_index())}")
        top = (frame[PRIMARY_OUTCOME] >= 5).mean()
        print(f"  fraction Braak 5 or 6: {top:.2f}  (range compression, see docstring)")
    print()

    results = []
    primary_clears = False
    print("PRIMARY (pre-specified): Braak stage on cortex-wide gBSC, ordinal logit")
    for measure in MEASURES:
        usable = frame.dropna(subset=[measure, PRIMARY_OUTCOME, "age_at_death", "female"])
        if len(usable) < 40:
            print(f"  {measure:16s} n={len(usable)} too few")
            continue
        try:
            model = fit_ordinal(usable, PRIMARY_OUTCOME, measure)
        except Exception as exc:  # pragma: no cover
            print(f"  {measure:16s} failed to converge: {exc}")
            continue
        beta = float(model.params[measure])
        pvalue = float(model.pvalues[measure])
        se = float(model.bse[measure])
        tag = "PRIMARY" if measure == "bsc_dir_mean" else "secondary measure"
        results.append({"analysis": tag, "outcome": PRIMARY_OUTCOME, "measure": measure,
                        "n": int(len(usable)), "beta_per_sd": round(beta, 4),
                        "or_per_sd": round(float(np.exp(beta)), 4),
                        "ci_lo": round(float(np.exp(beta - 1.96 * se)), 4),
                        "ci_hi": round(float(np.exp(beta + 1.96 * se)), 4),
                        "p": pvalue})
        print(f"  {measure:16s} n={len(usable):3d} OR per SD={np.exp(beta):.3f} "
              f"[{np.exp(beta - 1.96 * se):.3f}, {np.exp(beta + 1.96 * se):.3f}] p={pvalue:.3g}")
        if measure == "bsc_dir_mean" and pvalue < 0.05:
            primary_clears = True

    if primary_clears:
        print("\nSECONDARY pathology outcomes (gate opened by a significant primary)")
        for outcome in SECONDARY_OUTCOMES:
            usable = frame.dropna(subset=["bsc_dir_mean", outcome, "age_at_death", "female"])
            if len(usable) < 40:
                continue
            model = fit_ordinal(usable, outcome, "bsc_dir_mean")
            beta = float(model.params["bsc_dir_mean"])
            pvalue = float(model.pvalues["bsc_dir_mean"])
            results.append({"analysis": "secondary_outcome", "outcome": outcome,
                            "measure": "bsc_dir_mean", "n": int(len(usable)),
                            "beta_per_sd": round(beta, 4),
                            "or_per_sd": round(float(np.exp(beta)), 4), "p": pvalue})
            print(f"  {outcome:10s} OR per SD={np.exp(beta):.3f} p={pvalue:.3g}")
    else:
        print("\nSECONDARY outcomes NOT run and no regional follow-up performed.")
        print("The pre-spec fixed this before the result was seen: at n=95 a null global")
        print("test ends the analysis rather than licensing a search.")
        results.append({"analysis": "secondary_not_run", "outcome": "", "measure": "",
                        "n": "", "p": "",
                        "note": "pre-spec forbids follow-up after a null primary"})

    out_path = Path(args.out_dir) / "a4_autopsy_results.csv"
    fields = ["analysis", "outcome", "measure", "n", "beta_per_sd", "or_per_sd",
              "ci_lo", "ci_hi", "p", "note"]
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
