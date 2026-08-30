"""Attach ADNI amyloid and tau biomarkers to the spec V3 cohort.

The editor who declined BRAINCOM-2026-1065 objected that the sample carries no
biomarkers, so conversion is a clinical label rather than a biological one. This
script answers that without shrinking the cohort: every subject keeps their row
and gains an amyloid status of positive, negative or unknown.

Amyloid status is taken from the closest measurement acquired at or before the
landmark visit (plus a 180 day grace window, since ADNI schedules PET and lumbar
puncture around, not exactly on, the clinical visit). Nothing acquired after the
landmark is used, because a scan ordered after someone converted would leak the
outcome into the predictor.

Amyloid PET is preferred where it exists, since it measures the plaques directly.
CSF Abeta42 from the Roche Elecsys assay is the fallback, dichotomized at the
ADNI reference cutoff of 977 pg/mL. Tau PET is attached where available but is
too sparse to stratify on and is carried for construct validity only.

Writes a new cohort CSV. It never modifies the input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

# Roche Elecsys CSF Abeta42 cutoff used throughout ADNI (Hansson et al. 2018).
CSF_ABETA42_CUTOFF = 977.0
# ADNI schedules PET and lumbar puncture near, not on, the clinical visit.
GRACE_DAYS = 180


def load_rda(data_dir: Path, name: str) -> pd.DataFrame:
    obj = pyreadr.read_r(str(data_dir / f"{name}.rda"))
    return obj[list(obj)[0]]


def numeric(series: pd.Series) -> pd.Series:
    """ADNI censors assay values as '>1700' or '<200'; keep the number."""
    return pd.to_numeric(
        series.astype(str).str.replace(">", "", regex=False)
                          .str.replace("<", "", regex=False)
                          .str.strip(),
        errors="coerce")


def nearest_before(cohort: pd.DataFrame, table: pd.DataFrame,
                   date_col: str, keep: list[str], suffix: str) -> pd.DataFrame:
    """Closest row at or before landmark + grace, one per subject."""
    t = table.copy()
    t["_dt"] = pd.to_datetime(t[date_col], errors="coerce")
    t = t.dropna(subset=["_dt"])
    m = cohort[["subject", "_lm"]].merge(
        t[["PTID", "_dt"] + keep].rename(columns={"PTID": "subject"}),
        on="subject", how="inner")
    m = m[m["_dt"] <= m["_lm"] + pd.Timedelta(days=GRACE_DAYS)]
    m = m.sort_values("_dt").groupby("subject", as_index=False).last()
    m[f"{suffix}_days_from_landmark"] = (m["_dt"] - m["_lm"]).dt.days
    m = m.drop(columns=["_lm", "_dt"])
    return m.rename(columns={c: f"{suffix}_{c}" for c in keep})


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--adnimerge", default="/Users/ishu/research/ADNIMERGE2/data")
    ap.add_argument("--out", default=str(root / "results/biomarker_v1/cohort_with_biomarkers.csv"))
    args = ap.parse_args()

    data_dir = Path(args.adnimerge)
    df = pd.read_csv(args.cohort)
    n0 = len(df)
    df["_lm"] = pd.to_datetime(df["landmark_date"], errors="coerce")

    # ---- amyloid PET -------------------------------------------------------
    amy = load_rda(data_dir, "UCBERKELEY_AMY_6MM")
    pet = nearest_before(df, amy, "SCANDATE",
                         ["AMYLOID_STATUS", "CENTILOIDS", "SUMMARY_SUVR", "TRACER"],
                         "pet")
    # ---- CSF ---------------------------------------------------------------
    csf = load_rda(data_dir, "UPENNBIOMK_ROCHE_ELECSYS")
    csf_pre = nearest_before(df, csf, "EXAMDATE",
                             ["ABETA42", "ABETA40", "PTAU", "TAU"], "csf")
    for c in ["csf_ABETA42", "csf_ABETA40", "csf_PTAU", "csf_TAU"]:
        csf_pre[c] = numeric(csf_pre[c])
    csf_pre["csf_ptau_abeta42_ratio"] = csf_pre["csf_PTAU"] / csf_pre["csf_ABETA42"]
    csf_pre["csf_amyloid_pos"] = (csf_pre["csf_ABETA42"] < CSF_ABETA42_CUTOFF).astype(float)
    csf_pre.loc[csf_pre["csf_ABETA42"].isna(), "csf_amyloid_pos"] = np.nan

    # ---- tau PET (construct validity only, too sparse to stratify on) ------
    taupet = load_rda(data_dir, "UCBERKELEY_TAU_6MM")
    tau = nearest_before(df, taupet, "SCANDATE",
                         ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"], "taupet")

    out = df.merge(pet, on="subject", how="left") \
            .merge(csf_pre, on="subject", how="left") \
            .merge(tau, on="subject", how="left")
    assert len(out) == n0, "merge changed the number of subjects"

    out["pet_amyloid_pos"] = pd.to_numeric(out["pet_AMYLOID_STATUS"], errors="coerce")

    # PET first because it measures plaques directly; CSF fills the gaps.
    out["amyloid_pos"] = out["pet_amyloid_pos"]
    filled = out["amyloid_pos"].isna() & out["csf_amyloid_pos"].notna()
    out.loc[filled, "amyloid_pos"] = out.loc[filled, "csf_amyloid_pos"]

    src = np.where(out["pet_amyloid_pos"].notna() & out["csf_amyloid_pos"].notna(), "pet+csf",
          np.where(out["pet_amyloid_pos"].notna(), "pet",
          np.where(out["csf_amyloid_pos"].notna(), "csf", "none")))
    out["amyloid_source"] = src

    # Missing-indicator coding keeps all subjects in the covariate model.
    out["amyloid_unknown"] = out["amyloid_pos"].isna().astype(float)
    out["amyloid_pos_flag"] = out["amyloid_pos"].fillna(0.0)
    out["amyloid_status_label"] = np.where(out["amyloid_pos"] == 1, "A+",
                                  np.where(out["amyloid_pos"] == 0, "A-", "unknown"))

    both = out[out["pet_amyloid_pos"].notna() & out["csf_amyloid_pos"].notna()]
    agree = float((both["pet_amyloid_pos"] == both["csf_amyloid_pos"]).mean()) if len(both) else float("nan")

    # Coverage if post-landmark assays were also allowed. Descriptive only:
    # amyloid status is trait-like, but a scan ordered after conversion would
    # leak the outcome, so the analysis never uses these.
    ever = set(amy.loc[amy.PTID.isin(df.subject), "PTID"]) | \
           set(csf.loc[csf.PTID.isin(df.subject), "PTID"])

    summary = {
        "n_subjects": int(n0),
        "n_events": int(df["event"].sum()),
        "coverage_pet": int(out["pet_amyloid_pos"].notna().sum()),
        "coverage_csf": int(out["csf_amyloid_pos"].notna().sum()),
        "coverage_any_prelandmark": int(out["amyloid_pos"].notna().sum()),
        "coverage_any_ever_descriptive": int(len(ever & set(df.subject))),
        "coverage_taupet": int(out["taupet_META_TEMPORAL_SUVR"].notna().sum()),
        "pet_csf_agreement": agree,
        "n_pet_csf_both": int(len(both)),
        "csf_cutoff_pg_ml": CSF_ABETA42_CUTOFF,
        "grace_days": GRACE_DAYS,
    }
    for lab in ["A+", "A-", "unknown"]:
        s = out[out["amyloid_status_label"] == lab]
        summary[f"n_{lab}"] = int(len(s))
        summary[f"events_{lab}"] = int(s["event"].sum())

    out = out.drop(columns=["_lm"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    with open(out_path.parent / "biomarker_coverage.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path} ({len(out)} subjects, {out.shape[1]} columns)")


if __name__ == "__main__":
    main()
