"""Landmark cohort for the ALZ-26-0928 revision (Trillium/657-subject analysis).

The submitted analysis started the clock at the MCI baseline scan while feeding
the model scans acquired up to several years later, some of them after the
subject had already been diagnosed with dementia. Reviewer 1 is right that this
measures detection rather than prediction. This script rebuilds the cohort so
that every feature comes from scans acquired while the subject was still MCI,
and follow-up starts at the last such scan.

Two landmark definitions are supported:

  fixed      the landmark sits a set number of months after the MCI baseline
             scan (default 24). Every scan in that window feeds the features,
             the subject must still be MCI at the landmark, and follow-up runs
             from the landmark date. This is the primary analysis.
  subject    the landmark is the subject's own last pre-conversion scan. Uses
             more scans per subject but gives shorter and more variable
             follow-up. Reported as a sensitivity analysis.

Conversion and censoring dates come from DXSUM clinical visits, not from the
diagnosis attached to a scan, so subjects who never convert keep the follow-up
they accrued after their last MRI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

SCAN_META = {"subject", "visit_code", "acq_date", "image_id", "diagnosis"}
DX_LABELS = {"CN": 1.0, "MCI": 2.0, "Dementia": 3.0, "1": 1.0, "2": 2.0, "3": 3.0}
MCI, AD = 2.0, 3.0

# Per-scan T1 measures that get both a cross-sectional and a longitudinal summary.
T1_TRAJECTORY = ["seg_brain_mm3", "seg_bpf", "seg_gm_total_mm3", "seg_wm_total_mm3",
                 "seg_csf_mm3", "qc_brain_mean", "qc_snr", "qc_brain_bg_ratio"]
T1_BASELINE_ONLY = ["seg_tiv_mm3", "qc_brain_mask_vol_mm3", "qc_brain_std",
                    "meta_field_strength_t"]


def load_scans(bsc_csv: str, t1_csv: str) -> tuple[pd.DataFrame, list[str]]:
    bsc = pd.read_csv(bsc_csv)
    t1 = pd.read_csv(t1_csv)
    t1_cols = [c for c in t1.columns if c not in SCAN_META]
    df = bsc.merge(t1[["image_id"] + t1_cols], on="image_id", how="left")
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).sort_values(["subject", "acq_date"])
    bsc_cols = [c for c in bsc.columns if c not in SCAN_META]
    return df.reset_index(drop=True), bsc_cols


def load_dx(path: str) -> dict[str, pd.DataFrame]:
    dx = pd.read_csv(path)
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["DIAGNOSIS"] = dx["DIAGNOSIS"].astype(str).str.strip().map(DX_LABELS)
    dx = dx.dropna(subset=["EXAMDATE"]).sort_values(["PTID", "EXAMDATE"])
    return {p: g for p, g in dx.groupby("PTID")}


def load_mmse(path: str) -> dict[str, pd.DataFrame]:
    m = pyreadr.read_r(path)["MMSE"][["PTID", "VISDATE", "MMSCORE"]].copy()
    m["VISDATE"] = pd.to_datetime(m["VISDATE"], errors="coerce")
    m["MMSCORE"] = pd.to_numeric(m["MMSCORE"], errors="coerce")
    m = m.dropna(subset=["VISDATE", "MMSCORE"])
    m = m[(m["MMSCORE"] >= 0) & (m["MMSCORE"] <= 30)]
    return {p: g.sort_values("VISDATE") for p, g in m.groupby("PTID")}


def load_demog(path: str) -> dict[str, dict]:
    d = pd.read_csv(path, low_memory=False)[["PTID", "PTGENDER", "PTDOB",
                                             "PTDOBYY", "PTEDUCAT"]].copy()
    d["PTEDUCAT"] = pd.to_numeric(d["PTEDUCAT"], errors="coerce")
    d.loc[d["PTEDUCAT"] < 0, "PTEDUCAT"] = np.nan
    dob = pd.to_datetime(d["PTDOB"], errors="coerce", format="mixed")
    yy = pd.to_numeric(d["PTDOBYY"], errors="coerce")
    yy_dt = pd.to_datetime(yy.where(yy > 1850).astype("Int64").astype(str) + "-07-01",
                           errors="coerce")
    d["dob"] = dob.fillna(yy_dt)
    sex = pd.to_numeric(d["PTGENDER"], errors="coerce")
    d["female"] = np.where(sex == 2, 1.0, np.where(sex == 1, 0.0, np.nan))
    d = d.dropna(subset=["dob"])
    agg = d.groupby("PTID").agg(dob=("dob", "first"), female=("female", "max"),
                                educ=("PTEDUCAT", "max"))
    return agg.to_dict("index")


def trajectory(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(times) & np.isfinite(values)
    t, v = times[ok], values[ok]
    if len(t) == 0:
        return {k: np.nan for k in ("baseline", "last", "delta", "pctchg",
                                    "slope_yr", "mean", "std", "r2")}
    out = {"baseline": float(v[0]), "last": float(v[-1]),
           "delta": float(v[-1] - v[0]),
           "pctchg": float((v[-1] - v[0]) / v[0] * 100.0) if v[0] != 0 else np.nan,
           "mean": float(v.mean()), "std": float(v.std(ddof=0))}
    if len(t) < 2 or np.ptp(t) == 0:
        out["slope_yr"] = np.nan
        out["r2"] = np.nan
        return out
    b1, b0 = np.polyfit(t, v, 1)
    pred = b0 + b1 * t
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    out["slope_yr"] = float(b1)
    out["r2"] = float(1.0 - np.sum((v - pred) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return out


def build(args) -> None:
    scans, bsc_cols = load_scans(args.bsc, args.t1_scans)
    dx_by_subj = load_dx(args.dxsum)
    mmse_by_subj = load_mmse(args.mmse)
    demog = load_demog(args.demog)

    counts = {k: 0 for k in ("subjects_with_scans", "baseline_not_mci",
                             "no_clinical_record", "converted_at_or_before_landmark",
                             "too_few_scans_in_window", "no_followup_after_landmark")}
    counts["subjects_with_scans"] = int(scans["subject"].nunique())

    rows = []
    for subj, g in scans.groupby("subject"):
        g = g.sort_values("acq_date")
        if float(g["diagnosis"].iloc[0]) != MCI:
            counts["baseline_not_mci"] += 1
            continue
        sdx = dx_by_subj.get(subj)
        if sdx is None or len(sdx) == 0:
            counts["no_clinical_record"] += 1
            continue

        first_scan = g["acq_date"].iloc[0]
        ad = sdx[(sdx["DIAGNOSIS"] == AD) & (sdx["EXAMDATE"] > first_scan)]
        converted = len(ad) > 0
        conv_date = ad["EXAMDATE"].iloc[0] if converted else pd.NaT
        last_visit = max(sdx["EXAMDATE"].max(), g["acq_date"].iloc[-1])

        if args.landmark == "legacy":
            # The submitted design: features from every scan, including any
            # acquired after the dementia diagnosis, and the clock started at
            # the MCI baseline scan. Reproduced here only for comparison.
            window = g
            t0 = first_scan
        elif args.landmark == "fixed":
            lm = first_scan + pd.Timedelta(days=round(args.landmark_months * 30.4375)
                                           + args.grace_days)
            if converted and conv_date <= lm:
                counts["converted_at_or_before_landmark"] += 1
                continue
            window = g[g["acq_date"] <= lm]
            t0 = lm
        else:
            window = g[g["acq_date"] < conv_date] if converted else g
            t0 = window["acq_date"].iloc[-1] if len(window) else pd.NaT

        if len(window) < args.min_scans:
            counts["too_few_scans_in_window"] += 1
            continue

        if args.landmark == "legacy":
            end = (conv_date if converted
                   else g["acq_date"].iloc[-1])
        else:
            end = conv_date if converted else last_visit
        time_years = (end - t0).days / 365.25
        if time_years <= 0:
            counts["no_followup_after_landmark"] += 1
            continue

        times = ((window["acq_date"] - window["acq_date"].iloc[0]).dt.days / 365.25).to_numpy()
        row = {"subject": subj, "event": int(converted),
               "time_years": time_years,
               "landmark_date": pd.Timestamp(t0).date().isoformat(),
               "n_scans_window": len(window),
               "window_span_years": round(float(times[-1]), 3),
               "n_scans_total": len(g)}

        for f in bsc_cols:
            tr = trajectory(times, window[f].to_numpy(dtype=float))
            row[f"{f}_baseline"] = tr["baseline"]
            row[f"{f}_final"] = tr["last"]
            row[f"{f}_slope"] = tr["slope_yr"]
            row[f"{f}_r2"] = tr["r2"]

        for f in T1_TRAJECTORY:
            if f not in window.columns:
                continue
            tr = trajectory(times, window[f].to_numpy(dtype=float))
            row[f"{f}_bl"] = tr["baseline"]
            for k in ("last", "delta", "pctchg", "slope_yr", "mean", "std"):
                row[f"{f}_{k}"] = tr[k]
        for f in T1_BASELINE_ONLY:
            if f in window.columns:
                row[f"{f}_bl"] = float(window[f].iloc[0])
        if "meta_field_strength_t" in window.columns:
            fs = pd.to_numeric(window["meta_field_strength_t"], errors="coerce").dropna()
            fs = fs.where(fs < 100, fs / 10000.0)
            row["field_strength_changed"] = int(fs.round(1).nunique() > 1)

        dem = demog.get(subj, {})
        dob = dem.get("dob")
        row["age_at_landmark"] = ((t0 - dob).days / 365.25) if pd.notna(dob) else np.nan
        row["female"] = dem.get("female", np.nan)
        row["educ_years"] = dem.get("educ", np.nan)

        sm = mmse_by_subj.get(subj)
        row["mmse_at_landmark"] = np.nan
        if sm is not None and len(sm):
            gap = (sm["VISDATE"] - t0).abs()
            j = gap.idxmin()
            if gap.loc[j] <= pd.Timedelta(days=args.mmse_window_days):
                row["mmse_at_landmark"] = float(sm.loc[j, "MMSCORE"])
        rows.append(row)

    out = pd.DataFrame(rows)
    counts.update({
        "subjects_retained": len(out),
        "converters": int(out["event"].sum()),
        "stable": int((out["event"] == 0).sum()),
        "median_followup_years": round(float(out["time_years"].median()), 3),
        "mean_followup_years": round(float(out["time_years"].mean()), 3),
        "mean_scans_in_window": round(float(out["n_scans_window"].mean()), 2),
        "mmse_available": int(out["mmse_at_landmark"].notna().sum()),
        "age_available": int(out["age_at_landmark"].notna().sum()),
        "educ_available": int(out["educ_years"].notna().sum()),
        "landmark": args.landmark,
        "landmark_months": args.landmark_months if args.landmark == "fixed" else None,
    })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "landmark_cohort.csv", index=False)
    with open(out_dir / "landmark_cohort.json", "w") as f:
        json.dump(counts, f, indent=2)
    for k, v in counts.items():
        print(f"  {k}: {v}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    base = "/Users/ishu/research/adni_trillium/from_cluster"
    p.add_argument("--bsc", default=f"{base}/bsc_simple_features.csv")
    p.add_argument("--t1_scans", default=f"{base}/t1_scan_features.csv")
    p.add_argument("--dxsum", default=f"{base}/dxsum.csv")
    p.add_argument("--mmse", default="/Users/ishu/research/ADNIMERGE2/data/MMSE.rda")
    p.add_argument("--demog", default="/Users/ishu/research/PTDEMOG_06Aug2026.csv")
    p.add_argument("--out_dir", default="analysis/results/landmark24")
    p.add_argument("--landmark", choices=["fixed", "subject", "legacy"], default="fixed")
    p.add_argument("--landmark_months", type=float, default=24.0)
    p.add_argument("--grace_days", type=int, default=90)
    p.add_argument("--min_scans", type=int, default=2)
    p.add_argument("--mmse_window_days", type=int, default=365)
    build(p.parse_args())


if __name__ == "__main__":
    main()
