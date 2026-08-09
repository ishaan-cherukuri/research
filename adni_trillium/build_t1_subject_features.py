"""
Build per-subject T1 morphometry/QC features (baseline + longitudinal) from
t1_scan_features.csv (extract_t1_scan_features.py output), matching the column schema
mri-bsc/code/ml/train_xgb_survival_combined.py's T1_FEATURE_COLS expects (minus
seg_ventricles_* -- not computed, see extract_t1_scan_features.py docstring).

Leakage prevention (per METHODOLOGY.md 2.1E): only scans up to each subject's event/censor
time (from survival_labels.csv: baseline_date + time_years) are used for longitudinal
aggregation, matching the original study's methodology.

Usage:
    python3 build_t1_subject_features.py --scan_features t1_scan_features.csv \
        --survival survival_labels.csv --out_csv features_t1_all.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

LONGITUDINAL_METRICS = {
    "brain_vol": "seg_brain_mm3",
    "brain_intensity": "qc_brain_mean",
    "snr": "qc_snr",
    "brain_bg_ratio": "qc_brain_bg_ratio",
}

BASELINE_COLS = [
    "seg_csf_mm3",
    "seg_gm_total_mm3",
    "seg_wm_total_mm3",
    "seg_brain_mm3",
    "seg_tiv_mm3",
    "seg_bpf",
    "qc_brain_mask_vol_mm3",
    "qc_brain_mean",
    "qc_brain_std",
    "qc_snr",
    "qc_brain_bg_ratio",
    "meta_field_strength_t",
]


def compute_longitudinal(times: np.ndarray, values: np.ndarray) -> dict:
    valid = ~np.isnan(values)
    if valid.sum() == 0:
        return {"last": np.nan, "delta": np.nan, "pctchg": np.nan, "slope_yr": np.nan,
                "mean": np.nan, "std": np.nan}

    t = times[valid]
    v = values[valid]

    last = float(v[-1])
    baseline = float(v[0])
    delta = last - baseline
    pctchg = (delta / baseline * 100.0) if baseline != 0 else np.nan

    if valid.sum() >= 2:
        slope, *_ = linregress(t, v)
    else:
        slope = np.nan

    return {
        "last": last,
        "delta": delta,
        "pctchg": pctchg,
        "slope_yr": float(slope) if not np.isnan(slope) else np.nan,
        "mean": float(v.mean()),
        "std": float(v.std()) if valid.sum() >= 2 else np.nan,
    }


def build_features(scan_features_csv: str, survival_csv: str, out_csv: str) -> pd.DataFrame:
    scans = pd.read_csv(scan_features_csv)
    survival = pd.read_csv(survival_csv)

    scans["acq_date_dt"] = pd.to_datetime(scans["acq_date"], errors="coerce")
    survival["baseline_date_dt"] = pd.to_datetime(survival["baseline_date"], errors="coerce")

    rows = []
    for _, srow in survival.iterrows():
        subject = srow["subject"]
        bl_date = srow["baseline_date_dt"]
        cutoff_date = bl_date + pd.Timedelta(days=float(srow["time_years"]) * 365.25)

        sub_scans = scans[
            (scans["subject"] == subject)
            & (scans["acq_date_dt"] >= bl_date)
            & (scans["acq_date_dt"] <= cutoff_date)
        ].sort_values("acq_date_dt")

        if sub_scans.empty:
            continue

        bl_row = sub_scans.iloc[0]
        out_row = {"subject": subject}

        for col in BASELINE_COLS:
            out_row[f"{col}_bl"] = bl_row.get(col, np.nan)

        fs_vals = sub_scans["meta_field_strength_t"].dropna()
        out_row["field_strength_mode_t"] = float(fs_vals.mode().iloc[0]) if len(fs_vals) else np.nan

        years_since_bl = (sub_scans["acq_date_dt"] - bl_date).dt.days / 365.25
        times = years_since_bl.values

        for prefix, col in LONGITUDINAL_METRICS.items():
            vals = sub_scans[col].values.astype(float) if col in sub_scans.columns else np.full(len(sub_scans), np.nan)
            agg = compute_longitudinal(times, vals)
            for k, v in agg.items():
                out_row[f"long_{prefix}_{k}"] = v

        out_row["n_visits_used"] = len(sub_scans)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"[OK] wrote {out_csv} ({len(out_df)} subjects)")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_features", required=True)
    ap.add_argument("--survival", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    build_features(args.scan_features, args.survival, args.out_csv)


if __name__ == "__main__":
    main()
