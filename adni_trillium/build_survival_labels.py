"""
Build MCI->AD conversion survival labels (subject, time_years, event) from manifest.csv,
matching the schema expected by mri-bsc's train_xgb_survival_combined.py
(y = df[["time_years", "event"]]).

Cohort definition mirrors the original study (METHODOLOGY.md): baseline diagnosis MCI,
event = conversion to AD at any later visit (right-censored at last available scan
otherwise). diagnosis codes follow ADNI convention: 1=CN, 2=MCI, 3=AD (see DX_MAP in
build_manifest.py).

Usage:
    python3 build_survival_labels.py --manifest manifest.csv --out_csv survival_labels.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MCI_CODE = 2
AD_CODE = 3


def build_labels(manifest_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    required = {"subject", "visit_code", "acq_date", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    df = df.dropna(subset=["diagnosis"]).copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["acq_date_dt"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date_dt"])
    df = df.sort_values(["subject", "acq_date_dt"])

    rows = []
    skipped_not_mci = 0
    skipped_single_visit = 0

    for subject, g in df.groupby("subject", sort=False):
        g = g.sort_values("acq_date_dt")
        bl_row = g.iloc[0]

        if int(bl_row["diagnosis"]) != MCI_CODE:
            skipped_not_mci += 1
            continue

        if len(g) < 2:
            skipped_single_visit += 1
            continue

        bl_date = bl_row["acq_date_dt"]
        later = g[g["acq_date_dt"] >= bl_date]
        ad_rows = later[later["diagnosis"] == AD_CODE].sort_values("acq_date_dt")

        if not ad_rows.empty:
            event = 1
            event_date = ad_rows.iloc[0]["acq_date_dt"]
        else:
            event = 0
            event_date = g.iloc[-1]["acq_date_dt"]

        time_years = (event_date - bl_date).days / 365.25

        rows.append(
            {
                "subject": subject,
                "event": event,
                "time_years": round(time_years, 4),
                "n_visits": len(g),
                "baseline_date": bl_date.date().isoformat(),
            }
        )

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df = out_df[out_df["time_years"] > 0].reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Wrote survival labels: {out_csv}")
    print(f"  subjects: {len(out_df)}")
    if len(out_df) > 0:
        print(f"  converters (event=1): {int(out_df['event'].sum())} "
              f"({out_df['event'].mean() * 100:.1f}%)")
    print(f"  skipped (baseline not MCI): {skipped_not_mci}")
    print(f"  skipped (single visit, no follow-up): {skipped_single_visit}")
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    build_labels(args.manifest, args.out_csv)


if __name__ == "__main__":
    main()
