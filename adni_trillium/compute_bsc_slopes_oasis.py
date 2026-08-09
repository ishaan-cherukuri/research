"""
Compute per-subject BSC longitudinal slopes for OASIS, bypassing
mri-bsc/code/features/extract_bsc_slopes.py's load_features_with_dates(), which assumes
ADNI-style 3-underscore-token subject IDs (e.g. "002_S_0619") when parsing "subject" back
out of "image_id" -- this silently breaks for OASIS's single-token IDs ("OAS30001"),
treating every scan as its own 1-visit "subject". bsc_simple_features.csv already has
correct "subject" and "acq_date" columns written directly by extract_bsc_features.py, so
this script uses those as-is and calls the (unmodified, correct) compute_slopes_per_subject
function directly.

Usage:
    python3 compute_bsc_slopes_oasis.py --features bsc_simple_features.csv \
        --out_csv bsc_longitudinal_slopes_oasis.csv --min_visits 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from code.features.extract_bsc_slopes import compute_slopes_per_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_visits", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).sort_values(["subject", "acq_date"]).reset_index(drop=True)

    metadata_cols = ["image_id", "subject", "acq_date", "visit_code", "diagnosis",
                      "n_boundary", "n_total", "N_boundary", "N_total"]
    feature_cols = [c for c in df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c])]

    print(f"BSC features to process: {len(feature_cols)}")
    slopes_df = compute_slopes_per_subject(df, feature_cols, args.min_visits)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slopes_df.to_csv(out_path, index=False)
    print(f"Saved: {args.out_csv} ({len(slopes_df)} subjects, {slopes_df.columns.size} columns)")


if __name__ == "__main__":
    main()
