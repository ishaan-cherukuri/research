#!/usr/bin/env python3
"""Generate an ADNI "bad rows only" report CSV.

This runs the ADNI folder-date matcher and writes a CSV containing only rows
that failed matching (missing subject folder, no matching year, no NIfTI found,
etc.), including the available session dates found on S3 for debugging.

It can also write the full manifest (optional) so you can keep the pipeline
compatible.

Example:
  python -m code.ingest.build_adni_bad_rows \
    --s3_raw_root s3://.../raw/adni_5 \
    --final_df_csv s3://.../raw/final_df.csv \
    --out_missing_csv s3://.../manifests/adni_missing_rows.csv \
    --out_csv s3://.../manifests/adni_manifest.csv
"""

import argparse

from code.ingest.build_manifest_adni import build_adni_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_df_csv", required=True)
    ap.add_argument("--s3_raw_root", required=True)

    ap.add_argument(
        "--out_missing_csv",
        required=True,
        help="Where to write the 'bad rows only' CSV (can be s3://... or local path).",
    )

    ap.add_argument(
        "--out_csv",
        default="data/manifests/adni_manifest.csv",
        help="Optional: write the full manifest too (default: data/manifests/adni_manifest.csv).",
    )

    ap.add_argument(
        "--subject_col",
        default=None,
        help="Override subject ID column in final_df (e.g., PTID or RID).",
    )
    ap.add_argument(
        "--visit_col",
        default=None,
        help="Override visit column in final_df (e.g., VISCODE).",
    )
    ap.add_argument(
        "--exam_date_col",
        default=None,
        help="Override exam date column in final_df (default: EXAMDATE).",
    )

    args = ap.parse_args()

    build_adni_manifest(
        final_df_csv=args.final_df_csv,
        s3_raw_root=args.s3_raw_root,
        out_csv=args.out_csv,
        subject_col=args.subject_col,
        visit_col=args.visit_col,
        exam_date_col=args.exam_date_col,
        out_missing_csv=args.out_missing_csv,
    )


if __name__ == "__main__":
    main()
