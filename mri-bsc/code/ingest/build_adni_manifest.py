#!/usr/bin/env python3
"""ADNI manifest builder.

This is a thin wrapper around `code.ingest.build_manifest_adni.build_adni_manifest`.

Usage:
  python -m code.ingest.build_adni_manifest --final_df_csv ... --s3_raw_root ... --out_csv ...
"""

import argparse

from code.ingest.build_manifest_adni import build_adni_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final_df_csv", required=True)
    ap.add_argument("--s3_raw_root", required=True)
    ap.add_argument("--out_csv", required=True)
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
    )


if __name__ == "__main__":
    main()
