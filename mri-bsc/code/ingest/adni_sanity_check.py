#!/usr/bin/env python3
import argparse
import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd

s3 = boto3.client("s3")


# -------------------------
# S3 helpers
# -------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


def ensure_trailing_slash(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


def list_common_prefixes(bucket: str, prefix: str) -> List[str]:
    """
    List "subdirectories" one level under prefix using Delimiter='/'.
    Returns full prefixes (strings ending with '/').
    """
    prefix = ensure_trailing_slash(prefix)
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"])
    return out


def scan_prefix_for_files(bucket: str, prefix: str) -> Tuple[bool, bool]:
    """
    Recursively scan under prefix until we find at least one .nii.gz (or .nii)
    and at least one .json. Stops early once both are found.
    """
    has_nii = False
    has_json = False

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"].lower()
            if (key.endswith(".nii.gz") or key.endswith(".nii")):
                has_nii = True
            elif key.endswith(".json"):
                has_json = True

            if has_nii and has_json:
                return True, True

    return has_nii, has_json


# -------------------------
# Date parsing
# -------------------------
_DATE_PATTERNS = [
    "%Y-%m-%d_%H_%M_%S.%f",
    "%Y-%m-%d_%H_%M_%S",
    "%Y-%m-%d",
]


def parse_datefolder_name(name: str) -> Optional[datetime]:
    """
    Examples you showed:
      2006-08-02_07_02_00.0
      2008-09-29_11_20_20.0
    We parse the folder basename into a datetime.
    """
    base = name.strip("/").split("/")[-1]
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(base, fmt)
        except Exception:
            pass
    return None


# -------------------------
# Helper CSV loading
# -------------------------
def load_helper_csv(path: str) -> pd.DataFrame:
    """
    Supports local path or s3://bucket/key
    Must contain: PTID, EXAMDATE, VISCODE, DIAGNOSIS (your final_df does).
    """
    if path.startswith("s3://"):
        b, k = parse_s3_uri(path)
        obj = s3.get_object(Bucket=b, Key=k)
        data = obj["Body"].read()
        df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_csv(path)

    # normalize + parse dates
    if "EXAMDATE" not in df.columns or "PTID" not in df.columns:
        raise ValueError("Helper CSV must contain at least PTID and EXAMDATE columns.")

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    return df


def yesno(x: bool) -> str:
    return "yes" if x else "no"


# -------------------------
# Main subject sanity checks
# -------------------------
def run_sanity_checks(
    s3_root: str,
    helper_csv: str,
    out_csv: str,
    max_subjects: Optional[int] = None,
    subject_regex: Optional[str] = None,
) -> None:
    bucket, root_key = parse_s3_uri(s3_root)
    root_key = ensure_trailing_slash(root_key)

    helper = load_helper_csv(helper_csv)

    # helper grouped by PTID: years + dates
    helper = helper.dropna(subset=["EXAMDATE"])
    helper_by_ptid = {}
    for ptid, sub in helper.groupby("PTID"):
        dates = list(sub["EXAMDATE"].dt.to_pydatetime())
        years = sorted({d.year for d in dates})
        helper_by_ptid[ptid] = {"dates": dates, "years": years}

    # list subjects under root
    subject_prefixes = list_common_prefixes(bucket, root_key)

    # optional filtering
    if subject_regex:
        rx = re.compile(subject_regex)
        subject_prefixes = [p for p in subject_prefixes if rx.search(p.split("/")[-2])]

    if max_subjects is not None:
        subject_prefixes = subject_prefixes[:max_subjects]

    rows = []

    for subj_prefix in subject_prefixes:
        subject_id = subj_prefix.strip("/").split("/")[-1]  # e.g., 002_S_0729

        # list date folders under subject
        date_prefixes = list_common_prefixes(bucket, ensure_trailing_slash(subj_prefix))

        # gather scan info per date folder
        date_infos = []
        for dp in date_prefixes:
            dt = parse_datefolder_name(dp)
            has_nii, has_json = scan_prefix_for_files(bucket, dp)
            date_infos.append(
                {
                    "date_prefix": dp,
                    "acq_dt": dt,
                    "acq_year": dt.year if dt else None,
                    "has_nii": has_nii,
                    "has_json": has_json,
                    "has_both": has_nii and has_json,
                }
            )

        n_datefolders = len(date_infos)
        n_nii = sum(x["has_nii"] for x in date_infos)
        n_json = sum(x["has_json"] for x in date_infos)
        n_both = sum(x["has_both"] for x in date_infos)

        # check #1: every date folder has both
        all_datefolders_have_both = (n_datefolders > 0) and all(x["has_both"] for x in date_infos)

        # check #2: >= 4 distinct scans (distinct datefolders with nii)
        # (you said "different timepoints", which in your structure == different date folders)
        n_distinct_scans = sum(x["has_nii"] for x in date_infos)
        at_least_4_scans = n_distinct_scans >= 4

        # check #3: year match vs helper EXAMDATE year for that subject
        helper_info = helper_by_ptid.get(subject_id)
        helper_years = helper_info["years"] if helper_info else []
        helper_dates = helper_info["dates"] if helper_info else []

        # count scans that match *any* helper year
        year_matched = 0
        for x in date_infos:
            if not x["has_nii"]:
                continue
            if x["acq_year"] is None:
                continue
            if x["acq_year"] in helper_years:
                year_matched += 1
            else:
                # no year match
                pass

        # Your instruction: "if there are multiple years match the csv then use datetime to pick closest"
        # That logic matters for building the final manifest mapping; for THIS sanity check we record whether
        # the scan year exists in helper years.
        # We'll require >=4 scans that have a year match.
        year_match_pass = year_matched >= 4

        passed_all = all_datefolders_have_both and at_least_4_scans and year_match_pass

        rows.append(
            {
                "subject": subject_id,
                "check_has_nii_and_json_in_each_datefolder": yesno(all_datefolders_have_both),
                "n_datefolders": n_datefolders,
                "n_scans_with_nii": n_nii,
                "n_scans_with_json": n_json,
                "n_scans_with_nii_and_json": n_both,
                "check_at_least_4_distinct_scans": yesno(at_least_4_scans),
                "n_scans_year_matched_to_helper": year_matched,
                "check_year_match_to_helper": yesno(year_match_pass),
                "passed_all": yesno(passed_all),
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["passed_all", "subject"], ascending=[False, True])
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}  (subjects={len(out_df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-root", required=True, help="e.g. s3://ishaan-research/data/raw/adni_5/")
    ap.add_argument("--helper-csv", required=True, help="e.g. s3://ishaan-research/data/raw/final_df.csv or local file")
    ap.add_argument("--out", default="sanity_checks.csv", help="Output CSV path")
    ap.add_argument("--max-subjects", type=int, default=None, help="Optional limit for quick tests")
    ap.add_argument("--subject-regex", default=None, help="Optional regex to filter subjects (matches subject folder name)")
    args = ap.parse_args()

    run_sanity_checks(
        s3_root=args.s3_root,
        helper_csv=args.helper_csv,
        out_csv=args.out,
        max_subjects=args.max_subjects,
        subject_regex=args.subject_regex,
    )


if __name__ == "__main__":
    main()
