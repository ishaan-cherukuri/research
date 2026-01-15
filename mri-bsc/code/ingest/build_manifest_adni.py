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
    prefix = ensure_trailing_slash(prefix)
    paginator = s3.get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            out.append(cp["Prefix"])
    return out


def list_objects(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj


def read_csv_any(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        b, k = parse_s3_uri(path)
        obj = s3.get_object(Bucket=b, Key=k)
        data = obj["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def write_csv_any(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        b, k = parse_s3_uri(path)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        s3.put_object(Bucket=b, Key=k, Body=buf.getvalue().encode("utf-8"))
    else:
        df.to_csv(path, index=False)


# -------------------------
# Date parsing (folder names)
# -------------------------
_DATE_PATTERNS = [
    "%Y-%m-%d_%H_%M_%S.%f",  # 2012-06-01_13_08_46.0
    "%Y-%m-%d_%H_%M_%S",
    "%Y-%m-%d",
]


def parse_datefolder_name(prefix: str) -> Optional[datetime]:
    base = prefix.strip("/").split("/")[-1]
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(base, fmt)
        except Exception:
            pass
    return None


# -------------------------
# Helper loading
# -------------------------
def load_helper(helper_csv: str) -> pd.DataFrame:
    df = read_csv_any(helper_csv)
    if "PTID" not in df.columns or "EXAMDATE" not in df.columns:
        raise ValueError("helper must contain PTID and EXAMDATE")

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    # visit_code
    v2 = df["VISCODE2"].astype(str) if "VISCODE2" in df.columns else pd.Series([""] * len(df))
    v1 = df["VISCODE"].astype(str) if "VISCODE" in df.columns else pd.Series([""] * len(df))
    visit_code = []
    for a, b in zip(v2, v1):
        a = "" if str(a).lower() == "nan" else str(a)
        b = "" if str(b).lower() == "nan" else str(b)
        vc = a.strip() if a.strip() else b.strip()
        visit_code.append(vc)
    df["visit_code"] = visit_code

    if "DIAGNOSIS" not in df.columns:
        df["DIAGNOSIS"] = ""

    # keep only meaningful rows
    df = df.dropna(subset=["EXAMDATE"])
    df = df[df["visit_code"].astype(str).str.strip() != ""].copy()
    return df


# -------------------------
# Scan inspection
# -------------------------
def inspect_datefolder(bucket: str, date_prefix: str) -> Dict:
    has_json = False
    best_nifti_key = None
    best_size = -1
    has_nii = False

    for obj in list_objects(bucket, date_prefix):
        key = obj["Key"]
        key_l = key.lower()
        size = int(obj.get("Size", 0))

        if key_l.endswith(".json"):
            has_json = True
        if key_l.endswith(".nii.gz") or key_l.endswith(".nii"):
            has_nii = True
            if size > best_size:
                best_size = size
                best_nifti_key = key

        if has_json and has_nii and best_nifti_key is not None:
            # still keep scanning to maybe find larger nii, but you can early-exit if you want speed
            pass

    return {"has_json": has_json, "has_nii": has_nii, "best_key": best_nifti_key}


# -------------------------
# Selection: pick 4 scans spread out in time
# -------------------------
def can_add_with_gap(chosen_dts: List[datetime], candidate: datetime, min_gap_days: int) -> bool:
    if not chosen_dts:
        return True
    min_gap = min(abs((candidate - d).days) for d in chosen_dts)
    return min_gap >= min_gap_days


def pick_spread_out_scans(scans: List[Dict], k: int = 4, min_gap_days: int = 120) -> List[Dict]:
    """
    scans: list of dict with acq_dt
    Returns up to k scans spread out, relaxing gap if needed.
    """
    scans = sorted(scans, key=lambda x: x["acq_dt"])
    if len(scans) <= k:
        return scans

    # start with earliest + latest
    chosen = [scans[0], scans[-1]]
    remaining = scans[1:-1]

    gap = min_gap_days
    while len(chosen) < k:
        # pick candidate that maximizes distance to current chosen set
        best = None
        best_score = -1

        for sc in remaining:
            if not can_add_with_gap([c["acq_dt"] for c in chosen], sc["acq_dt"], gap):
                continue
            score = min(abs((sc["acq_dt"] - c["acq_dt"]).days) for c in chosen)
            if score > best_score:
                best_score = score
                best = sc

        if best is None:
            # relax the gap until we can fill
            gap = max(0, gap - 30)
            if gap == 0:
                # no gap constraint: just pick by farthest-point
                best = max(
                    remaining,
                    key=lambda sc: min(abs((sc["acq_dt"] - c["acq_dt"]).days) for c in chosen),
                )
            else:
                continue

        chosen.append(best)
        remaining.remove(best)

    return sorted(chosen, key=lambda x: x["acq_dt"])


def nearest_helper_row(helper_rows: List[Dict], acq_dt: datetime) -> Dict:
    """
    Assign scan to nearest helper EXAMDATE (no year restriction).
    """
    best = None
    best_delta = float("inf")
    for hr in helper_rows:
        dt = hr["EXAMDATE"]
        delta = abs((acq_dt - dt).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best = hr
    return best


# -------------------------
# Main
# -------------------------
def build_manifest_less_strict(
    s3_root: str,
    helper_csv: str,
    out_manifest: str,
    out_report: str,
    min_gap_days: int = 120,
    max_subjects: Optional[int] = None,
    subject_regex: Optional[str] = None,
) -> None:
    bucket, root_key = parse_s3_uri(s3_root)
    root_key = ensure_trailing_slash(root_key)

    helper = load_helper(helper_csv)
    helper_by_ptid = {ptid: sub.sort_values("EXAMDATE").copy() for ptid, sub in helper.groupby("PTID")}

    subject_prefixes = list_common_prefixes(bucket, root_key)

    if subject_regex:
        rx = re.compile(subject_regex)
        subject_prefixes = [p for p in subject_prefixes if rx.search(p.split("/")[-2])]
    if max_subjects is not None:
        subject_prefixes = subject_prefixes[:max_subjects]

    manifest_rows = []
    report_rows = []

    for subj_prefix in subject_prefixes:
        subject_id = subj_prefix.strip("/").split("/")[-1]

        if subject_id not in helper_by_ptid:
            report_rows.append({"subject": subject_id, "passed": "no", "reason": "not_in_helper"})
            continue

        sub_helper = helper_by_ptid[subject_id]
        # Make python datetimes WITHOUT deprecated dt.to_pydatetime bulk behavior:
        helper_rows = []
        for _, r in sub_helper.iterrows():
            ts = r["EXAMDATE"]  # pandas Timestamp
            if pd.isna(ts):
                continue
            helper_rows.append(
                {
                    "EXAMDATE": ts.to_pydatetime(),  # per-row conversion (no warning)
                    "visit_code": str(r["visit_code"]),
                    "DIAGNOSIS": str(r.get("DIAGNOSIS", "")),
                }
            )

        # Expecting 4 helper rows per subject, but we won't hard-crash if not
        if len(helper_rows) < 4:
            report_rows.append({"subject": subject_id, "passed": "no", "reason": f"helper_rows<{len(helper_rows)}"})
            continue

        # List datefolders
        date_prefixes_all = list_common_prefixes(bucket, ensure_trailing_slash(subj_prefix))

        scan_infos = []
        for dp in date_prefixes_all:
            acq_dt = parse_datefolder_name(dp)
            if acq_dt is None:
                continue
            info = inspect_datefolder(bucket, dp)
            if not (info["has_nii"] and info["has_json"] and info["best_key"]):
                continue

            scan_infos.append(
                {
                    "acq_dt": acq_dt,
                    "best_key": info["best_key"],
                }
            )

        if len(scan_infos) < 4:
            report_rows.append({"subject": subject_id, "passed": "no", "reason": f"valid_scans<{len(scan_infos)}"})
            continue

        # Pick 4 spread-out scans (less strict)
        chosen_scans = pick_spread_out_scans(scan_infos, k=4, min_gap_days=min_gap_days)

        # Assign each chosen scan to nearest helper row
        # Ensure each helper visit_code doesn't get duplicated too badly:
        # We'll do a simple unique assignment by trying to use distinct helper rows when possible.
        unused_helpers = helper_rows.copy()
        assigned = []

        for sc in chosen_scans:
            if unused_helpers:
                # pick nearest among unused helpers first
                hr = nearest_helper_row(unused_helpers, sc["acq_dt"])
                unused_helpers.remove(hr)
            else:
                # fallback (shouldn't happen with 4 scans / 4 helper rows)
                hr = nearest_helper_row(helper_rows, sc["acq_dt"])

            assigned.append((sc, hr))

        # Write manifest rows
        for sc, hr in sorted(assigned, key=lambda t: t[0]["acq_dt"]):
            manifest_rows.append(
                {
                    "subject": subject_id,
                    "visit_code": hr["visit_code"],
                    "acq_date": sc["acq_dt"].date().isoformat(),
                    "path": f"s3://{bucket}/{sc['best_key']}",
                    "diagnosis": hr["DIAGNOSIS"],
                }
            )

        report_rows.append(
            {
                "subject": subject_id,
                "passed": "yes",
                "n_valid_scans": len(scan_infos),
                "chosen_acq_dates": ";".join([c["acq_dt"].date().isoformat() for c in chosen_scans]),
            }
        )

    man_df = pd.DataFrame(manifest_rows)
    rep_df = pd.DataFrame(report_rows)

    if len(man_df) > 0:
        man_df = man_df.sort_values(["subject", "acq_date", "visit_code"])

    write_csv_any(man_df, out_manifest)
    write_csv_any(rep_df, out_report)

    print(f"Wrote manifest: {out_manifest} (rows={len(man_df)})")
    print(f"Wrote report:   {out_report} (subjects={len(rep_df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-root", required=True)
    ap.add_argument("--helper-csv", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-report", required=True)
    ap.add_argument("--min-gap-days", type=int, default=120)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--subject-regex", default=None)
    args = ap.parse_args()

    build_manifest_less_strict(
        s3_root=args.s3_root,
        helper_csv=args.helper_csv,
        out_manifest=args.out_manifest,
        out_report=args.out_report,
        min_gap_days=args.min_gap_days,
        max_subjects=args.max_subjects,
        subject_regex=args.subject_regex,
    )


if __name__ == "__main__":
    main()
