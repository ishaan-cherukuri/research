"""code.preprocess.check_manifest_vs_preproc

Check that every scan referenced in a manifest exists in a local preproc folder.

For each manifest row, we compute:
  image_id = <subject>_<visit_code>_<acq_date>

We then verify the folder exists under:
  <preproc_root>/<image_id>/

If missing, we print:
  subject,visit_code,acq_date

Usage:
  python3 -m code.preprocess.check_manifest_vs_preproc \
    --manifest /path/to/adni_manifest.csv \
    --preproc_root /Volumes/YAAGL/derivatives/preprocess/adni_5

Notes:
- `acq_date` is normalized to YYYY-MM-DD to match folder naming.
- Requires manifest columns: subject, visit_code, acq_date.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


def _parse_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _image_id(subject: str, visit_code: str, acq_date_yyyy_mm_dd: str) -> str:
    return f"{subject}_{visit_code}_{acq_date_yyyy_mm_dd}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preproc_root", required=True)
    ap.add_argument(
        "--print_header",
        action="store_true",
        help="Print CSV header for missing rows.",
    )
    ap.add_argument(
        "--require_files",
        default="t1w_preproc.nii.gz",
        help="Comma-separated list of files that must exist inside each folder.",
    )
    args = ap.parse_args()

    preproc_root = Path(args.preproc_root)
    required_files = [
        x.strip() for x in str(args.require_files).split(",") if x.strip()
    ]

    if args.print_header:
        print("subject,visit_code,acq_date")

    total = 0
    missing = 0

    with open(args.manifest, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise SystemExit("Manifest CSV missing header")
        required_cols = {"subject", "visit_code", "acq_date"}
        missing_cols = required_cols - set(r.fieldnames)
        if missing_cols:
            raise SystemExit(
                f"Manifest missing required columns: {sorted(missing_cols)}"
            )

        for row in r:
            subject = (row.get("subject") or "").strip()
            visit_code = (row.get("visit_code") or "").strip()
            acq_raw = (row.get("acq_date") or "").strip()
            dt = _parse_date(acq_raw)
            if not subject or not visit_code or dt is None:
                continue
            acq_date = dt.strftime("%Y-%m-%d")
            image_id = _image_id(subject, visit_code, acq_date)
            total += 1

            folder = preproc_root / image_id
            ok = folder.is_dir()
            if ok and required_files:
                for rf in required_files:
                    if not (folder / rf).exists():
                        ok = False
                        break

            if not ok:
                missing += 1
                print(f"{subject},{visit_code},{acq_date}")

    print(f"[SUMMARY] total_checked={total} missing={missing}")


if __name__ == "__main__":
    main()
