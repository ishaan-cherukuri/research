"""
Build a mri-bsc-style manifest (subject, visit_code, acq_date, path, diagnosis) from the
OASIS-3 BIDS data staged on Trillium at /project/rrg-mchakrav-ab/moncia/oasis_3_bids,
joined against the OASIS UDS B4 CDR table for per-visit diagnosis staging.

OASIS BIDS session labels are day-offsets from a subject-specific reference date
("ses-d0129" = 129 days), which already matches the CDR table's "days_to_visit" field --
no date arithmetic needed, just nearest-day matching (imaging and UDS assessment visits
don't fall on exactly the same day).

Diagnosis mapping (CDRTOT -> ADNI-style 1/2/3 code, for pipeline compatibility):
    CDRTOT == 0   -> 1 (CN)
    CDRTOT == 0.5 -> 2 (MCI)
    CDRTOT >= 1   -> 3 (AD/Dementia)

Usage (run on Trillium, where the BIDS data lives):
    python3 build_manifest_oasis.py \
        --bids_root /project/rrg-mchakrav-ab/moncia/oasis_3_bids \
        --cdr_csv OASIS3_UDSb4_cdr.csv \
        --out_csv manifest_oasis.csv \
        --max_day_gap 180
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# OASIS BIDS/UDS data only carries day-offsets from a de-identified per-subject reference
# date, not real calendar dates. The rest of this pipeline (extract_bsc_slopes.py,
# build_survival_labels.py, etc.) expects acq_date to be a real parseable date for
# pd.to_datetime()-based slope/time-to-event arithmetic. Since only *relative* time matters
# for those computations, synthesize acq_date = arbitrary anchor + day-offset -- this is
# not a real acquisition date, purely a compatibility shim.
DATE_ANCHOR = date(2000, 1, 1)


def day_to_pseudo_date(day: int) -> str:
    return (DATE_ANCHOR + timedelta(days=day)).isoformat()


def cdr_to_code(cdrtot: float) -> int | None:
    if pd.isna(cdrtot):
        return None
    if cdrtot == 0:
        return 1
    if cdrtot == 0.5:
        return 2
    if cdrtot >= 1:
        return 3
    return None


def load_cdr_history(cdr_csv: str) -> dict[str, list[tuple[int, float]]]:
    df = pd.read_csv(cdr_csv)
    df = df.dropna(subset=["days_to_visit", "CDRTOT"])
    df["days_to_visit"] = df["days_to_visit"].astype(int)

    history: dict[str, list[tuple[int, float]]] = {}
    for subject, g in df.groupby("OASISID"):
        g = g.sort_values("days_to_visit")
        history[subject] = list(zip(g["days_to_visit"].tolist(), g["CDRTOT"].tolist()))
    return history


def nearest_cdr(history_rows: list[tuple[int, float]], target_day: int, max_gap: int) -> float | None:
    if not history_rows:
        return None
    day, cdr = min(history_rows, key=lambda r: abs(r[0] - target_day))
    if abs(day - target_day) > max_gap:
        return None
    return cdr


def find_bids_scans(bids_root: str) -> list[dict]:
    out = []
    root_p = Path(bids_root)
    for sub_dir in sorted(root_p.glob("sub-OAS*")):
        subject = sub_dir.name[len("sub-") :]
        for ses_dir in sorted(sub_dir.glob("ses-d*")):
            ses_name = ses_dir.name[len("ses-") :]
            m = re.match(r"^d(\d+)$", ses_name)
            if not m:
                continue
            day = int(m.group(1))
            anat_dir = ses_dir / "anat"
            if not anat_dir.is_dir():
                continue
            # Prefer run-01 T1w if multiple runs exist; skip acq-TSE T2w etc.
            t1w_files = sorted(anat_dir.glob("*_T1w.nii.gz"))
            t1w_files = [f for f in t1w_files if "T2" not in f.name]
            if not t1w_files:
                continue
            chosen = next((f for f in t1w_files if "run-01" in f.name), t1w_files[0])
            out.append({"subject": subject, "day": day, "path": str(chosen)})
    return out


def build_manifest(bids_root: str, cdr_csv: str, out_csv: str, max_day_gap: int) -> pd.DataFrame:
    history = load_cdr_history(cdr_csv)
    print(f"Loaded CDR visit history for {len(history)} subjects")

    scans = find_bids_scans(bids_root)
    print(f"Found {len(scans)} BIDS T1w scans")

    rows = []
    skipped_no_history = 0
    skipped_no_cdr_match = 0

    for sc in scans:
        subject = sc["subject"]
        hist_rows = history.get(subject)
        if not hist_rows:
            skipped_no_history += 1
            continue

        cdr = nearest_cdr(hist_rows, sc["day"], max_day_gap)
        dx_code = cdr_to_code(cdr)
        if dx_code is None:
            skipped_no_cdr_match += 1
            continue

        rows.append(
            {
                "subject": subject,
                "visit_code": f"d{sc['day']:04d}",
                "acq_date": day_to_pseudo_date(sc["day"]),
                "path": sc["path"],
                "diagnosis": dx_code,
                "cdrtot": cdr,
            }
        )

    man_df = pd.DataFrame(rows)
    if len(man_df) > 0:
        man_df = man_df.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    man_df.to_csv(out_csv, index=False)

    print(f"Wrote manifest: {out_csv} (rows={len(man_df)})")
    print(f"  skipped (no CDR history): {skipped_no_history}")
    print(f"  skipped (no CDR visit within {max_day_gap} days): {skipped_no_cdr_match}")
    print(f"  subjects in manifest: {man_df['subject'].nunique() if len(man_df) else 0}")
    return man_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", required=True)
    ap.add_argument("--cdr_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_day_gap", type=int, default=180)
    args = ap.parse_args()

    build_manifest(
        bids_root=args.bids_root,
        cdr_csv=args.cdr_csv,
        out_csv=args.out_csv,
        max_day_gap=args.max_day_gap,
    )


if __name__ == "__main__":
    main()
