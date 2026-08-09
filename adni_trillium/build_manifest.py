"""
Build a mri-bsc-style manifest (subject, visit_code, acq_date, path, diagnosis) from the
BIDS-converted ADNI data already staged on Trillium, joined against ADNIMERGE2's DXSUM
table for real per-visit diagnosis/date information.

BIDS T1w NIfTIs carry no absolute acquisition date (de-identified), so acq_date is
approximated as: subject's earliest DXSUM visit date (baseline) + the BIDS session's month
offset (parsed from "ses-mNN"; "ses-bl"/"ses-sc" -> offset 0). Diagnosis is then assigned
by nearest-date match against the subject's DXSUM visit history. Sessions with no parseable
month offset (e.g. "ses-nv", "ses-uns1") are skipped.

Usage (run on Trillium, where the BIDS data lives):
    python3 build_manifest.py \
        --bids_roots /project/rrg-mchakrav-ab/moncia/ADNI/bids_adni1 \
                     /project/rrg-mchakrav-ab/moncia/ADNI/bids_adni3 \
                     /project/rrg-mchakrav-ab/moncia/ADNI/bids_adni4_qc \
                     /project/rrg-mchakrav-ab/moncia/ADNI/bids_adni_go2 \
        --dxsum_csv dxsum.csv \
        --subjects_txt matched_subjects.txt \
        --out_csv manifest.csv
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DX_MAP = {
    "CN": 1,
    "SMC": 1,
    "EMCI": 2,
    "MCI": 2,
    "LMCI": 2,
    "AD": 3,
    "Dementia": 3,
}

SES_MONTH_RE = re.compile(r"^(bl|sc)$|^m(\d+)$")


def parse_month_offset(visit_code: str) -> float | None:
    m = SES_MONTH_RE.match(visit_code)
    if not m:
        return None
    if m.group(1) is not None:
        return 0.0
    return float(m.group(2))


def load_ida_history(dxsum_csv: str) -> dict[str, list[tuple[datetime, str]]]:
    """Per-visit diagnosis history from ADNIMERGE2's DXSUM table (PTID, EXAMDATE,
    DIAGNOSIS: CN/MCI/Dementia). NOTE: idaSearch_*.csv's "Research Group" column is
    static per subject (does not change across visit rows) and cannot be used for
    longitudinal diagnosis tracking -- confirmed empirically, 0/1863 subjects had more
    than one distinct value. DXSUM is the real source, matching what the original study
    used (ADNIMERGE-derived final_df.csv)."""
    df = pd.read_csv(dxsum_csv)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df.dropna(subset=["EXAMDATE", "DIAGNOSIS"])

    history: dict[str, list[tuple[datetime, str]]] = {}
    for subject, g in df.groupby("PTID"):
        g = g.sort_values("EXAMDATE")
        rows = list(zip(g["EXAMDATE"].tolist(), g["DIAGNOSIS"].astype(str).tolist()))
        history[subject] = rows
    return history


def nearest_dx(history_rows: list[tuple[datetime, str]], target: datetime) -> str | None:
    if not history_rows:
        return None
    best = min(history_rows, key=lambda r: abs((r[0] - target).total_seconds()))
    return best[1]


def find_bids_scans(bids_roots: list[str], subjects: set[str]) -> list[dict]:
    """subjects are bare IDs without underscores, e.g. '002S0619'."""
    out = []
    for root in bids_roots:
        root_p = Path(root)
        for sub_dir in sorted(root_p.glob("sub-*")):
            bare_id = sub_dir.name[len("sub-") :]
            if bare_id not in subjects:
                continue
            for ses_dir in sorted(sub_dir.glob("ses-*")):
                visit_code = ses_dir.name[len("ses-") :]
                anat_dir = ses_dir / "anat"
                if not anat_dir.is_dir():
                    continue
                t1w_files = sorted(anat_dir.glob("*_T1w.nii.gz"))
                if not t1w_files:
                    continue
                out.append(
                    {
                        "bare_id": bare_id,
                        "visit_code": visit_code,
                        "path": str(t1w_files[0]),
                        "dataset": root_p.name,
                    }
                )
    return out


def bare_to_underscored(bare_id: str) -> str:
    m = re.match(r"^(\d+)S(\d+)$", bare_id)
    if not m:
        raise ValueError(f"Unexpected subject ID format: {bare_id}")
    return f"{m.group(1)}_S_{m.group(2)}"


def build_manifest(bids_roots: list[str], dxsum_csv: str, subjects_txt: str, out_csv: str) -> pd.DataFrame:
    subjects = {s.strip() for s in Path(subjects_txt).read_text().splitlines() if s.strip()}
    print(f"Loaded {len(subjects)} target subjects from {subjects_txt}")

    history = load_ida_history(dxsum_csv)
    print(f"Loaded DXSUM visit history for {len(history)} subjects")

    scans = find_bids_scans(bids_roots, subjects)
    print(f"Found {len(scans)} BIDS T1w scans across {len(bids_roots)} dataset roots")

    rows = []
    skipped_no_offset = 0
    skipped_no_history = 0

    for sc in scans:
        subject = bare_to_underscored(sc["bare_id"])
        hist_rows = history.get(subject)
        if not hist_rows:
            skipped_no_history += 1
            continue

        offset_months = parse_month_offset(sc["visit_code"])
        if offset_months is None:
            skipped_no_offset += 1
            continue

        baseline_date = hist_rows[0][0]
        approx_date = baseline_date + timedelta(days=offset_months * 30.44)

        dx_str = nearest_dx(hist_rows, approx_date)
        dx_code = DX_MAP.get(dx_str)

        rows.append(
            {
                "subject": subject,
                "visit_code": sc["visit_code"],
                "acq_date": approx_date.date().isoformat(),
                "path": sc["path"],
                "diagnosis": dx_code if dx_code is not None else "",
                "diagnosis_str": dx_str or "",
                "dataset": sc["dataset"],
            }
        )

    man_df = pd.DataFrame(rows)
    if len(man_df) > 0:
        man_df = man_df.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    man_df.to_csv(out_csv, index=False)

    print(f"Wrote manifest: {out_csv} (rows={len(man_df)})")
    print(f"  skipped (no DXSUM history): {skipped_no_history}")
    print(f"  skipped (unscheduled/no month offset): {skipped_no_offset}")
    print(f"  subjects in manifest: {man_df['subject'].nunique() if len(man_df) else 0}")
    return man_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_roots", nargs="+", required=True)
    ap.add_argument("--dxsum_csv", required=True)
    ap.add_argument("--subjects_txt", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    build_manifest(
        bids_roots=args.bids_roots,
        dxsum_csv=args.dxsum_csv,
        subjects_txt=args.subjects_txt,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
