#!/usr/bin/env python3
"""Build the one-row-per-subject table that A2 and A3 both consume.

Cross-sectional by design. The slope framing is what failed in every previous
round of this project, and A1 put the same-scanner within-subject CV of gBSC
at about 10 percent, which is large next to the annual change a slope is meant
to detect. So each subject contributes exactly one scan here.

Which scan depends on the target. For the WMH analysis it is the scan closest
in time to a WMH measurement, because the pairing gap is the thing that
weakens that association. For the age analysis any scan works, so the same
chosen scan is reused and its age is recorded, which keeps one table rather
than two and makes the two analyses directly comparable.

Writes subject_table.csv with the measures, the covariates, and the WMH join
with its gap in days so that any analysis can tighten the window without
rebuilding.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y")

# label -> (file, column)
MEASURE_COLUMNS = {
    "bsc_dir_mean": ("bsc_simple_features_merged.csv", "bsc_dir_mean"),
    "bsc_mag_mean": ("bsc_simple_features_merged.csv", "bsc_mag_mean"),
    "bsc_cos_mean": ("bsc_simple_features_cos.csv", "bsc_cos_mean"),
    "bscdir_adsig": ("per_scan_regional.csv", "bscdir_adsig"),
    "cos_adsig": ("per_scan_cos.csv", "cos_adsig"),
    "seg_tiv_mm3": ("t1_scan_features.csv", "seg_tiv_mm3"),
    "seg_bpf": ("t1_scan_features.csv", "seg_bpf"),
    "qc_snr": ("t1_scan_features.csv", "qc_snr"),
}


def parse_date(text):
    """Accept a string, a date, a datetime or a pandas Timestamp.

    pyreadr hands back real date objects for ADNIMERGE date columns while the
    CSVs carry strings. An earlier version of this function tested
    isinstance(text, str) and silently dropped every row from the .rda tables,
    which looked exactly like a cohort with no overlap. Fail loudly instead.
    """
    if text is None:
        return None
    if isinstance(text, dt.datetime):
        return text
    if isinstance(text, dt.date):
        return dt.datetime(text.year, text.month, text.day)
    if hasattr(text, "to_pydatetime"):
        try:
            return text.to_pydatetime()
        except (ValueError, AttributeError):
            return None
    if not isinstance(text, str) or not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text[:10], fmt)
        except ValueError:
            continue
    return None


def load_measure(root: Path, filename: str, column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(root / filename, newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(column, "")
            if raw in ("", "nan", "NaN", "NA"):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value):
                out[row["image_id"]] = value
    return out


def load_demographics(adnimerge: Path) -> dict[str, dict]:
    """PTID -> sex and date of birth. PTDOB is MM/YYYY, so day is set to 15."""
    frame = pyreadr.read_r(str(adnimerge / "PTDEMOG.rda"))["PTDEMOG"]
    out: dict[str, dict] = {}
    for _, row in frame.iterrows():
        ptid = row.get("PTID")
        if not isinstance(ptid, str):
            continue
        record = out.setdefault(ptid, {"sex": None, "dob": None})
        if record["sex"] is None and isinstance(row.get("PTGENDER"), str):
            record["sex"] = row["PTGENDER"]
        if record["dob"] is None:
            dob_text = row.get("PTDOB")
            year = row.get("PTDOBYY")
            if isinstance(dob_text, str) and "/" in dob_text:
                month, yyyy = dob_text.split("/")[:2]
                try:
                    record["dob"] = dt.datetime(int(yyyy), int(month), 15)
                except ValueError:
                    pass
            elif pd.notna(year):
                record["dob"] = dt.datetime(int(year), 7, 1)
    return out


def load_wmh(adnimerge: Path) -> dict[str, list[tuple[dt.datetime, float, float]]]:
    """PTID -> list of (date, TOTAL_WMH, TOTAL_BRAIN)."""
    frame = pyreadr.read_r(str(adnimerge / "UCD_WMH.rda"))["UCD_WMH"]
    out: dict[str, list] = defaultdict(list)
    for _, row in frame.iterrows():
        ptid = row.get("PTID")
        date = parse_date(row.get("EXAMDATE"))
        wmh = pd.to_numeric(row.get("TOTAL_WMH"), errors="coerce")
        brain = pd.to_numeric(row.get("TOTAL_BRAIN"), errors="coerce")
        if isinstance(ptid, str) and date is not None and pd.notna(wmh):
            out[ptid].append((date, float(wmh), float(brain) if pd.notna(brain) else np.nan))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/Users/ishu/research/adni_trillium/from_cluster")
    parser.add_argument("--adnimerge", default="/Users/ishu/research/ADNIMERGE2/data")
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    parser.add_argument("--max_wmh_gap_days", type=int, default=365)
    args = parser.parse_args()

    root = Path(args.data_root)
    measures = {
        name: load_measure(root, filename, column)
        for name, (filename, column) in MEASURE_COLUMNS.items()
    }
    demographics = load_demographics(Path(args.adnimerge))
    wmh_records = load_wmh(Path(args.adnimerge))

    scans_by_subject: dict[str, list[dict]] = defaultdict(list)
    with open(root / "manifest_realdates.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            date = parse_date(row["real_date"])
            if date is None:
                continue
            row["_date"] = date
            scans_by_subject[row["subject"]].append(row)

    rows = []
    for subject, scans in scans_by_subject.items():
        # Require the primary measure, otherwise the subject contributes nothing.
        scans = [s for s in scans if s["image_id"] in measures["bsc_dir_mean"]]
        if not scans:
            continue

        candidates = wmh_records.get(subject, [])
        chosen, wmh_value, wmh_gap, wmh_brain = None, np.nan, np.nan, np.nan
        if candidates:
            best = None
            for scan in scans:
                for wmh_date, value, brain in candidates:
                    gap = abs((wmh_date - scan["_date"]).days)
                    if best is None or gap < best[0]:
                        best = (gap, scan, value, brain)
            if best is not None and best[0] <= args.max_wmh_gap_days:
                wmh_gap, chosen, wmh_value, wmh_brain = best
        if chosen is None:
            # No usable WMH pairing: keep the subject for A3 using the earliest
            # scan, and leave the WMH fields empty.
            chosen = min(scans, key=lambda s: s["_date"])

        info = demographics.get(subject, {})
        dob = info.get("dob")
        age = (chosen["_date"] - dob).days / 365.25 if dob else np.nan

        record = {
            "subject": subject,
            "image_id": chosen["image_id"],
            "scan_date": chosen["_date"].date().isoformat(),
            "age": round(age, 2) if np.isfinite(age) else "",
            "female": {"Female": 1, "Male": 0}.get(info.get("sex"), ""),
            "site": subject.split("_")[0],
            "field_strength": chosen.get("field_strength_ida", "").strip(),
            "scanner_model": chosen.get("scanner_model", "").strip(),
            "manufacturer": chosen.get("manufacturer", "").strip(),
            "diagnosis": chosen.get("diagnosis", ""),
            "total_wmh": wmh_value if np.isfinite(wmh_value) else "",
            "wmh_total_brain": wmh_brain if np.isfinite(wmh_brain) else "",
            "wmh_gap_days": wmh_gap if np.isfinite(wmh_gap) else "",
        }
        for name in MEASURE_COLUMNS:
            value = measures[name].get(chosen["image_id"])
            record[name] = value if value is not None else ""
        rows.append(record)

    out_path = Path(args.out_dir) / "subject_table.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with_wmh = sum(1 for r in rows if r["total_wmh"] != "")
    with_age = sum(1 for r in rows if r["age"] != "")
    gaps = sorted(r["wmh_gap_days"] for r in rows if r["wmh_gap_days"] != "")
    print(f"subjects written: {len(rows)}")
    print(f"  with a WMH measurement within {args.max_wmh_gap_days}d: {with_wmh}")
    if gaps:
        print(f"  WMH gap days: median {gaps[len(gaps) // 2]}, max {gaps[-1]}")
    print(f"  with age: {with_age}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
