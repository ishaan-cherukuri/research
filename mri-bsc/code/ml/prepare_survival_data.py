"""code.ml.prepare_survival_data

Prepare survival analysis datasets from longitudinal BSC features.

Survival analysis for Alzheimer's conversion:
  - Time-to-event: time from first scan to diagnosis = 3 (AD)
  - Censoring: subjects who don't reach dx=3 are right-censored
  - Time-dependent covariates: BSC features change over time

Outputs:
  1. Time-to-conversion CSV (for Cox regression)
  2. Landmark dataset (for time-dependent Cox with BSC at specific timepoints)
  3. Joint model dataset (for joint modeling of longitudinal + survival)

Usage:
  python3 -m code.ml.prepare_survival_data \
    --features data/index/bsc_simple_features.csv \
    --out_dir data/ml/survival

"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def _parse_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _years_between(a: datetime, b: datetime) -> float:
    return (b - a).days / 365.25


@dataclass
class ScanRow:
    subject: str
    visit_code: str
    acq_date: str
    dt: datetime
    diagnosis: Optional[float]
    features: dict[str, float]


def read_features(csv_path: str) -> tuple[list[str], dict[str, list[ScanRow]]]:
    """Read scan-level features and group by subject."""
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("Empty CSV")

        meta_cols = {"subject", "visit_code", "acq_date", "image_id", "diagnosis"}
        feat_cols = [c for c in r.fieldnames if c not in meta_cols]

        by_subject: dict[str, list[ScanRow]] = defaultdict(list)

        for row in r:
            s = row["subject"].strip()
            v = row["visit_code"].strip()
            d = row["acq_date"].strip()
            dt = _parse_date(d)

            if not s or not dt:
                continue

            dx_str = row.get("diagnosis", "").strip()
            try:
                dx = float(dx_str) if dx_str and dx_str != "nan" else None
            except:
                dx = None

            feats = {}
            for c in feat_cols:
                try:
                    feats[c] = float(row[c])
                except:
                    feats[c] = float("nan")

            by_subject[s].append(
                ScanRow(
                    subject=s,
                    visit_code=v,
                    acq_date=d,
                    dt=dt,
                    diagnosis=dx,
                    features=feats,
                )
            )

    # Sort by date
    for s in by_subject:
        by_subject[s].sort(key=lambda x: x.dt)

    return feat_cols, by_subject


def compute_time_to_conversion(by_subject: dict[str, list[ScanRow]]) -> list[dict]:
    """
    Compute time-to-conversion for Cox regression.

    Returns list of dicts with:
      - subject
      - event: 1 if converted to dx=3, 0 if censored
      - time_years: years from first scan to conversion or last scan
      - baseline features (from first scan)
    """
    results = []

    for subj, scans in by_subject.items():
        if len(scans) < 2:
            continue

        t0 = scans[0].dt
        bl_feats = scans[0].features

        # Find first dx=3 visit
        conversion_time = None
        for scan in scans:
            if scan.diagnosis == 3.0:
                conversion_time = _years_between(t0, scan.dt)
                break

        if conversion_time is not None:
            event = 1
            time = conversion_time
        else:
            event = 0
            time = _years_between(t0, scans[-1].dt)

        row = {
            "subject": subj,
            "event": event,
            "time_years": time,
        }

        # Add baseline features with _bl suffix
        for k, v in bl_feats.items():
            row[f"{k}_bl"] = v

        results.append(row)

    return results


def compute_landmark_dataset(
    by_subject: dict[str, list[ScanRow]], landmark_year: float = 1.0
) -> list[dict]:
    """
    Create landmark dataset at a specific time point.

    Only includes subjects alive (not converted) at landmark_year.
    Uses features from scan closest to landmark_year.

    Args:
        landmark_year: years from baseline to create landmark (e.g., 1.0 = 1 year)
    """
    results = []

    for subj, scans in by_subject.items():
        if len(scans) < 2:
            continue

        t0 = scans[0].dt

        # Find conversion time
        conversion_time = None
        for scan in scans:
            if scan.diagnosis == 3.0:
                conversion_time = _years_between(t0, scan.dt)
                break

        # Skip if converted before landmark
        if conversion_time is not None and conversion_time < landmark_year:
            continue

        # Find scan closest to landmark
        closest_scan = None
        min_diff = float("inf")
        for scan in scans:
            t = _years_between(t0, scan.dt)
            if t <= landmark_year:
                diff = abs(t - landmark_year)
                if diff < min_diff:
                    min_diff = diff
                    closest_scan = scan

        if closest_scan is None:
            continue

        # Time from landmark to event/censor
        if conversion_time is not None:
            event = 1
            time = conversion_time - landmark_year
        else:
            event = 0
            time = _years_between(t0, scans[-1].dt) - landmark_year

        if time <= 0:
            continue

        row = {
            "subject": subj,
            "event": event,
            "time_years": time,
            "landmark_year": landmark_year,
        }

        for k, v in closest_scan.features.items():
            row[f"{k}_lm"] = v

        results.append(row)

    return results


def compute_joint_model_dataset(by_subject: dict[str, list[ScanRow]]) -> list[dict]:
    """
    Create dataset for joint longitudinal-survival modeling.

    Each row is one scan with:
      - time from baseline
      - features
      - subject-level event and total follow-up time
    """
    results = []

    for subj, scans in by_subject.items():
        if len(scans) < 2:
            continue

        t0 = scans[0].dt

        # Find conversion
        conversion_time = None
        for scan in scans:
            if scan.diagnosis == 3.0:
                conversion_time = _years_between(t0, scan.dt)
                break

        if conversion_time is not None:
            event = 1
            total_time = conversion_time
        else:
            event = 0
            total_time = _years_between(t0, scans[-1].dt)

        for scan in scans:
            t = _years_between(t0, scan.dt)

            row = {
                "subject": subj,
                "time_from_baseline": t,
                "event": event,
                "total_follow_up": total_time,
                "diagnosis": scan.diagnosis if scan.diagnosis else float("nan"),
            }

            for k, v in scan.features.items():
                row[k] = v

            results.append(row)

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Scan-level features CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--landmark_years",
        default="1,2,3",
        help="Comma-separated landmark timepoints in years",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading features...")
    feat_cols, by_subject = read_features(args.features)
    print(f"[INFO] {len(by_subject)} subjects, {len(feat_cols)} features")

    # 1. Time-to-conversion (baseline Cox)
    print("[INFO] Creating time-to-conversion dataset...")
    ttc = compute_time_to_conversion(by_subject)
    ttc_path = out_dir / "time_to_conversion.csv"

    if ttc:
        with open(ttc_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ttc[0].keys())
            w.writeheader()
            w.writerows(ttc)

        events = sum(r["event"] for r in ttc)
        print(f"[OK] {ttc_path}")
        print(
            f"  Subjects: {len(ttc)}, Events: {events}, Censored: {len(ttc) - events}"
        )

    # 2. Landmark datasets
    landmarks = [float(x) for x in args.landmark_years.split(",")]
    for lm in landmarks:
        print(f"[INFO] Creating landmark dataset at {lm} years...")
        lm_data = compute_landmark_dataset(by_subject, lm)

        if lm_data:
            lm_path = out_dir / f"landmark_{int(lm)}yr.csv"
            with open(lm_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=lm_data[0].keys())
                w.writeheader()
                w.writerows(lm_data)

            events = sum(r["event"] for r in lm_data)
            print(f"[OK] {lm_path}")
            print(f"  Subjects: {len(lm_data)}, Events: {events}")

    # 3. Joint model dataset
    print("[INFO] Creating joint model dataset...")
    joint = compute_joint_model_dataset(by_subject)
    joint_path = out_dir / "joint_longitudinal_survival.csv"

    if joint:
        with open(joint_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=joint[0].keys())
            w.writeheader()
            w.writerows(joint)

        print(f"[OK] {joint_path}")
        print(f"  Total observations: {len(joint)}")
        print(f"  Unique subjects: {len(set(r['subject'] for r in joint))}")

    print("\n[DONE] Survival datasets created in:", out_dir)


if __name__ == "__main__":
    main()
