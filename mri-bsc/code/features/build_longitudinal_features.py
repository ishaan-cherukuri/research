"""code.features.build_longitudinal_features

Build subject-level longitudinal features from scan-level BSC features.

Given scan-level rows (one per visit), we compute for each numeric feature:
  - baseline value (first scan)
  - last value (last scan)
  - delta (last - baseline)
  - slope per year (linear fit vs time)

Also produces a subject label from the manifest:
  label = 1 if diagnosis reaches 3 at any timepoint, else 0.

Usage:
  python3 -m code.features.build_longitudinal_features \
    --manifest /Users/ishu/Downloads/adni_manifest.csv \
    --scan_features /Users/ishu/research/mri-bsc/code/index/bsc_scan_features_v2.csv \
    --out_csv /Users/ishu/research/mri-bsc/code/index/bsc_subject_features_v2.csv

"""

from __future__ import annotations

import argparse
import csv
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
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _years_between(a: datetime, b: datetime) -> float:
    return float((b - a).days / 365.25)


def _is_dx3(x: str) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return int(round(v)) == 3


def read_subject_labels(
    manifest_csv: str, min_visits: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (label_by_subject, visit_count)."""
    label: dict[str, int] = {}
    count: dict[str, int] = {}
    with open(manifest_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("subject") or "").strip()
            if not s:
                continue
            count[s] = count.get(s, 0) + 1
            dx = (row.get("diagnosis") or "").strip()
            label[s] = int(label.get(s, 0) or _is_dx3(dx))

    # filter counts outside
    return label, count


@dataclass
class ScanRow:
    subject: str
    dt: datetime
    feats: dict[str, float]


def _to_float(v: str) -> float:
    try:
        if v is None:
            return float("nan")
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def read_scan_features(
    scan_features_csv: str,
) -> tuple[list[str], dict[str, list[ScanRow]]]:
    with open(scan_features_csv, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("scan_features CSV missing header")

        base = {"subject", "visit_code", "acq_date", "image_id", "diagnosis"}
        feat_cols = [c for c in r.fieldnames if c not in base]

        by_subject: dict[str, list[ScanRow]] = {}
        for row in r:
            s = (row.get("subject") or "").strip()
            d = (row.get("acq_date") or "").strip()
            dt = _parse_date(d)
            if not s or dt is None:
                continue

            feats = {c: _to_float(row.get(c, "")) for c in feat_cols}
            by_subject.setdefault(s, []).append(ScanRow(subject=s, dt=dt, feats=feats))

    # sort each subject by date
    for s in by_subject:
        by_subject[s] = sorted(by_subject[s], key=lambda x: x.dt)

    return feat_cols, by_subject


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scan_features", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_visits_per_subject", type=int, default=4)
    args = ap.parse_args()

    label_by_subject, visit_count = read_subject_labels(
        args.manifest, args.min_visits_per_subject
    )
    feat_cols, by_subject = read_scan_features(args.scan_features)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Output columns: label + per-feature baseline/last/delta/slope
    cols = ["subject", "label", "n_visits"]
    for c in feat_cols:
        cols.extend([f"{c}__bl", f"{c}__last", f"{c}__delta", f"{c}__slope_per_year"])

    wrote = 0
    skipped = 0

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for subject, scans in by_subject.items():
            if visit_count.get(subject, 0) < int(args.min_visits_per_subject):
                continue
            if len(scans) < 2:
                skipped += 1
                continue

            bl = scans[0]
            t0 = bl.dt
            times = np.array([_years_between(t0, s.dt) for s in scans], dtype=float)

            row_out: dict[str, object] = {
                "subject": subject,
                "label": int(label_by_subject.get(subject, 0)),
                "n_visits": int(len(scans)),
            }

            for c in feat_cols:
                y = np.array([s.feats.get(c, float("nan")) for s in scans], dtype=float)

                blv = float(y[0])
                lv = float(y[-1])
                row_out[f"{c}__bl"] = blv
                row_out[f"{c}__last"] = lv
                row_out[f"{c}__delta"] = (
                    lv - blv if np.isfinite(lv) and np.isfinite(blv) else float("nan")
                )

                # slope per year using finite points
                m = np.isfinite(times) & np.isfinite(y)
                slope = float("nan")
                if np.count_nonzero(m) >= 2:
                    coef = np.polyfit(times[m], y[m], 1)
                    slope = float(coef[0])
                row_out[f"{c}__slope_per_year"] = slope

            w.writerow(row_out)
            wrote += 1

    print("[OK] wrote:", str(out_path))
    print("  subjects:", wrote)
    print("  skipped (insufficient scans):", skipped)


if __name__ == "__main__":
    main()
