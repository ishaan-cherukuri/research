"""code.index.build_bsc_file_index

Build a scan-level CSV index pointing to BSC derivative files, plus a subject-level label.

Label rule (as requested):
  label = 1 if the subject's manifest contains diagnosis == 3 at ANY timepoint, else 0.

This script is intentionally lightweight: it can run without pandas (stdlib CSV).
If pandas is available, it will use it for convenience.

Output columns (one row per scan/visit in manifest):
  subject,visit_code,acq_date,image_id,diagnosis,label,
  bsc_dir_map,bsc_mag_map,boundary_band_mask,bsc_metrics,
  t1w_preproc,gm_prob,wm_prob,brain_mask

Typical BSC folder layout assumed:
  <bsc_root>/<image_id>/bsc_dir_map.nii.gz
  <bsc_root>/<image_id>/bsc_mag_map.nii.gz
  ... and related files.

Example:
  python3 -m code.index.build_bsc_file_index \
    --manifest /path/to/adni_manifest.csv \
    --bsc_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
    --out_csv code/index/bsc_file_index.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _join(root: str, *parts: str) -> str:
    root = str(root).rstrip("/")
    tail = "/".join(str(p).strip("/") for p in parts if p is not None)
    return f"{root}/{tail}" if tail else root


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return "" if s.lower() in {"nan", "none"} else s


def _as_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def _is_dx3(x: Any) -> bool:
    v = _as_float(x)
    return v is not None and int(round(v)) == 3


@dataclass(frozen=True)
class ManifestRow:
    subject: str
    visit_code: str
    acq_date: str
    path: str
    diagnosis: float | None


def _read_manifest_rows(manifest_csv: str) -> list[ManifestRow]:
    """Read the manifest using pandas if available, otherwise stdlib CSV."""

    # Try pandas first (fast/robust), but don't require it.
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(manifest_csv)
        required = {"subject", "visit_code", "acq_date", "path", "diagnosis"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        rows: list[ManifestRow] = []
        for _, r in df.iterrows():
            rows.append(
                ManifestRow(
                    subject=_as_str(r.get("subject")),
                    visit_code=_as_str(r.get("visit_code")),
                    acq_date=_as_str(r.get("acq_date")),
                    path=_as_str(r.get("path")),
                    diagnosis=_as_float(r.get("diagnosis")),
                )
            )
        return rows
    except ModuleNotFoundError:
        pass

    # Fallback: stdlib CSV
    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        required = {"subject", "visit_code", "acq_date", "path", "diagnosis"}
        if reader.fieldnames is None:
            raise ValueError("Manifest CSV has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        rows: list[ManifestRow] = []
        for r in reader:
            rows.append(
                ManifestRow(
                    subject=_as_str(r.get("subject")),
                    visit_code=_as_str(r.get("visit_code")),
                    acq_date=_as_str(r.get("acq_date")),
                    path=_as_str(r.get("path")),
                    diagnosis=_as_float(r.get("diagnosis")),
                )
            )
        return rows


def build_bsc_file_index(
    manifest_csv: str,
    bsc_root: str,
    out_csv: str,
    min_visits_per_subject: int = 1,
) -> None:
    rows = _read_manifest_rows(manifest_csv)

    # Subject label: ever hit diagnosis==3 at any timepoint
    label_by_subject: dict[str, int] = {}
    visit_count: dict[str, int] = {}
    for r in rows:
        visit_count[r.subject] = visit_count.get(r.subject, 0) + 1
        prev = label_by_subject.get(r.subject, 0)
        label_by_subject[r.subject] = int(prev or _is_dx3(r.diagnosis))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "subject",
        "visit_code",
        "acq_date",
        "image_id",
        "diagnosis",
        "label",
        "bsc_dir_map",
        "bsc_mag_map",
        "boundary_band_mask",
        "bsc_metrics",
        "t1w_preproc",
        "gm_prob",
        "wm_prob",
        "brain_mask",
    ]

    wrote = 0
    skipped = 0

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        for r in rows:
            if not r.subject or not r.visit_code or not r.acq_date:
                skipped += 1
                continue

            if visit_count.get(r.subject, 0) < int(min_visits_per_subject):
                skipped += 1
                continue

            image_id = f"{r.subject}_{r.visit_code}_{r.acq_date}"
            visit_dir = _join(bsc_root, image_id)

            w.writerow(
                {
                    "subject": r.subject,
                    "visit_code": r.visit_code,
                    "acq_date": r.acq_date,
                    "image_id": image_id,
                    "diagnosis": "" if r.diagnosis is None else r.diagnosis,
                    "label": label_by_subject.get(r.subject, 0),
                    "bsc_dir_map": _join(visit_dir, "bsc_dir_map.nii.gz"),
                    "bsc_mag_map": _join(visit_dir, "bsc_mag_map.nii.gz"),
                    "boundary_band_mask": _join(visit_dir, "boundary_band_mask.nii.gz"),
                    "bsc_metrics": _join(visit_dir, "bsc_metrics.json"),
                    "t1w_preproc": _join(visit_dir, "t1w_preproc.nii.gz"),
                    "gm_prob": _join(visit_dir, "gm_prob.nii.gz"),
                    "wm_prob": _join(visit_dir, "wm_prob.nii.gz"),
                    "brain_mask": _join(visit_dir, "brain_mask.nii.gz"),
                }
            )
            wrote += 1

    print("[OK] Wrote index:", str(out_path))
    print("  rows:", wrote)
    print("  skipped:", skipped)
    print("  subjects:", len(visit_count))
    print(
        "  label=1 subjects:",
        sum(1 for s, lab in label_by_subject.items() if lab == 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Manifest CSV (local or S3)")
    ap.add_argument(
        "--bsc_root",
        required=True,
        help="BSC derivatives root prefix (e.g., s3://.../derivatives/bsc/.../atropos)",
    )
    ap.add_argument(
        "--out_csv",
        default="code/index/bsc_file_index.csv",
        help="Output CSV path (local or S3)",
    )
    ap.add_argument(
        "--min_visits_per_subject",
        type=int,
        default=1,
        help="Optionally filter subjects with fewer than this many manifest rows.",
    )
    args = ap.parse_args()

    build_bsc_file_index(
        manifest_csv=args.manifest,
        bsc_root=args.bsc_root,
        out_csv=args.out_csv,
        min_visits_per_subject=args.min_visits_per_subject,
    )


if __name__ == "__main__":
    main()
