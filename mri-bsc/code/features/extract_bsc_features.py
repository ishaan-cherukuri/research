"""code.features.extract_bsc_features

Extract ML-ready features from boundary-only BSC maps.

Outputs a scan-level CSV (one row per scan / image_id) with:
  - global boundary distribution stats for bsc_dir and bsc_mag
  - optional coarse spatial-bin means (pseudo-ROIs) to add regional signal without atlas

Assumptions:
  <bsc_root>/<image_id>/bsc_dir_map.nii.gz
  <bsc_root>/<image_id>/bsc_mag_map.nii.gz (optional)
  <bsc_root>/<image_id>/boundary_band_mask.nii.gz  (we treat this as the final mask)

Usage:
  python3 -m code.features.extract_bsc_features \
    --manifest /Users/ishu/Downloads/adni_manifest.csv \
    --bsc_root /Volumes/YAAGL/derivatives/bsc/adni/atropos_v2 \
    --out_csv  /Users/ishu/research/mri-bsc/code/index/bsc_scan_features_v2.csv \
    --bins 2,2,2

"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


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


def _image_id(subject: str, visit_code: str, acq_date: str) -> str:
    return f"{subject}_{visit_code}_{acq_date}"


def _load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(str(path)).dataobj)


def _stats(x: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        out = {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        for p in percentiles:
            out[f"p{int(p)}"] = float("nan")
        return out

    out = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
    }
    qs = np.quantile(x, [p / 100.0 for p in percentiles])
    for p, q in zip(percentiles, qs):
        out[f"p{int(p)}"] = float(q)
    return out


def _bin_means(
    values: np.ndarray, mask: np.ndarray, bins: tuple[int, int, int]
) -> dict[str, float]:
    """Compute mean of values within each spatial bin, restricted to mask."""
    bx, by, bz = bins
    shape = mask.shape

    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return {f"bin_{i}_mean": float("nan") for i in range(bx * by * bz)}

    # bin index per voxel
    xs = np.clip((idx[:, 0] * bx) // shape[0], 0, bx - 1)
    ys = np.clip((idx[:, 1] * by) // shape[1], 0, by - 1)
    zs = np.clip((idx[:, 2] * bz) // shape[2], 0, bz - 1)

    bin_id = xs * (by * bz) + ys * bz + zs

    out: dict[str, float] = {}
    for i in range(bx * by * bz):
        vox = idx[bin_id == i]
        if vox.size == 0:
            out[f"bin_{i}_mean"] = float("nan")
        else:
            v = values[vox[:, 0], vox[:, 1], vox[:, 2]]
            v = v[np.isfinite(v)]
            out[f"bin_{i}_mean"] = float(np.mean(v)) if v.size else float("nan")
    return out


@dataclass(frozen=True)
class Row:
    subject: str
    visit_code: str
    acq_date: str
    diagnosis: Optional[float]
    dt: datetime


def read_manifest(manifest_csv: str) -> list[Row]:
    with open(manifest_csv, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("Manifest CSV missing header")
        required = {"subject", "visit_code", "acq_date", "diagnosis"}
        missing = required - set(r.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        rows: list[Row] = []
        for row in r:
            s = (row.get("subject") or "").strip()
            v = (row.get("visit_code") or "").strip()
            d_raw = (row.get("acq_date") or "").strip()
            dx = row.get("diagnosis")
            try:
                dxv = float(dx) if dx not in (None, "", "nan", "NaN") else None
            except Exception:
                dxv = None
            dt = _parse_date(d_raw)
            if not s or not v or not d_raw or dt is None:
                continue
            d = dt.strftime("%Y-%m-%d")
            rows.append(Row(subject=s, visit_code=v, acq_date=d, diagnosis=dxv, dt=dt))

    rows.sort(key=lambda x: (x.subject, x.dt))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bsc_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_visits_per_subject", type=int, default=4)
    ap.add_argument("--percentiles", default="10,25,50,75,90")
    ap.add_argument(
        "--bins",
        default="2,2,2",
        help="Spatial bins as x,y,z. Use 1,1,1 to disable regional bins.",
    )
    args = ap.parse_args()

    percentiles = [float(x) for x in args.percentiles.split(",") if x.strip()]
    bx, by, bz = [int(x) for x in args.bins.split(",")]

    rows = read_manifest(args.manifest)

    # subject filter
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.subject] = counts.get(r.subject, 0) + 1

    bsc_root = Path(args.bsc_root)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine all output columns
    base_cols = ["subject", "visit_code", "acq_date", "image_id", "diagnosis"]

    def stat_cols(prefix: str) -> list[str]:
        cols = [f"{prefix}_mean", f"{prefix}_std", f"{prefix}_median"]
        cols += [f"{prefix}_p{int(p)}" for p in percentiles]
        return cols

    cols = base_cols + ["Nboundary"]
    cols += stat_cols("bsc_dir")
    cols += stat_cols("bsc_mag")

    if (bx, by, bz) != (1, 1, 1):
        for i in range(bx * by * bz):
            cols.append(f"bsc_dir_bin_{i}_mean")
        for i in range(bx * by * bz):
            cols.append(f"bsc_mag_bin_{i}_mean")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        wrote = 0
        skipped = 0

        for r in rows:
            if counts.get(r.subject, 0) < int(args.min_visits_per_subject):
                continue

            image_id = _image_id(r.subject, r.visit_code, r.acq_date)
            d = bsc_root / image_id
            bdir_p = d / "bsc_dir_map.nii.gz"
            bmag_p = d / "bsc_mag_map.nii.gz"
            mask_p = d / "boundary_band_mask.nii.gz"

            if not bdir_p.exists() or not mask_p.exists():
                skipped += 1
                continue

            mask = _load(mask_p).astype(np.uint8)
            vals_dir = _load(bdir_p).astype(np.float32)
            vals_dir = vals_dir[mask > 0]

            row_out: dict[str, object] = {
                "subject": r.subject,
                "visit_code": r.visit_code,
                "acq_date": r.acq_date,
                "image_id": image_id,
                "diagnosis": "" if r.diagnosis is None else r.diagnosis,
                "Nboundary": int(np.count_nonzero(mask)),
            }

            sdir = _stats(vals_dir, percentiles)
            for k, v in sdir.items():
                row_out[f"bsc_dir_{k}"] = v

            if bmag_p.exists():
                vals_mag = _load(bmag_p).astype(np.float32)
                vals_mag = vals_mag[mask > 0]
                smag = _stats(vals_mag, percentiles)
                for k, v in smag.items():
                    row_out[f"bsc_mag_{k}"] = v
            else:
                for k in (
                    "mean",
                    "std",
                    "median",
                    *[f"p{int(p)}" for p in percentiles],
                ):
                    row_out[f"bsc_mag_{k}"] = float("nan")

            if (bx, by, bz) != (1, 1, 1):
                bd = _load(bdir_p).astype(np.float32)
                bins_dir = _bin_means(bd, mask, (bx, by, bz))
                for i in range(bx * by * bz):
                    row_out[f"bsc_dir_bin_{i}_mean"] = bins_dir[f"bin_{i}_mean"]

                if bmag_p.exists():
                    bm = _load(bmag_p).astype(np.float32)
                    bins_mag = _bin_means(bm, mask, (bx, by, bz))
                    for i in range(bx * by * bz):
                        row_out[f"bsc_mag_bin_{i}_mean"] = bins_mag[f"bin_{i}_mean"]
                else:
                    for i in range(bx * by * bz):
                        row_out[f"bsc_mag_bin_{i}_mean"] = float("nan")

            w.writerow(row_out)
            wrote += 1

    print("[OK] wrote:", str(out_path))
    print("  rows:", wrote)
    print("  skipped (missing files):", skipped)


if __name__ == "__main__":
    main()
