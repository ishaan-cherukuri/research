"""
Per-scan T1 morphometry/QC feature extraction, computed mostly from files already produced
by run_bsc_batch.py (no new segmentation compute):

  QC features (brain/background intensity stats, SNR, brain mask volume):
    computed directly from the raw BIDS T1w volume with an Otsu threshold (skimage) to
    separate brain/head from background. NOTE: this does NOT reuse preprocess_local.py's
    naive z>-1 mask -- that threshold turned out too crude on real data (near-zero
    background voxels survived, giving degenerate bg_mean/snr/ratio); Otsu on raw
    intensities is the same class of method atropos_bsc.py already uses for its own mask.

  Morphometry volumes (CSF/GM/WM/brain/TIV/BPF):
    from run_bsc_batch.py's atropos_bsc outputs: gm_prob.nii.gz, wm_prob.nii.gz,
    brain_mask.nii.gz (1mm-isotropic, real N4+Otsu+atropos segmentation). CSF is recovered
    as 1 - gm_prob - wm_prob within the brain mask, since atropos's 3-class kmeans
    probabilities sum to ~1 per voxel (the third/unretained class from atropos_bsc.py).

  Field strength: from the BIDS JSON sidecar (MagneticFieldStrength).

NOTE: seg_ventricles_total_mm3 / seg_ventricles_norm are NOT computed here -- a 3-class
tissue segmentation cannot distinguish ventricular CSF from other CSF without an
atlas/prior-based method. Downstream training handles missing T1 columns by skipping them.

Usage:
    python3 extract_t1_scan_features.py --manifest manifest.csv \
        --bsc_root $SCRATCH/.../bsc --out_csv t1_scan_features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_image_id(row: pd.Series) -> str:
    return f"{row['subject']}_{row['visit_code']}_{row['acq_date']}"


def load_arr(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)


def voxel_volume_mm3(path: Path) -> float | None:
    if not path.exists():
        return None
    img = nib.load(str(path))
    return float(np.abs(np.linalg.det(img.affine[:3, :3])))


def qc_features(raw_t1_path: str) -> dict:
    """QC stats computed directly from the raw BIDS T1w volume with an Otsu threshold
    (skimage), rather than reusing preprocess_local.py's naive z>-1 mask -- that threshold
    turned out too crude to separate brain/head from background on real data (near-zero
    background voxels), giving degenerate bg_mean/snr/ratio. Otsu is the same class of
    method atropos_bsc.py already uses for its own brain mask, just applied here directly
    to raw (unprocessed, unmasked) intensities so a real background region survives."""
    from skimage.filters import threshold_otsu

    p = Path(raw_t1_path)
    if not p.exists():
        return {}
    img = nib.load(str(p))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    vox_vol = float(np.abs(np.linalg.det(img.affine[:3, :3])))

    try:
        thresh = threshold_otsu(data)
    except Exception:
        return {}

    mask_b = data > thresh
    brain_vals = data[mask_b]
    bg_vals = data[~mask_b]

    brain_mean = float(brain_vals.mean()) if brain_vals.size else float("nan")
    brain_std = float(brain_vals.std()) if brain_vals.size else float("nan")
    bg_mean = float(bg_vals.mean()) if bg_vals.size else float("nan")
    bg_std = float(bg_vals.std()) if bg_vals.size else float("nan")

    return {
        "qc_brain_mask_vol_mm3": float(mask_b.sum()) * vox_vol,
        "qc_brain_mean": brain_mean,
        "qc_brain_std": brain_std,
        "qc_snr": brain_mean / (brain_std + 1e-6),
        "qc_bg_mean": bg_mean,
        "qc_bg_std": bg_std,
        "qc_brain_bg_ratio": brain_mean / (bg_mean + 1e-6) if not np.isnan(bg_mean) else float("nan"),
    }


def seg_features(bsc_dir: Path) -> dict:
    gm_path = bsc_dir / "gm_prob.nii.gz"
    wm_path = bsc_dir / "wm_prob.nii.gz"
    mask_path = bsc_dir / "brain_mask.nii.gz"

    gm = load_arr(gm_path)
    wm = load_arr(wm_path)
    mask = load_arr(mask_path)
    if gm is None or wm is None or mask is None:
        return {}

    vox_vol = voxel_volume_mm3(mask_path) or 1.0
    mask_b = mask > 0.5

    gm_prob_m = np.clip(gm[mask_b], 0, 1)
    wm_prob_m = np.clip(wm[mask_b], 0, 1)
    csf_prob_m = np.clip(1.0 - gm_prob_m - wm_prob_m, 0, 1)

    gm_vol = float(gm_prob_m.sum()) * vox_vol
    wm_vol = float(wm_prob_m.sum()) * vox_vol
    csf_vol = float(csf_prob_m.sum()) * vox_vol
    brain_vol = gm_vol + wm_vol
    tiv_vol = float(mask_b.sum()) * vox_vol

    return {
        "seg_csf_mm3": csf_vol,
        "seg_gm_total_mm3": gm_vol,
        "seg_wm_total_mm3": wm_vol,
        "seg_brain_mm3": brain_vol,
        "seg_tiv_mm3": tiv_vol,
        "seg_bpf": brain_vol / tiv_vol if tiv_vol > 0 else float("nan"),
    }


def field_strength(bids_t1_path: str) -> float:
    json_path = Path(bids_t1_path.replace(".nii.gz", ".json"))
    if not json_path.exists():
        return float("nan")
    try:
        meta = json.loads(json_path.read_text())
        return float(meta.get("MagneticFieldStrength", float("nan")))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bsc_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    start = args.skip
    stop = None if args.limit is None else args.skip + args.limit
    df = df.iloc[start:stop].reset_index(drop=True)

    bsc_root = Path(args.bsc_root)

    rows = []
    pbar = tqdm(total=len(df), desc="T1 scan features", unit="scan")

    for _, row in df.iterrows():
        image_id = make_image_id(row)
        out_row = {
            "subject": row["subject"],
            "visit_code": row["visit_code"],
            "acq_date": row["acq_date"],
            "image_id": image_id,
        }
        out_row.update(qc_features(row["path"]))
        out_row.update(seg_features(bsc_root / image_id))
        out_row["meta_field_strength_t"] = field_strength(row["path"])
        rows.append(out_row)
        pbar.update(1)

    pbar.close()

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
