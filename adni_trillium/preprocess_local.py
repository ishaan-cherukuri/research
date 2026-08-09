"""
Local-filesystem port of mri-bsc/code/preprocess/simple_preproc.py (which is hard-coded to
S3). Same transform: per-volume z-score normalize + naive intensity-threshold brain mask.
This matches the original pipeline stage that feeds atropos_bsc.run_atropos_bsc, which does
its own real N4 bias correction / Otsu masking / resampling / tissue segmentation on
whatever T1 volume it's given.

Usage:
    python3 preprocess_local.py --manifest manifest.csv --out_root $SCRATCH/.../preprocess \
        [--skip N] [--limit N]
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


def preprocess_one(t1_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(t1_path)
    data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)

    data = (data - float(data.mean())) / (float(data.std()) + 1e-6)

    brain_mask = (data > -1).astype(np.uint8)
    gm = np.clip(data, 0, 1).astype(np.float32)
    wm = np.clip(1 - gm, 0, 1).astype(np.float32)

    out_files = {
        "t1w_preproc.nii.gz": data,
        "gm_prob.nii.gz": gm,
        "wm_prob.nii.gz": wm,
        "brain_mask.nii.gz": brain_mask,
    }
    for name, arr in out_files.items():
        nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(out_dir / name))

    meta = {"source": t1_path, "steps": ["normalize", "mask", "gm_wm_placeholder"]}
    (out_dir / "preprocess_metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    required = {"subject", "visit_code", "acq_date", "path", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    start = args.skip
    stop = None if args.limit is None else args.skip + args.limit
    df = df.iloc[start:stop].reset_index(drop=True)

    out_root = Path(args.out_root)
    pbar = tqdm(total=len(df), desc="Preprocessing", unit="scan")

    for _, row in df.iterrows():
        image_id = make_image_id(row)
        out_dir = out_root / image_id
        pbar.set_postfix_str(image_id)
        try:
            preprocess_one(row["path"], out_dir)
        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
        pbar.update(1)

    pbar.close()
    print("[DONE] All scans preprocessed")


if __name__ == "__main__":
    main()
