"""
Local-path driver for segmentation + BSC computation, calling mri-bsc's
code.seg.atropos_bsc.run_atropos_bsc directly (already local-path clean). Replaces
mri-bsc/code/pipeline/run_batch.py, which is hard S3-dependent.

Takes a manifest slice (--skip/--limit) so GNU Parallel can fan this out across cores
within one whole-node SLURM job on Trillium.

Usage:
    python3 run_bsc_batch.py --manifest manifest.csv \
        --preproc_root $SCRATCH/.../preprocess --out_root $SCRATCH/.../bsc \
        [--skip N] [--limit N] [--eps 0.05] [--sigma_mm 1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from code.seg.atropos_bsc import run_atropos_bsc


def make_image_id(row: pd.Series) -> str:
    return f"{row['subject']}_{row['visit_code']}_{row['acq_date']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--preproc_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work_dir", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    start = args.skip
    stop = None if args.limit is None else args.skip + args.limit
    df = df.iloc[start:stop].reset_index(drop=True)

    preproc_root = Path(args.preproc_root)
    out_root = Path(args.out_root)

    pbar = tqdm(total=len(df), desc="BSC (atropos)", unit="scan")
    n_ok, n_skip, n_err = 0, 0, 0

    for _, row in df.iterrows():
        image_id = make_image_id(row)
        pbar.set_postfix_str(image_id)

        out_dir = out_root / image_id
        if (out_dir / "bsc_dir_map.nii.gz").exists():
            n_skip += 1
            pbar.update(1)
            continue

        t1_path = preproc_root / image_id / "t1w_preproc.nii.gz"
        if not t1_path.exists():
            print(f"[SKIP] missing preproc input: {t1_path}")
            n_skip += 1
            pbar.update(1)
            continue

        try:
            run_atropos_bsc(
                str(t1_path),
                str(out_dir),
                eps=args.eps,
                sigma_mm=args.sigma_mm,
                work_dir=args.work_dir,
            )
            n_ok += 1
        except Exception as e:
            print(f"[ERROR] {image_id}: {e}")
            n_err += 1

        pbar.update(1)

    pbar.close()
    print(f"[DONE] ok={n_ok} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
