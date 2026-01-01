import argparse
import pandas as pd
from tqdm import tqdm
from importlib import import_module
from pathlib import Path

from code.io.s3 import (
    download_to_temp,
    upload_file,
    clear_s3_prefix,
)


def copy_preproc_to_bsc(image_id, preproc_root, bsc_dir):
    files = {
        "t1w_preproc.nii.gz": f"{preproc_root}/{image_id}/t1w_preproc.nii.gz",
        "gm_prob.nii.gz": f"{preproc_root}/{image_id}/gm_prob.nii.gz",
        "wm_prob.nii.gz": f"{preproc_root}/{image_id}/wm_prob.nii.gz",
        "brain_mask.nii.gz": f"{preproc_root}/{image_id}/brain_mask.nii.gz",
    }

    for name, s3_path in files.items():
        local = download_to_temp(s3_path)
        upload_file(local, f"{bsc_dir}/{name}")


def run_batch(
    manifest_csv,
    engine,
    out_root,
    limit=None,
    skip=0,
    preproc_root="s3://ishaan-research/data/derivatives/preprocess/adni",
    **kwargs,
):
    df = pd.read_csv(manifest_csv)
    df = df.iloc[int(skip) :]
    if limit is not None:
        df = df.head(int(limit))

    # Clear output directory before starting
    clear_s3_prefix(out_root)

    if engine == "atropos":
        mod = import_module("code.seg.atropos_bsc")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row["image_id"]

            bsc_dir = f"{out_root}/{image_id}"

            # 1. Copy preprocessing outputs into BSC folder
            copy_preproc_to_bsc(image_id, preproc_root, bsc_dir)

            # 2. Run Atropos using the copied T1 (download locally first since ANTs can't read S3 paths)
            t1_s3_path = f"{bsc_dir}/t1w_preproc.nii.gz"
            t1_path = download_to_temp(t1_s3_path)

            mod.run_atropos_bsc(
                t1_path=t1_path,
                out_dir=bsc_dir,
                eps=kwargs.get("eps", 0.05),
                sigma_mm=kwargs.get("sigma_mm", 1.0),
            )

            print(f"[OK] BSC bundle complete → {image_id}")

    elif engine == "freesurfer":
        mod = import_module("code.seg.freesurfer_bsc")
        subjects_dir = kwargs.get("subjects_dir")
        if not subjects_dir:
            raise ValueError("Provide --subjects_dir for FreeSurfer engine")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            subject_id = row["subject"]
            out_dir = f"{out_root}/{subject_id}"

            mod.run_fs_bsc(
                subjects_dir,
                subject_id,
                t1_mgz=kwargs.get("t1_mgz", "mri/brain.mgz"),
                offsets_mm=tuple(
                    map(float, kwargs.get("offsets", "-2,-1,0,1,2").split(","))
                ),
                out_dir=out_dir,
            )

            print(f"[OK] FS-BSC complete → {subject_id}")

    else:
        raise ValueError("Unknown engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--engine", required=True, choices=["atropos", "freesurfer"])
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--skip", type=int, default=0, help="Skip first N images (0-indexed)"
    )
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    ap.add_argument("--subjects_dir")
    ap.add_argument("--t1_mgz", default="mri/brain.mgz")
    ap.add_argument("--offsets", default="-2,-1,0,1,2")
    args = ap.parse_args()

    run_batch(
        args.manifest,
        args.engine,
        args.out_root,
        limit=args.limit,
        skip=args.skip,
        eps=args.eps,
        sigma_mm=args.sigma_mm,
        subjects_dir=args.subjects_dir,
        t1_mgz=args.t1_mgz,
        offsets=args.offsets,
    )
