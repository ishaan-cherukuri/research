import argparse
import io
import re
import pandas as pd
from tqdm import tqdm
from importlib import import_module

import boto3

from code.io.s3 import (
    parse_s3_uri,
    download_to_temp,
    upload_file,
    clear_s3_prefix,
    ensure_s3_prefix,
)

s3 = boto3.client("s3")


def read_csv_any(path: str) -> pd.DataFrame:
    """Read CSV from local path or s3://bucket/key."""
    if path.startswith("s3://"):
        b, k = parse_s3_uri(path)
        obj = s3.get_object(Bucket=b, Key=k)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(path)


def make_image_id(row: pd.Series) -> str:
    """
    Manifest schema:
      subject, visit_code, acq_date, path, diagnosis
    Build stable image id used by preprocess outputs.
    """
    subject = str(row["subject"])
    visit = str(row["visit_code"])
    acq = str(row["acq_date"])
    raw = f"{subject}_{visit}_{acq}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def copy_preproc_to_bsc(image_id: str, preproc_root: str, bsc_dir: str):
    preproc_root = preproc_root.rstrip("/")
    bsc_dir = bsc_dir.rstrip("/")
    ensure_s3_prefix(preproc_root)
    ensure_s3_prefix(bsc_dir)

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
    preproc_root="s3://ishaan-research/data/derivatives/preprocess/adni_5",
    clear_out=False,
    **kwargs,
):
    df = read_csv_any(manifest_csv)

    required = {"subject", "visit_code", "acq_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    df = df.iloc[int(skip):]
    if limit is not None:
        df = df.head(int(limit))

    # Optional clear: safer to NOT delete unless you explicitly ask
    if clear_out:
        clear_s3_prefix(out_root)

    if engine == "atropos":
        mod = import_module("code.seg.atropos_bsc")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Atropos BSC", unit="scan"):
            image_id = make_image_id(row)
            bsc_dir = f"{out_root.rstrip('/')}/{image_id}"

            # 1) Copy preprocessing outputs into this BSC folder
            copy_preproc_to_bsc(image_id, preproc_root, bsc_dir)

            # 2) Run Atropos BSC from the PREPROCESSED T1
            t1_s3_path = f"{bsc_dir}/t1w_preproc.nii.gz"
            t1_path = download_to_temp(t1_s3_path)

            mod.run_atropos_bsc(
                t1_path=t1_path,
                out_dir=bsc_dir,
                eps=float(kwargs.get("eps", 0.05)),
                sigma_mm=float(kwargs.get("sigma_mm", 1.0)),
            )

            # tqdm-friendly logging
            tqdm.write(f"[OK] BSC bundle complete → {image_id}")

    elif engine == "freesurfer":
        mod = import_module("code.seg.freesurfer_bsc")
        subjects_dir = kwargs.get("subjects_dir")
        if not subjects_dir:
            raise ValueError("Provide --subjects_dir for FreeSurfer engine")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="FS BSC", unit="scan"):
            subject_id = str(row["subject"])
            out_dir = f"{out_root.rstrip('/')}/{subject_id}"

            mod.run_fs_bsc(
                subjects_dir,
                subject_id,
                t1_mgz=kwargs.get("t1_mgz", "mri/brain.mgz"),
                offsets_mm=tuple(map(float, str(kwargs.get("offsets", "-2,-1,0,1,2")).split(","))),
                out_dir=out_dir,
            )

            tqdm.write(f"[OK] FS-BSC complete → {subject_id}")

    else:
        raise ValueError("Unknown engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--engine", required=True, choices=["atropos", "freesurfer"])
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip", type=int, default=0, help="Skip first N rows (0-indexed)")

    # Atropos params
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)

    # Preproc root (important!)
    ap.add_argument(
        "--preproc_root",
        default="s3://ishaan-research/data/derivatives/preprocess/adni_5",
        help="Root prefix containing <image_id>/t1w_preproc.nii.gz etc.",
    )

    # Safety: don’t wipe outputs unless explicitly requested
    ap.add_argument("--clear_out", action="store_true", help="DANGEROUS: clear out_root before running")

    # Freesurfer params
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
        preproc_root=args.preproc_root,
        clear_out=args.clear_out,
        eps=args.eps,
        sigma_mm=args.sigma_mm,
        subjects_dir=args.subjects_dir,
        t1_mgz=args.t1_mgz,
        offsets=args.offsets,
    )
