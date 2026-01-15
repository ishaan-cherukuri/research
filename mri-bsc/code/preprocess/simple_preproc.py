import argparse
import io
import json
import re
import shutil
from pathlib import Path
from tqdm import tqdm
import boto3
import nibabel as nib
import numpy as np
import pandas as pd

from code.io.s3 import parse_s3_uri, upload_file, ensure_s3_prefix

s3 = boto3.client("s3")


def _read_csv_any(path: str) -> pd.DataFrame:
    """Read CSV from local path or s3://bucket/key."""
    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(path)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def _clean_stem(filename: str) -> str:
    """
    Make a safe stem for files like *.nii.gz:
      "scan.nii.gz" -> "scan"
    """
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = Path(name).stem
    # sanitize to safe chars
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name


def _make_image_id(row: pd.Series) -> str:
    """
    Build a stable id from manifest columns.
    """
    subject = str(row["subject"])
    visit = str(row["visit_code"])
    acq = str(row["acq_date"])
    # keep it filesystem + s3-key safe
    raw = f"{subject}_{visit}_{acq}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def preprocess_one(
    image_s3_path: str,
    out_s3_dir: str,
    temp_root: Path,
):
    ensure_s3_prefix(out_s3_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    # Create temp folder for this image
    bucket, key = parse_s3_uri(image_s3_path)
    image_stem = _clean_stem(key)
    temp_dir = temp_root / f"temp_{image_stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- download ---
        local_nii = temp_dir / Path(key).name
        s3.download_file(bucket, key, str(local_nii))

        # --- load ---
        img = nib.load(str(local_nii))
        # Avoid get_fdata() default float64 conversion; keep it float32.
        data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)

        # --- normalization ---
        data = (data - float(data.mean())) / (float(data.std()) + 1e-6)

        # --- placeholders ---
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
            out_path = temp_dir / name
            nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(out_path))
            upload_file(out_path, f"{out_s3_dir}/{name}")
            out_path.unlink()

        meta = {
            "source": image_s3_path,
            "steps": ["normalize", "mask", "gm_wm_placeholder"],
        }
        meta_path = temp_dir / "preprocess_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        upload_file(meta_path, f"{out_s3_dir}/preprocess_metadata.json")
        meta_path.unlink()

        local_nii.unlink()

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", required=True, help="local CSV path or s3://.../manifest.csv"
    )
    ap.add_argument("--out_root", required=True, help="s3://... output root prefix")
    ap.add_argument("--temp_root", default="data/splits", help="Local temp dir root")
    ap.add_argument(
        "--skip", type=int, default=0, help="Skip first N rows of the manifest"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="Process at most N rows (after skip)"
    )
    args = ap.parse_args()

    df = _read_csv_any(args.manifest)

    # Validate expected manifest columns
    required = {"subject", "visit_code", "acq_date", "path", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}. Found: {list(df.columns)}"
        )

    if args.skip < 0:
        raise ValueError("--skip must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    start = args.skip
    stop = None if args.limit is None else args.skip + args.limit
    df = df.iloc[start:stop].reset_index(drop=True)

    temp_root = Path(args.temp_root)

    pbar = tqdm(total=len(df), desc="Preprocessing", unit="scan")

    for _, row in df.iterrows():
        image_path = row["path"]
        image_id = _make_image_id(row)

        subj = str(row["subject"])
        vc = str(row["visit_code"])
        acq = str(row["acq_date"])
        pbar.set_postfix_str(f"{subj} {vc} {acq}")

        out_dir = f"{args.out_root.rstrip('/')}/{image_id}"
        preprocess_one(image_path, out_dir, temp_root)

        pbar.update(1)

    pbar.close()
    print("[DONE] All scans preprocessed")


if __name__ == "__main__":
    main()
