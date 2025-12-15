import argparse
import json
import pandas as pd
import shutil
import boto3
from pathlib import Path

import nibabel as nib
import numpy as np

from code.io.s3 import parse_s3_uri, upload_file, ensure_s3_prefix

s3 = boto3.client("s3")


def preprocess_one(
    image_s3_path: str,
    out_s3_dir: str,
    temp_root: Path = None,
):
    if temp_root is None:
        temp_root = Path("data/splits")

    ensure_s3_prefix(out_s3_dir)

    # Create temp folder for this image
    image_stem = Path(image_s3_path).stem
    temp_dir = temp_root / f"temp_{image_stem}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- download to temp folder ---
        bucket, key = parse_s3_uri(image_s3_path)
        local_nii = temp_dir / Path(key).name
        s3.download_file(bucket, key, str(local_nii))

        # --- load ---
        img = nib.load(local_nii)
        data = img.get_fdata().astype(np.float32)

        # --- simple normalization (placeholder for N4/skullstrip) ---
        data = (data - data.mean()) / (data.std() + 1e-6)

        # --- fake masks (replace with real segmentation later) ---
        brain_mask = (data > -1).astype(np.uint8)
        gm = np.clip(data, 0, 1)
        wm = np.clip(1 - gm, 0, 1)

        # --- save locally and upload ---
        out_files = {
            "t1w_preproc.nii.gz": data,
            "gm_prob.nii.gz": gm,
            "wm_prob.nii.gz": wm,
            "brain_mask.nii.gz": brain_mask,
        }

        for name, arr in out_files.items():
            out_path = temp_dir / name
            nib.save(nib.Nifti1Image(arr, img.affine, img.header), out_path)
            upload_file(out_path, f"{out_s3_dir}/{name}")
            out_path.unlink()  # Delete after upload

        meta = {
            "source": image_s3_path,
            "steps": ["normalize", "mask", "gm_wm_placeholder"],
        }
        meta_path = temp_dir / "preprocess_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        upload_file(meta_path, f"{out_s3_dir}/preprocess_metadata.json")
        meta_path.unlink()  # Delete after upload

        # Delete downloaded image
        local_nii.unlink()

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument(
        "--temp_root",
        default="data/splits",
        help="Temporary directory root for processing",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["path"]

        out_dir = f"{args.out_root}/{image_id}"
        preprocess_one(image_path, out_dir, Path(args.temp_root))

        print(f"[OK] Preprocessed {image_id}")


if __name__ == "__main__":
    main()
