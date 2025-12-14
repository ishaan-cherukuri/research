import argparse
import json
import pandas as pd
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from code.io.s3 import download_to_temp, upload_file, ensure_s3_prefix


def preprocess_one(
    image_s3_path: str,
    out_s3_dir: str,
):
    ensure_s3_prefix(out_s3_dir)

    # --- download ---
    local_nii = download_to_temp(image_s3_path)

    # --- load ---
    img = nib.load(local_nii)
    data = img.get_fdata().astype(np.float32)

    # --- simple normalization (placeholder for N4/skullstrip) ---
    data = (data - data.mean()) / (data.std() + 1e-6)

    # --- fake masks (replace with real segmentation later) ---
    brain_mask = (data > -1).astype(np.uint8)
    gm = np.clip(data, 0, 1)
    wm = np.clip(1 - gm, 0, 1)

    # --- save locally ---
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        out_files = {
            "t1w_preproc.nii.gz": data,
            "gm_prob.nii.gz": gm,
            "wm_prob.nii.gz": wm,
            "brain_mask.nii.gz": brain_mask,
        }

        for name, arr in out_files.items():
            out_path = td / name
            nib.save(nib.Nifti1Image(arr, img.affine, img.header), out_path)
            upload_file(out_path, f"{out_s3_dir}/{name}")

        meta = {
            "source": image_s3_path,
            "steps": ["normalize", "mask", "gm_wm_placeholder"],
        }
        meta_path = td / "preprocess_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        upload_file(meta_path, f"{out_s3_dir}/preprocess_metadata.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)

    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_path = row["path"]

        out_dir = f"{args.out_root}/{image_id}"
        preprocess_one(image_path, out_dir)

        print(f"[OK] Preprocessed {image_id}")


if __name__ == "__main__":
    main()
