import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def normalize(img):
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()
    if std == 0:
        return img
    return (img - mean) / std


def preprocess(manifest_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)

    for _, row in df.iterrows():
        in_path = Path(row["path"])
        subject = row["subject"]

        out_subdir = out_dir / subject
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_path = out_subdir / in_path.name

        nii = nib.load(in_path)
        data = nii.get_fdata()

        data = normalize(data)

        new_nii = nib.Nifti1Image(data, nii.affine, nii.header)
        nib.save(new_nii, out_path)

        print(f"[preprocess] Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    preprocess(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
