import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def compute_summary(bsc_root, out_csv):
    bsc_root = Path(bsc_root)
    records = []

    for subj_dir in sorted(bsc_root.iterdir()):
        if not subj_dir.is_dir():
            continue

        subject = subj_dir.name

        nii_files = list(subj_dir.glob("*.nii")) + list(subj_dir.glob("*.nii.gz"))
        if not nii_files:
            print(f"[WARN] No BSC file found for {subject}")
            continue

        bsc_path = nii_files[0]

        nii = nib.load(bsc_path)
        data = nii.get_fdata()

        finite_mask = np.isfinite(data)
        valid = data[finite_mask]

        if valid.size == 0:
            print(f"[WARN] No valid voxels for {subject}")
            continue

        rec = {
            "subject": subject,
            "mean_bsc": float(valid.mean()),
            "std_bsc": float(valid.std()),
            "n_voxels": int(valid.size),
            "bsc_path": str(bsc_path),
        }
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[eval] Wrote BSC summary for {len(df)} subjects to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsc_root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    compute_summary(args.bsc_root, args.out_csv)


if __name__ == "__main__":
    main()
