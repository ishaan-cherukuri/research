import argparse
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd


def _subject_from_filename(fname: str):
    """
    Extract OAS subject ID from filenames like:
    OASIS_OAS30090_20250430003539526.nii
    """
    m = re.search(r"OAS\d+", fname)
    return m.group(0) if m else None


def build_manifest_oasis(
    image_root: str,
    csv_path: str,
    out_csv: str,
) -> pd.DataFrame:

    image_root = Path(image_root)
    csv_path = Path(csv_path)
    out_csv = Path(out_csv)

    df = pd.read_csv(csv_path)

    # Build lookup: subject -> list of image paths
    image_map = {}
    for p in image_root.glob("*.nii*"):
        subj = _subject_from_filename(p.name)
        if subj is not None:
            image_map[subj] = p

    records = []
    missing = []

    for _, row in df.iterrows():
        subject = row["Subject"]
        session = row["Visit"]

        img_path = image_map.get(subject)

        if img_path is None:
            missing.append(subject)
            continue

        try:
            nii = nib.load(img_path)
            data = nii.get_fdata(dtype=np.float32)
            shape = data.shape
            zooms = nii.header.get_zooms()[:3]
        except Exception:
            shape = (np.nan, np.nan, np.nan)
            zooms = (np.nan, np.nan, np.nan)

        rec = {
            "subject": subject,
            "session": session,
            "path": str(img_path),
            "modality": "T1w",
            "shape_x": shape[0],
            "shape_y": shape[1],
            "shape_z": shape[2],
            "zoom_x": zooms[0],
            "zoom_y": zooms[1],
            "zoom_z": zooms[2],
            "sex": row["Sex"],
            "age": row["Age"],
            "group": row["Group"],
        }

        records.append(rec)

    if not records:
        raise RuntimeError("OASIS manifest is EMPTY. No CSV subjects matched image files.")

    df_out = pd.DataFrame.from_records(records)
    df_out = df_out.sort_values(["subject", "session"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    print(f"[build_manifest_oasis] Wrote manifest with {len(df_out)} rows to {out_csv}")

    if missing:
        print(f"[build_manifest_oasis] WARNING: {len(missing)} subjects missing images.")
        print("Examples:", missing[:10])

    return df_out


def main():
    ap = argparse.ArgumentParser(description="Build OASIS T1 manifest from curated CSV + folder.")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    build_manifest_oasis(
        image_root=args.image_root,
        csv_path=args.csv_path,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
