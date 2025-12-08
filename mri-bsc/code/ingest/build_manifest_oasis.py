# file: code/ingest/build_manifest_oasis.py

import argparse
import os
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _ensure_extract_from_zip(zip_path: str, nifti_root: str) -> dict:
    """
    Ensure NIfTI files from the OASIS3 derivatives ZIP are extracted.

    Returns
    -------
    stem_to_relpath : dict
        Maps BIDS stem (e.g. 'sub-OAS30096_sess-d2948_T1w') to relative path
        under nifti_root, e.g. 'sub-OAS30096/sess-d2948/anat/sub-OAS30096_sess-d2948_T1w.nii.gz'.
    """
    zip_path = Path(zip_path)
    nifti_root = Path(nifti_root)
    nifti_root.mkdir(parents=True, exist_ok=True)

    stem_to_relpath = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            # Skip folders and non-NIfTI
            if name.endswith("/"):
                continue
            if not (name.lower().endswith(".nii") or name.lower().endswith(".nii.gz")):
                continue

            # Where this file will live on disk
            out_path = nifti_root / name
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

            stem = Path(name).name
            # Remove .nii or .nii.gz extension
            if stem.lower().endswith(".nii.gz"):
                stem = stem[:-7]
            elif stem.lower().endswith(".nii"):
                stem = stem[:-4]

            stem_to_relpath[stem] = name

    return stem_to_relpath


def _parse_bids_from_filename(filename: str):
    """
    Parse subject / session from BIDS-style filenames used in OASIS3.

    Example filenames in scan_summary:
        sub-OAS30096_sess-d2948_T1w.json
        sub-OAS30096_sess-d2948_T1w.nii.gz  (after matching)

    Returns
    -------
    subject : str
        Subject ID (e.g. 'OAS30096').
    session : str
        Session identifier (e.g. 'd2948') or None if not parseable.
    """
    base = os.path.basename(filename)
    if base.endswith(".json"):
        base = base[:-5]
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    parts = base.split("_")
    subject = None
    session = None

    for token in parts:
        if token.startswith("sub-"):
            subject = token.replace("sub-", "")
        elif token.startswith("ses-") or token.startswith("sess-"):
            # OASIS3 sometimes uses "sess-d0123"
            session = token.split("-", 1)[1]

    return subject, session


def build_manifest_oasis(
    scan_csv: str,
    zip_path: str,
    nifti_root: str,
    out_csv: str,
) -> pd.DataFrame:
    """
    Build a manifest for OASIS3 T1w scans using the scan summary CSV and derivatives ZIP.

    Parameters
    ----------
    scan_csv : str
        Path to OASIS3_MR_scan_summary.csv.
    zip_path : str
        Path to OASIS3_derivative_files.zip (containing all NIfTI files).
    nifti_root : str
        Directory where NIfTI files will live (extracted if needed).
    out_csv : str
        Path to write the manifest CSV.

    Returns
    -------
    df_manifest : pandas.DataFrame
    """
    scan_df = pd.read_csv(scan_csv)

    # Filter to T1w MRI scans
    t1_df = scan_df[scan_df["scan category"] == "T1w"].copy()
    if t1_df.empty:
        raise RuntimeError("No T1w scans found in OASIS3_MR_scan_summary.csv")

    # Extract NIfTIs from ZIP and build mapping from BIDS stem -> relative path
    stem_to_relpath = _ensure_extract_from_zip(zip_path, nifti_root)

    records = []
    missing = []

    for _, row in t1_df.iterrows():
        json_filename = row["filename"]  # e.g. sub-OAS30096_sess-d2948_T1w.json
        stem = os.path.basename(json_filename)
        if stem.endswith(".json"):
            stem = stem[:-5]

        relpath = stem_to_relpath.get(stem)
        if relpath is None:
            missing.append(json_filename)
            continue

        full_path = str(Path(nifti_root) / relpath)

        subject, session = _parse_bids_from_filename(json_filename)

        # Basic shape / zooms
        try:
            nii = nib.load(full_path)
            data = nii.get_fdata(dtype=np.float32)
            shape = data.shape
            zooms = nii.header.get_zooms()[:3]
        except Exception:
            shape = (np.nan, np.nan, np.nan)
            zooms = (np.nan, np.nan, np.nan)

        rec = {
            "subject": subject or row.get("subject_id"),
            "session": session,
            "path": full_path,
            "modality": "T1w",
            "shape_x": shape[0],
            "shape_y": shape[1],
            "shape_z": shape[2],
            "zoom_x": zooms[0],
            "zoom_y": zooms[1],
            "zoom_z": zooms[2],
            # Useful scanner metadata from scan_summary
            "MagneticFieldStrength": row.get("MagneticFieldStrength"),
            "Manufacturer": row.get("Manufacturer"),
            "ManufacturersModelName": row.get("ManufacturersModelName"),
            "SoftwareVersions": row.get("SoftwareVersions"),
            "SeriesDescription": row.get("SeriesDescription"),
            "RepetitionTime": row.get("RepetitionTime"),
            "EchoTime": row.get("EchoTime"),
            "FlipAngle": row.get("FlipAngle"),
        }
        records.append(rec)

    if missing:
        print(
            f"[build_manifest_oasis] WARNING: {len(missing)} T1w entries "
            f"could not be matched to NIfTI files in the ZIP. "
            f"Examples: {missing[:5]}"
        )

    df_manifest = pd.DataFrame.from_records(records)
    df_manifest = df_manifest.sort_values(["subject", "session", "path"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(out_csv, index=False)
    print(f"[build_manifest_oasis] Wrote manifest with {len(df_manifest)} rows to {out_csv}")
    return df_manifest


def main():
    ap = argparse.ArgumentParser(description="Build OASIS3 T1w manifest from scan summary + derivatives ZIP.")
    ap.add_argument(
        "--scan_csv",
        required=True,
        help="Path to OASIS3_MR_scan_summary.csv"
    )
    ap.add_argument(
        "--zip_path",
        required=True,
        help="Path to OASIS3_derivative_files.zip"
    )
    ap.add_argument(
        "--nifti_root",
        required=True,
        help="Directory where NIfTI files will be extracted (e.g., data/raw/oasis3/nifti)"
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output manifest CSV (e.g., data/manifests/oasis3_manifest.csv)"
    )
    args = ap.parse_args()

    build_manifest_oasis(
        scan_csv=args.scan_csv,
        zip_path=args.zip_path,
        nifti_root=args.nifti_root,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
