# file: code/ingest/build_manifest_adni.py

import argparse
import os
import re
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ADNI_SUBJECT_RE = re.compile(
    r"(?P<subject>\d{3}_S_\d{4})", re.IGNORECASE
)
ADNI_IMAGEID_RE = re.compile(
    r"(?P<subject>I\d{4,7})", re.IGNORECASE
)


def _parse_adni_ids(basename: str):
    """
    Best-effort parsing of ADNI subject/session from a NIfTI basename.

    Examples of possible basenames:
        ADNI_002_S_0413_MR_MPRAGE_br_raw_2006-02-23.nii.gz
        002_S_0413_MR_MPRAGE_...nii.gz
        I123456.nii.gz

    Returns
    -------
    subject : str
        Parsed subject ID, or basename without extension if nothing else matches.
    session : str or None
        Optional session / visit identifier, if we can infer something like a date.
    """
    base = basename
    if base.lower().endswith(".nii.gz"):
        base = base[:-7]
    elif base.lower().endswith(".nii"):
        base = base[:-4]

    # Try ADNI "002_S_0413" pattern
    m = ADNI_SUBJECT_RE.search(base)
    if m:
        subject = m.group("subject")
    else:
        # Try "I123456" image ID
        m2 = ADNI_IMAGEID_RE.search(base)
        if m2:
            subject = m2.group("subject")
        else:
            # Fallback: use first chunk before underscore
            subject = base.split("_")[0]

    # Try to use a YYYYMMDD or YYYY-MM-DD date chunk as session
    session = None
    # YYYYMMDD
    m_date1 = re.search(r"(20\d{2}[01]\d[0-3]\d)", base)
    # YYYY-MM-DD
    m_date2 = re.search(r"(20\d{2}-[01]\d-[0-3]\d)", base)

    if m_date1:
        session = m_date1.group(1)
    elif m_date2:
        session = m_date2.group(1).replace("-", "")

    return subject, session


def _ensure_extract_from_zip(zip_path: str, nifti_root: str) -> list:
    """
    Extract all NIfTI files from ADNI Imaging.zip into nifti_root
    (while preserving internal directory structure).

    Returns
    -------
    extracted_paths : list[str]
        List of full paths to extracted NIfTI files.
    """
    zip_path = Path(zip_path)
    nifti_root = Path(nifti_root)
    nifti_root.mkdir(parents=True, exist_ok=True)

    extracted_paths = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            name = member.filename
            if name.endswith("/"):
                continue
            if not (name.lower().endswith(".nii") or name.lower().endswith(".nii.gz")):
                continue

            out_path = nifti_root / name
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())

            extracted_paths.append(str(out_path))

    return extracted_paths


def build_manifest_adni(
    zip_path: str,
    nifti_root: str,
    out_csv: str,
    modality: str = "T1w",
) -> pd.DataFrame:
    """
    Build a manifest for ADNI scans from Imaging.zip.

    Parameters
    ----------
    zip_path : str
        Path to ADNI Imaging.zip (containing all .nii/.nii.gz).
    nifti_root : str
        Directory where NIfTI files will be extracted (e.g., data/raw/adni/nifti).
    out_csv : str
        Path to output manifest CSV.
    modality : str
        Modality label to assign (default 'T1w'). You can change or post-filter later.

    Returns
    -------
    df_manifest : pandas.DataFrame
    """
    paths = _ensure_extract_from_zip(zip_path, nifti_root)

    records = []
    for full_path in paths:
        base = os.path.basename(full_path)
        subject, session = _parse_adni_ids(base)

        try:
            nii = nib.load(full_path)
            data = nii.get_fdata(dtype=np.float32)
            shape = data.shape
            zooms = nii.header.get_zooms()[:3]
        except Exception:
            shape = (np.nan, np.nan, np.nan)
            zooms = (np.nan, np.nan, np.nan)

        rec = {
            "subject": subject,
            "session": session,
            "path": full_path,
            "modality": modality,
            "shape_x": shape[0],
            "shape_y": shape[1],
            "shape_z": shape[2],
            "zoom_x": zooms[0],
            "zoom_y": zooms[1],
            "zoom_z": zooms[2],
        }
        records.append(rec)

    df_manifest = pd.DataFrame.from_records(records)
    df_manifest = df_manifest.sort_values(["subject", "session", "path"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_manifest.to_csv(out_csv, index=False)
    print(f"[build_manifest_adni] Wrote manifest with {len(df_manifest)} rows to {out_csv}")
    return df_manifest


def main():
    ap = argparse.ArgumentParser(description="Build ADNI manifest from Imaging.zip.")
    ap.add_argument(
        "--zip_path",
        required=True,
        help="Path to Imaging.zip"
    )
    ap.add_argument(
        "--nifti_root",
        required=True,
        help="Directory where NIfTI files will be extracted (e.g., data/raw/adni/nifti)"
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Output manifest CSV (e.g., data/manifests/adni_manifest.csv)"
    )
    ap.add_argument(
        "--modality",
        default="T1w",
        help="Modality label to assign to these scans (default: T1w)"
    )
    args = ap.parse_args()

    build_manifest_adni(
        zip_path=args.zip_path,
        nifti_root=args.nifti_root,
        out_csv=args.out_csv,
        modality=args.modality,
    )


if __name__ == "__main__":
    main()
