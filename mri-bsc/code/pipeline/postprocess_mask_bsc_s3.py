"""Post-process voxel-wise BSC maps in S3 to be non-zero only at the GM/WM boundary.

Why this exists
--------------
Some downstream tooling/viewers interpret broad boundary bands (or uncertain tissue posteriors)
so that BSC appears non-zero across much of the brain. This script enforces a *hard mask*
so that BSC maps are exactly zero outside a boundary mask.

Default behavior
----------------
- For each scan folder <out_root>/<image_id>/ in S3:
  - download bsc_dir_map.nii.gz and bsc_mag_map.nii.gz
  - build a boundary mask (default: boundary_band_mask.nii.gz)
  - set map voxels outside the mask to 0
  - upload (overwrite) the maps back to S3

Mask modes
----------
- band: uses boundary_band_mask.nii.gz (already produced by Atropos engine)
- interface: builds a 1-voxel-ish GM/WM interface from gm_prob + wm_prob (+ optional brain_mask)

Optional FreeSurfer integration
-------------------------------
If you have FreeSurfer volumes available (e.g., ribbon.mgz + wm.mgz) you can provide
--fs_root to use them as an additional constraint mask.

Example
-------
python -m code.pipeline.postprocess_mask_bsc_s3 \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_root  s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --temp_root data/splits \
  --mask_mode interface

Notes
-----
- Uses data/splits/tmp for all temp files (macOS-safe).
- Shows tqdm with average seconds/scan.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import boto3
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from code.io.s3 import parse_s3_uri, upload_file


s3 = boto3.client("s3")


def _is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _sanitize_image_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def make_image_id(row: pd.Series) -> str:
    subject = str(row["subject"])
    visit = str(row["visit_code"])
    acq = str(row["acq_date"])
    return _sanitize_image_id(f"{subject}_{visit}_{acq}")


def read_csv_any(path: str) -> pd.DataFrame:
    if _is_s3_path(path):
        b, k = parse_s3_uri(path)
        obj = s3.get_object(Bucket=b, Key=k)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(path)


def s3_exists(s3_path: str) -> bool:
    bucket, key = parse_s3_uri(s3_path)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", None)
        if status == 404:
            return False
        raise


def download_to(local_path: Path, s3_uri: str) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = parse_s3_uri(s3_uri)
    s3.download_file(bucket, key, str(local_path))
    return local_path


def _ensure_project_tmp(temp_root: Path) -> Path:
    tmp_dir = (temp_root / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)

    return tmp_dir


@dataclass(frozen=True)
class ScanPaths:
    image_id: str
    s3_dir: str

    bsc_dir: str
    bsc_mag: str

    boundary_band_mask: str
    gm_prob: str
    wm_prob: str
    brain_mask: str

    metrics_json: str
    metrics_csv: str


def build_scan_paths(out_root: str, image_id: str) -> ScanPaths:
    s3_dir = f"{out_root.rstrip('/')}/{image_id}"
    return ScanPaths(
        image_id=image_id,
        s3_dir=s3_dir,
        bsc_dir=f"{s3_dir}/bsc_dir_map.nii.gz",
        bsc_mag=f"{s3_dir}/bsc_mag_map.nii.gz",
        boundary_band_mask=f"{s3_dir}/boundary_band_mask.nii.gz",
        gm_prob=f"{s3_dir}/gm_prob.nii.gz",
        wm_prob=f"{s3_dir}/wm_prob.nii.gz",
        brain_mask=f"{s3_dir}/brain_mask.nii.gz",
        metrics_json=f"{s3_dir}/bsc_metrics.json",
        metrics_csv=f"{s3_dir}/subject_metrics.csv",
    )


def _load_any_image(path: Path) -> nib.spatialimages.SpatialImage:
    return nib.load(str(path))


def _as_bool_mask(
    img: nib.spatialimages.SpatialImage, threshold: float = 0.5
) -> np.ndarray:
    data = np.asanyarray(img.dataobj)
    if data.dtype.kind in ("i", "u", "b"):
        return data != 0
    return data > threshold


def build_mask_band(
    bsc_img: nib.spatialimages.SpatialImage,
    band_mask_path: Path,
) -> np.ndarray:
    band_img = _load_any_image(band_mask_path)
    band_res = resample_to_img(band_img, bsc_img, interpolation="nearest")
    return _as_bool_mask(band_res, threshold=0.5)


def build_mask_interface(
    bsc_img: nib.spatialimages.SpatialImage,
    gm_prob_path: Path,
    wm_prob_path: Path,
    brain_mask_path: Optional[Path] = None,
) -> np.ndarray:
    gm_img = _load_any_image(gm_prob_path)
    wm_img = _load_any_image(wm_prob_path)

    gm_res = resample_to_img(gm_img, bsc_img, interpolation="linear")
    wm_res = resample_to_img(wm_img, bsc_img, interpolation="linear")

    gm = np.asarray(gm_res.dataobj, dtype=np.float32)
    wm = np.asarray(wm_res.dataobj, dtype=np.float32)

    # Hard labels by max posterior between GM and WM.
    gm_lab = gm >= wm
    wm_lab = wm > gm

    # Thin-ish interface: GM voxels adjacent to WM.
    wm_dil = binary_dilation(wm_lab, iterations=1)
    interface = gm_lab & wm_dil

    if brain_mask_path is not None and brain_mask_path.exists():
        bm_img = _load_any_image(brain_mask_path)
        bm_res = resample_to_img(bm_img, bsc_img, interpolation="nearest")
        interface &= _as_bool_mask(bm_res, threshold=0.5)

    return interface


def build_mask_fs_cortex(
    bsc_img: nib.spatialimages.SpatialImage,
    aparc_or_aseg_path: Path,
) -> np.ndarray:
    """Build cortex-only mask from FreeSurfer segmentation.

    Uses label IDs:
      - left cerebral cortex: 3
      - right cerebral cortex: 42
    """
    seg_img = _load_any_image(aparc_or_aseg_path)
    seg_res = resample_to_img(seg_img, bsc_img, interpolation="nearest")
    seg = np.asarray(seg_res.dataobj)
    return (seg == 3) | (seg == 42)


def write_mask_nifti(
    mask: np.ndarray, ref_img: nib.spatialimages.SpatialImage, out_path: Path
) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, hdr), str(out_path))


def recompute_bsc_metrics_from_maps(
    bsc_dir_img: nib.spatialimages.SpatialImage,
    bsc_mag_img: Optional[nib.spatialimages.SpatialImage],
    mask: np.ndarray,
) -> dict:
    bsc_dir = np.asarray(bsc_dir_img.dataobj, dtype=np.float32)
    vals_dir = bsc_dir[mask]

    if vals_dir.size == 0:
        out = {
            "BSC_dir": float("nan"),
            "BSC_mag": float("nan"),
            "Nboundary": int(mask.sum()),
        }
        return out

    out = {"BSC_dir": float(np.median(vals_dir)), "Nboundary": int(mask.sum())}
    if bsc_mag_img is None:
        out["BSC_mag"] = float("nan")
    else:
        bsc_mag = np.asarray(bsc_mag_img.dataobj, dtype=np.float32)
        vals_mag = bsc_mag[mask]
        out["BSC_mag"] = float(np.median(vals_mag)) if vals_mag.size else float("nan")
    return out


def apply_mask_inplace(
    bsc_map_path: Path,
    mask: np.ndarray,
    outside_value: float = 0.0,
    skip_if_already_masked: bool = True,
) -> bool:
    """Apply mask to a NIfTI map. Returns True if file was modified."""

    img = nib.load(str(bsc_map_path))
    data = img.get_fdata(dtype=np.float32)

    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} != map shape {data.shape}")

    if skip_if_already_masked:
        outside = ~mask
        # If everything outside is already ~0, skip.
        if np.all(np.abs(data[outside]) <= 1e-8):
            return False

    data[~mask] = float(outside_value)

    # NOTE: nibabel infers format from filename extension. A name like
    # "bsc_dir_map.nii.gz.tmp" is not recognized; keep a valid NIfTI suffix.
    name = bsc_map_path.name
    if name.endswith(".nii.gz"):
        tmp_name = name[:-7] + ".tmp.nii.gz"
    elif name.endswith(".nii"):
        tmp_name = name[:-4] + ".tmp.nii"
    else:
        tmp_name = name + ".tmp.nii.gz"
    tmp_path = bsc_map_path.with_name(tmp_name)
    nib.save(
        nib.Nifti1Image(data.astype(np.float32), img.affine, img.header), str(tmp_path)
    )
    tmp_path.replace(bsc_map_path)
    return True


@dataclass(frozen=True)
class ScanRef:
    image_id: str
    subject: Optional[str] = None


def iter_scans_from_manifest(
    manifest: str, limit: Optional[int], skip: int
) -> Iterable[ScanRef]:
    df = read_csv_any(manifest)
    required = {"subject", "visit_code", "acq_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    df = df.iloc[int(skip) :]
    if limit is not None:
        df = df.head(int(limit))

    for _, row in df.iterrows():
        yield ScanRef(image_id=make_image_id(row), subject=str(row["subject"]))


def _parse_image_ids_arg(
    image_id: Optional[str], image_ids: Optional[str]
) -> list[str]:
    out: list[str] = []
    if image_id:
        out.append(image_id.strip().rstrip("/"))
    if image_ids:
        out.extend([x.strip().rstrip("/") for x in image_ids.split(",") if x.strip()])
    # de-dupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="Local path or s3://... manifest CSV")
    ap.add_argument(
        "--image_id",
        default=None,
        help="Process a single image_id (overrides manifest)",
    )
    ap.add_argument(
        "--image_ids",
        default=None,
        help="Comma-separated image_ids to process (overrides manifest).",
    )
    ap.add_argument(
        "--out_root", required=True, help="S3 prefix with <image_id>/bsc_*.nii.gz"
    )
    ap.add_argument(
        "--temp_root",
        default="data/splits",
        help="Local temp root (will use <temp_root>/tmp)",
    )

    ap.add_argument("--mask_mode", choices=["band", "interface"], default="band")
    ap.add_argument("--outside_value", type=float, default=0.0)
    ap.add_argument("--skip_if_already_masked", action="store_true", default=True)
    ap.add_argument(
        "--no_skip_if_already_masked",
        dest="skip_if_already_masked",
        action="store_false",
    )

    ap.add_argument(
        "--write_mask",
        action="store_true",
        help="Overwrite boundary_band_mask.nii.gz in S3 with the final mask used by this script.",
    )
    ap.add_argument(
        "--write_metrics",
        action="store_true",
        help="Overwrite bsc_metrics.json and subject_metrics.csv in S3 after masking.",
    )

    ap.add_argument(
        "--keep_local",
        action="store_true",
        help="Keep per-scan local working folders under <temp_root>/tmp for debugging.",
    )

    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip", type=int, default=0)

    # Optional extra mask from FreeSurfer volumes if you have them (must be specified)
    ap.add_argument(
        "--fs_root",
        default=None,
        help=(
            "Optional S3 prefix for FreeSurfer recon outputs. Expected layout: "
            "<fs_root>/<fs_id>/mri/<seg>.mgz. If provided, the final mask is AND-ed with a "
            "FreeSurfer-derived mask (resampled to the BSC grid)."
        ),
    )
    ap.add_argument(
        "--fs_id",
        choices=["image_id", "subject"],
        default="image_id",
        help="How to name FreeSurfer subject folder under fs_root.",
    )

    ap.add_argument(
        "--fs_mask",
        choices=["interface", "cortex", "both"],
        default="interface",
        help=(
            "Which FreeSurfer constraint to apply if --fs_root is set: "
            "interface=ribbon+wm adjacency, cortex=labels 3/42 from aparc+aseg/aseg, both=AND of both."
        ),
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    temp_root = Path(args.temp_root)
    tmp_dir = _ensure_project_tmp(temp_root)

    explicit_ids = _parse_image_ids_arg(args.image_id, args.image_ids)
    if explicit_ids:
        scans = [ScanRef(image_id=x, subject=None) for x in explicit_ids]
        if args.fs_root and args.fs_id == "subject":
            raise ValueError(
                "--fs_id subject requires --manifest (subject not known for --image_id/--image_ids)"
            )
    else:
        if not args.manifest:
            raise ValueError("Provide --manifest or --image_id/--image_ids")
        scans = list(iter_scans_from_manifest(args.manifest, args.limit, args.skip))
        if not scans:
            raise ValueError("No rows to process after skip/limit")

    times: list[float] = []

    pbar = tqdm(scans, desc="Mask BSC maps", unit="scan")
    for scan in pbar:
        image_id = scan.image_id
        start = time.perf_counter()

        sp = build_scan_paths(args.out_root, image_id)

        # Ensure required files exist in S3
        if not s3_exists(sp.bsc_dir):
            pbar.write(f"[SKIP] Missing bsc_dir_map in S3 → {image_id}")
            continue

        # Work dir per scan
        work_dir = tmp_dir / image_id
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download maps
            local_bsc_dir = work_dir / "bsc_dir_map.nii.gz"
            download_to(local_bsc_dir, sp.bsc_dir)

            local_bsc_mag: Optional[Path]
            if s3_exists(sp.bsc_mag):
                local_bsc_mag = work_dir / "bsc_mag_map.nii.gz"
                download_to(local_bsc_mag, sp.bsc_mag)
            else:
                local_bsc_mag = None

            bsc_img = nib.load(str(local_bsc_dir))

            # Build base mask
            if args.mask_mode == "band":
                if not s3_exists(sp.boundary_band_mask):
                    pbar.write(f"[SKIP] Missing boundary_band_mask in S3 → {image_id}")
                    continue
                local_band = work_dir / "boundary_band_mask.nii.gz"
                download_to(local_band, sp.boundary_band_mask)
                mask = build_mask_band(bsc_img, local_band)

            elif args.mask_mode == "interface":
                if not (s3_exists(sp.gm_prob) and s3_exists(sp.wm_prob)):
                    pbar.write(f"[SKIP] Missing gm_prob/wm_prob in S3 → {image_id}")
                    continue
                local_gm = work_dir / "gm_prob.nii.gz"
                local_wm = work_dir / "wm_prob.nii.gz"
                download_to(local_gm, sp.gm_prob)
                download_to(local_wm, sp.wm_prob)

                local_brain_mask = work_dir / "brain_mask.nii.gz"
                if s3_exists(sp.brain_mask):
                    download_to(local_brain_mask, sp.brain_mask)
                    brain_mask_path: Optional[Path] = local_brain_mask
                else:
                    brain_mask_path = None

                mask = build_mask_interface(
                    bsc_img,
                    local_gm,
                    local_wm,
                    brain_mask_path=brain_mask_path,
                )
            else:
                raise ValueError("Unknown mask_mode")

            # Optional FreeSurfer constraint
            if args.fs_root:
                fs_id = image_id if args.fs_id == "image_id" else (scan.subject or "")
                if not fs_id:
                    pbar.write(
                        f"[WARN] FS subject id unavailable for {image_id}; skipping FS mask"
                    )
                else:
                    fs_dir = f"{args.fs_root.rstrip('/')}/{fs_id}/mri"

                    # Cortex-only constraint via aparc+aseg/aseg
                    if args.fs_mask in {"cortex", "both"}:
                        fs_aparc = f"{fs_dir}/aparc+aseg.mgz"
                        fs_aseg = f"{fs_dir}/aseg.mgz"
                        seg_s3 = (
                            fs_aparc
                            if s3_exists(fs_aparc)
                            else (fs_aseg if s3_exists(fs_aseg) else None)
                        )
                        if seg_s3:
                            local_seg = work_dir / Path(seg_s3).name
                            download_to(local_seg, seg_s3)
                            mask &= build_mask_fs_cortex(bsc_img, local_seg)
                        else:
                            pbar.write(
                                f"[WARN] FS aparc+aseg/aseg not found for {fs_id}; skipping cortex mask"
                            )

                    # Interface constraint via ribbon + wm adjacency
                    if args.fs_mask in {"interface", "both"}:
                        fs_ribbon = f"{fs_dir}/ribbon.mgz"
                        fs_wm = f"{fs_dir}/wm.mgz"
                        if s3_exists(fs_ribbon) and s3_exists(fs_wm):
                            local_ribbon = work_dir / "ribbon.mgz"
                            local_wm_mgz = work_dir / "wm.mgz"
                            download_to(local_ribbon, fs_ribbon)
                            download_to(local_wm_mgz, fs_wm)

                            rib_img = _load_any_image(local_ribbon)
                            wm_img = _load_any_image(local_wm_mgz)

                            rib_res = resample_to_img(
                                rib_img, bsc_img, interpolation="nearest"
                            )
                            wm_res = resample_to_img(
                                wm_img, bsc_img, interpolation="nearest"
                            )

                            rib = _as_bool_mask(rib_res, threshold=0.5)
                            wm_mask = _as_bool_mask(wm_res, threshold=0.5)

                            wm_dil = binary_dilation(wm_mask, iterations=1)
                            mask &= rib & wm_dil
                        else:
                            pbar.write(
                                f"[WARN] FS ribbon/wm not found for {fs_id}; skipping FS interface mask"
                            )

            # Apply mask + upload back
            changed_dir = apply_mask_inplace(
                local_bsc_dir,
                mask,
                outside_value=float(args.outside_value),
                skip_if_already_masked=bool(args.skip_if_already_masked),
            )
            if changed_dir:
                upload_file(local_bsc_dir, sp.bsc_dir)

            if local_bsc_mag is not None and local_bsc_mag.exists():
                changed_mag = apply_mask_inplace(
                    local_bsc_mag,
                    mask,
                    outside_value=float(args.outside_value),
                    skip_if_already_masked=bool(args.skip_if_already_masked),
                )
                if changed_mag:
                    upload_file(local_bsc_mag, sp.bsc_mag)

            # Optionally overwrite boundary mask in S3
            if args.write_mask:
                out_mask_path = work_dir / "boundary_band_mask.nii.gz"
                write_mask_nifti(mask, bsc_img, out_mask_path)
                upload_file(out_mask_path, sp.boundary_band_mask)

            # Optionally recompute and overwrite metrics
            if args.write_metrics:
                bsc_dir_img2 = nib.load(str(local_bsc_dir))
                bsc_mag_img2 = (
                    nib.load(str(local_bsc_mag)) if local_bsc_mag is not None else None
                )
                new_metrics = recompute_bsc_metrics_from_maps(
                    bsc_dir_img2, bsc_mag_img2, mask
                )
                new_metrics.update(
                    {
                        "mask_mode_used": str(args.mask_mode),
                        "fs_mask_used": (str(args.fs_mask) if args.fs_root else None),
                    }
                )

                # JSON
                local_metrics_json = work_dir / "bsc_metrics.json"
                if s3_exists(sp.metrics_json):
                    download_to(local_metrics_json, sp.metrics_json)
                    try:
                        old = json.loads(local_metrics_json.read_text())
                    except Exception:
                        old = {}
                else:
                    old = {}
                old.update(new_metrics)
                local_metrics_json.write_text(json.dumps(old, indent=2))
                upload_file(local_metrics_json, sp.metrics_json)

                # CSV
                local_metrics_csv = work_dir / "subject_metrics.csv"
                if s3_exists(sp.metrics_csv):
                    download_to(local_metrics_csv, sp.metrics_csv)
                    try:
                        dfm = pd.read_csv(local_metrics_csv)
                    except Exception:
                        dfm = pd.DataFrame([{}])
                else:
                    dfm = pd.DataFrame([{}])
                if dfm.empty:
                    dfm = pd.DataFrame([{}])
                for k, v in new_metrics.items():
                    dfm.loc[0, k] = v
                dfm.to_csv(local_metrics_csv, index=False)
                upload_file(local_metrics_csv, sp.metrics_csv)

            dt = time.perf_counter() - start
            times.append(dt)
            avg = float(np.mean(times)) if times else 0.0
            pbar.set_postfix(avg_sec=f"{avg:.1f}", last_sec=f"{dt:.1f}")

        finally:
            if not args.keep_local:
                shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
