"""code.pipeline.bsc_v2_local

Local-first BSC v2 pipeline:

- Runs Atropos-based BSC on preprocessed T1s.
- Builds a tight GM/WM interface mask from GM/WM posteriors.
- Optionally intersects with FreeSurfer cortex labels.
- Hard-zeros BSC maps outside the final mask.

This is intended for local datasets (e.g., /Volumes/YAAGL).

Inputs:
  <preproc_root>/<image_id>/t1w_preproc.nii.gz

Outputs:
  <out_root>/<image_id>/bsc_dir_map.nii.gz
  <out_root>/<image_id>/bsc_mag_map.nii.gz
  <out_root>/<image_id>/bsc_metrics.json
  <out_root>/<image_id>/boundary_band_mask.nii.gz  (optionally overwritten to final interface mask)

Optional FreeSurfer (SUBJECTS_DIR):
  <fs_subjects_dir>/<image_id>/mri/aparc+aseg.mgz or aseg.mgz
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import nibabel as nib
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing Python dependency (e.g., numpy/nibabel). "
        "Activate your venv and install project deps, e.g.:\n"
        "  python3 -m venv .venv\n"
        "  source .venv/bin/activate\n"
        "  python3 -m pip install -U pip\n"
        "  python3 -m pip install -e .\n"
        f"Original error: {e}"
    )

from code.seg.atropos_bsc import run_atropos_bsc


def _parse_date(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _image_id(subject: str, visit_code: str, acq_date: str) -> str:
    return f"{subject}_{visit_code}_{acq_date}"


def _parse_image_id(image_id: str) -> tuple[str, str, str]:
    """Return (subject, visit_code, acq_date) from <subject>_<visit_code>_<acq_date>."""
    parts = image_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id: {image_id}")
    visit_code = parts[-2]
    acq_date = parts[-1]
    subject = "_".join(parts[:-2])
    return subject, visit_code, acq_date


def _load_nii(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine, img.header


def _save_like(
    path: Path, data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nib.Nifti1Image(data, affine, header=header), str(tmp))
    tmp.replace(path)


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    try:
        from scipy.ndimage import binary_dilation  # type: ignore

        return binary_dilation(mask, iterations=iterations)
    except Exception:
        return mask


def build_interface_mask(
    gm_prob: np.ndarray, wm_prob: np.ndarray, brain_mask: np.ndarray
) -> np.ndarray:
    gm = gm_prob > 0.5
    wm = wm_prob > 0.5
    gm_d = _binary_dilation(gm, iterations=1)
    wm_d = _binary_dilation(wm, iterations=1)
    interface = gm_d & wm_d
    if brain_mask is not None:
        interface = interface & (brain_mask > 0)
    return interface.astype(np.uint8)


def load_fs_seg(fs_subjects_dir: Path, image_id: str) -> Optional[Path]:
    base = fs_subjects_dir / image_id / "mri"
    for name in ("aparc+aseg.mgz", "aseg.mgz"):
        p = base / name
        if p.exists():
            return p
    return None


def _resample_nearest_to_target(
    src_img: nib.spatialimages.SpatialImage, target_img: nib.spatialimages.SpatialImage
) -> np.ndarray:
    try:
        from nibabel.processing import resample_from_to  # type: ignore

        res = resample_from_to(src_img, target_img, order=0)
        return np.asanyarray(res.dataobj)
    except Exception:
        # If resampling is unavailable, return raw data (may misalign if grids differ)
        return np.asanyarray(src_img.dataobj)


def fs_cortex_mask_in_target_space(
    fs_seg_path: Path, target_nii_path: Path
) -> np.ndarray:
    fs_img = nib.load(str(fs_seg_path))
    fs_data = np.asanyarray(fs_img.dataobj)
    cortex = (fs_data == 3) | (fs_data == 42)
    target_img = nib.load(str(target_nii_path))
    cortex_rs = _resample_nearest_to_target(
        nib.Nifti1Image(cortex.astype(np.uint8), fs_img.affine), target_img
    )
    return (cortex_rs > 0).astype(np.uint8)


@dataclass(frozen=True)
class Visit:
    subject: str
    visit_code: str
    acq_date: str
    dt: datetime


def read_visits(manifest_csv: str) -> list[Visit]:
    with open(manifest_csv, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise ValueError("Manifest CSV missing header")
        required = {"subject", "visit_code", "acq_date"}
        missing = required - set(r.fieldnames)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        out: list[Visit] = []
        for row in r:
            subject = (row.get("subject") or "").strip()
            visit_code = (row.get("visit_code") or "").strip()
            acq_date_raw = (row.get("acq_date") or "").strip()
            dt = _parse_date(acq_date_raw)
            if not subject or not visit_code or not acq_date_raw or dt is None:
                continue
            # Canonicalize to match local folder naming.
            acq_date = dt.strftime("%Y-%m-%d")
            out.append(
                Visit(subject=subject, visit_code=visit_code, acq_date=acq_date, dt=dt)
            )

    out.sort(key=lambda x: (x.subject, x.dt))
    return out


def is_done(out_dir: Path) -> bool:
    return (out_dir / "bsc_dir_map.nii.gz").exists() and (
        out_dir / "bsc_metrics.json"
    ).exists()


def run_one(
    image_id: str,
    preproc_root: Path,
    out_root: Path,
    work_root: Path,
    eps: float,
    sigma_mm: float,
    write_mask: bool,
    fs_subjects_dir: Optional[Path],
    fs_cortex: bool,
    missing_input: str,
) -> bool:
    candidates = [
        preproc_root / image_id / "t1w_preproc.nii.gz",
    ]
    t1 = next((p for p in candidates if p.exists()), None)
    if t1 is None:
        msg = f"Missing preproc T1 for {image_id}. Tried:\n" + "\n".join(
            [f"  - {p}" for p in candidates]
        )
        if missing_input == "skip":
            print("[SKIP] " + msg)
            return False
        raise FileNotFoundError(msg)

    out_dir = out_root / image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_work = Path(tempfile.mkdtemp(dir=str(work_root)))
    # Force temp usage into per-scan folder to avoid system temp blowups and
    # to keep work_root empty after each scan.
    prev_tmpdir = os.environ.get("TMPDIR")
    prev_tmp = os.environ.get("TMP")
    prev_temp = os.environ.get("TEMP")
    prev_tempfile_tempdir = getattr(tempfile, "tempdir", None)
    os.environ["TMPDIR"] = str(scan_work)
    os.environ["TMP"] = str(scan_work)
    os.environ["TEMP"] = str(scan_work)
    tempfile.tempdir = str(scan_work)
    try:
        bsc_dir_p = out_dir / "bsc_dir_map.nii.gz"
        bsc_mag_p = out_dir / "bsc_mag_map.nii.gz"
        gm_p = out_dir / "gm_prob.nii.gz"
        wm_p = out_dir / "wm_prob.nii.gz"
        brain_p = out_dir / "brain_mask.nii.gz"

        # Retry once if outputs are corrupted (e.g., truncated .nii.gz -> EOFError)
        for attempt in (1, 2):
            run_atropos_bsc(
                t1_path=str(t1),
                out_dir=str(out_dir),
                eps=float(eps),
                sigma_mm=float(sigma_mm),
                work_dir=str(scan_work),
            )
            try:
                gm, _, _ = _load_nii(gm_p)
                wm, _, _ = _load_nii(wm_p)
                brain, _, _ = _load_nii(brain_p)
                break
            except Exception as e:
                if attempt == 1:
                    print(
                        f"[WARN] Corrupt/incomplete outputs for {image_id}; retrying once. Error: {e}"
                    )
                    shutil.rmtree(out_dir, ignore_errors=True)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    continue
                raise

        interface = build_interface_mask(gm, wm, brain)

        if fs_cortex and fs_subjects_dir is not None:
            fs_seg = load_fs_seg(fs_subjects_dir, image_id)
            if fs_seg is not None:
                cortex = fs_cortex_mask_in_target_space(fs_seg, bsc_dir_p)
                interface = ((interface > 0) & (cortex > 0)).astype(np.uint8)

        bdir, aff_b, hdr_b = _load_nii(bsc_dir_p)
        bdir = bdir.astype(np.float32)
        bdir[interface == 0] = 0
        _save_like(bsc_dir_p, bdir, aff_b, hdr_b)

        if bsc_mag_p.exists():
            bmag, aff_m, hdr_m = _load_nii(bsc_mag_p)
            bmag = bmag.astype(np.float32)
            bmag[interface == 0] = 0
            _save_like(bsc_mag_p, bmag, aff_m, hdr_m)

        if write_mask:
            _save_like(
                out_dir / "boundary_band_mask.nii.gz",
                interface.astype(np.uint8),
                aff_b,
                hdr_b,
            )
    finally:
        # Restore prior temp settings
        if prev_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = prev_tmpdir
        if prev_tmp is None:
            os.environ.pop("TMP", None)
        else:
            os.environ["TMP"] = prev_tmp
        if prev_temp is None:
            os.environ.pop("TEMP", None)
        else:
            os.environ["TEMP"] = prev_temp
        tempfile.tempdir = prev_tempfile_tempdir
        shutil.rmtree(scan_work, ignore_errors=True)

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Local manifest CSV")
    ap.add_argument(
        "--preproc_root",
        required=True,
        help="Local preproc root containing <image_id>/t1w_preproc.nii.gz",
    )
    ap.add_argument(
        "--plan_from",
        choices=("manifest", "preproc"),
        default="manifest",
        help="Plan scans from the manifest or from folders under preproc_root.",
    )
    ap.add_argument("--out_root", required=True, help="Local output root for BSC v2")
    ap.add_argument(
        "--work_root", default="/Volumes/YAAGL/tmp/bsc_v2", help="Local scratch root"
    )
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    ap.add_argument("--min_visits_per_subject", type=int, default=4)
    ap.add_argument("--limit_scans", type=int, default=0)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--write_mask", action="store_true")
    ap.add_argument(
        "--missing_input",
        choices=("skip", "error"),
        default="skip",
        help="What to do when a scan's preproc T1 is missing.",
    )
    ap.add_argument(
        "--fs_subjects_dir",
        default="",
        help="Local FreeSurfer SUBJECTS_DIR (per scan image_id)",
    )
    ap.add_argument(
        "--fs_cortex",
        action="store_true",
        help="Intersect interface mask with FS cortex GM (labels 3/42)",
    )
    args = ap.parse_args()

    preproc_root = Path(args.preproc_root)
    out_root = Path(args.out_root)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    fs_subjects_dir: Optional[Path] = (
        Path(args.fs_subjects_dir) if args.fs_subjects_dir else None
    )

    plan: list[str] = []
    if args.plan_from == "preproc":
        # Process exactly what's present locally.
        for p in sorted(preproc_root.iterdir()):
            if p.is_dir():
                plan.append(p.name)
        # Apply >= min visits filter based on folder names.
        counts: dict[str, int] = {}
        for image_id in plan:
            subject, _, _ = _parse_image_id(image_id)
            counts[subject] = counts.get(subject, 0) + 1
        plan = [
            image_id
            for image_id in plan
            if counts.get(_parse_image_id(image_id)[0], 0)
            >= int(args.min_visits_per_subject)
        ]
    else:
        visits = read_visits(args.manifest)
        counts: dict[str, int] = {}
        for v in visits:
            counts[v.subject] = counts.get(v.subject, 0) + 1
        for v in visits:
            if counts.get(v.subject, 0) < int(args.min_visits_per_subject):
                continue
            plan.append(_image_id(v.subject, v.visit_code, v.acq_date))

    if int(args.limit_scans) > 0:
        plan = plan[: int(args.limit_scans)]

    total = len(plan)
    print("[INFO] scans planned:", total)
    print("[INFO] preproc_root:", str(preproc_root))
    print("[INFO] out_root:", str(out_root))
    print("[INFO] fs_cortex:", bool(args.fs_cortex))
    print("[INFO] plan_from:", str(args.plan_from))

    ran = 0
    skipped_missing = 0
    skipped_error = 0
    for i, image_id in enumerate(plan, start=1):
        out_dir = out_root / image_id
        if args.skip_done and is_done(out_dir):
            continue
        print(f"[RUN] {i}/{total} {image_id}")
        try:
            did_run = run_one(
                image_id=image_id,
                preproc_root=preproc_root,
                out_root=out_root,
                work_root=work_root,
                eps=float(args.eps),
                sigma_mm=float(args.sigma_mm),
                write_mask=bool(args.write_mask),
                fs_subjects_dir=fs_subjects_dir,
                fs_cortex=bool(args.fs_cortex),
                missing_input=str(args.missing_input),
            )
        except Exception as e:
            skipped_error += 1
            print(f"[ERROR] Failed scan {image_id}: {e}")
            continue
        if did_run:
            ran += 1
        else:
            skipped_missing += 1
    print("[DONE] ran:", ran)
    print("[DONE] skipped_missing:", skipped_missing)
    print("[DONE] skipped_error:", skipped_error)


if __name__ == "__main__":
    main()
