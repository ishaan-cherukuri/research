"""Measure how much of the cortical surface the stored boundary band actually reaches.

The volumetric pipeline defines gBSC only inside the atropos boundary band, the
voxels whose gray-matter probability lies within eps of one half, and zeroes it
everywhere else. Whether that band can be sampled on a surface is an empirical
question, and this answers it: for every scan, the fraction of cortex vertices
that receive a non-zero value when the stored map is projected onto the white
surface, unsmoothed.

Run this against the archived v1 sampling (surf_v1_band_projfrac05), whose
per-scan lh.gbsc.native.mgh files are exactly that projection. Smoothing would
spread isolated hits and overstate coverage, so the native maps are used and
the count is taken before any surface smoothing.

Also writes the two example maps panel B needs, for one representative scan:
the band's footprint and the recomputed dense field, both sampled at the same
surface and resampled to fsaverage6 without smoothing.

Usage (Trillium login node):
    module load freesurfer/7.4.1 && source $EBROOTFREESURFER/FreeSurferEnv.sh
    python3 band_surface_coverage.py --work $SCRATCH/adni_trillium/derivatives/vertexwise \
        --scans cohort/vertex_scans.csv --out_dir results
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_label

TRG = "fsaverage6"


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--v1_dir", default="surf_v1_band_projfrac05",
                    help="archived band-sampled run holding ?h.gbsc.native.mgh")
    ap.add_argument("--example", default=None,
                    help="image_id for panel B; defaults to a median-coverage scan")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    work = Path(args.work)
    v1 = work / args.v1_dir
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scans = pd.read_csv(args.scans)
    fs_home = Path(os.environ["FREESURFER_HOME"])
    subjects_dir = work / "freesurfer"

    # ---- panel A: per-scan coverage on each scan's own surface -------------
    rows = []
    for iid in scans["image_id"]:
        vals, ok = [], True
        for hemi in ("lh", "rh"):
            p = v1 / iid / f"{hemi}.gbsc.native.mgh"
            ctx_f = subjects_dir / iid / "label" / f"{hemi}.cortex.label"
            if not (p.exists() and ctx_f.exists()):
                ok = False
                break
            v = load(p)[read_label(str(ctx_f))]
            vals.append(float((v != 0).mean()))
        if ok:
            rows.append({"image_id": iid, "coverage": float(np.mean(vals))})
    cov = pd.DataFrame(rows)
    cov.to_csv(out / "band_coverage.csv", index=False)
    print(f"coverage for {len(cov)} scans -> {out/'band_coverage.csv'}")
    print(cov["coverage"].describe(percentiles=[.25, .5, .75]).round(4).to_string())
    print(f"scans sampling nothing at all: {int((cov['coverage'] <= 0.005).sum())}")

    # ---- panel B: one representative scan, band footprint vs dense field ---
    example = args.example
    if example is None:
        good = cov[cov["coverage"] > 0.005]
        example = good.iloc[(good["coverage"]
                             - good["coverage"].median()).abs().argsort().iloc[0]]["image_id"]
    print(f"example scan for panel B: {example}")

    bsc_dir = scans.loc[scans["image_id"] == example, "bsc_dir"].iloc[0]
    tmp = out / "_tmp_example"
    tmp.mkdir(exist_ok=True)
    field_nii = work / "field" / f"{example}.nii.gz"
    env = dict(os.environ, SUBJECTS_DIR=str(subjects_dir))
    # fsaverage6 must be visible inside SUBJECTS_DIR for --trgsubject
    if not (subjects_dir / TRG).exists():
        (subjects_dir / TRG).symlink_to(fs_home / "subjects" / TRG)

    maps = {}
    for name, mov, extra in (
        ("band", f"{bsc_dir}/bsc_dir_map.nii.gz", ["--projfrac", "0.5"]),
        ("field", str(field_nii), ["--projdist-avg", "0", "1.5", "0.5"]),
    ):
        native = tmp / f"lh.{name}.native.mgh"
        subprocess.run(["mri_vol2surf", "--mov", mov, "--regheader", example,
                        "--hemi", "lh", "--surf", "white", "--interp", "trilinear",
                        *extra, "--o", str(native)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        trg = tmp / f"lh.{name}.{TRG}.mgh"
        subprocess.run(["mri_surf2surf", "--srcsubject", example, "--trgsubject", TRG,
                        "--hemi", "lh", "--sval", str(native), "--tval", str(trg)],
                       check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        maps[name] = load(trg).astype(np.float32)

    np.savez_compressed(out / "band_example.npz", example_id=example, **maps)
    print(f"example maps -> {out/'band_example.npz'}")


if __name__ == "__main__":
    main()
