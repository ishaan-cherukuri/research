"""Smooth and resample per-vertex sigmoid outputs for one scan.

Port of smooth_sigmoid.sh and resample.sh from CoBrALab/BSC. Two stages, in the
order the paper specifies:

  1. depth_potential -smooth 20 on the subject's own mid surface. The paper
     smooths at 20 mm FWHM and repeats its main effect at 10 mm; both are
     produced here so that sensitivity analysis is available without a rerun.
     Nothing is smoothed before the fit.
  2. surface-resample onto the ICBM symmetric average mesh using CIVET's
     surfmap, which is what puts every subject's vertex n at the same anatomical
     location and makes cross-subject comparison meaningful.

The reference hardcodes the ICBM model path to a CIC-local /opt/quarantine
directory; --model_dir supplies it instead.

Usage:
    python3 postprocess_vertex.py --work_root $SCRATCH/bsc_sigmoid \
        --civet_root $SCRATCH/civet_out --civet_id scan000123 --prefix adni \
        --model_dir $CIVET_HOME/models/icbm
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Vertex quantities carried through smoothing and resampling. rmse and
# converged are QC and stay unsmoothed: averaging a convergence flag across
# 20 mm of cortex would turn a diagnostic into a meaningless fraction.
QUANTITIES = ["model_c_log", "model_c", "model_c_free_log", "model_c_free",
              "model_ratio"]

FWHMS = [20, 10]
HEMIS = ["left", "right"]


def run(cmd: list) -> None:
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def smooth(work: Path, cid: str, hemi: str) -> None:
    mid = (work / "surfaces" / "GM_50_surfaces"
           / f"{cid}_GM_50_surface_{hemi}.obj")
    src_dir = work / "sigmoid_fit" / "unsmoothed"
    out_dir = work / "sigmoid_fit" / "smoothed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in QUANTITIES:
        src = src_dir / f"{cid}_{q}_{hemi}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Missing fit output: {src}")
        for fwhm in FWHMS:
            out = out_dir / f"{cid}_{q}_{hemi}_{fwhm}mm.txt"
            if not out.exists():
                run(["depth_potential", "-smooth", fwhm, src, mid, out])


def resample(work: Path, civet_root: Path, cid: str, prefix: str,
             model_dir: Path, hemi: str) -> None:
    model = model_dir / f"icbm_avg_mid_sym_mc_{hemi}.obj"
    if not model.exists():
        raise FileNotFoundError(f"ICBM model surface missing: {model}")
    surfmap = (civet_root / cid / "transforms" / "surfreg"
               / f"{prefix}_{cid}_{hemi}_surfmap.sm")
    if not surfmap.exists():
        raise FileNotFoundError(f"CIVET surfmap missing: {surfmap}")

    mid = (work / "surfaces" / "GM_50_surfaces"
           / f"{cid}_GM_50_surface_{hemi}.obj")
    src_dir = work / "sigmoid_fit" / "smoothed"
    out_dir = work / "sigmoid_fit" / "resampled"
    out_dir.mkdir(parents=True, exist_ok=True)

    for q in QUANTITIES:
        for fwhm in FWHMS:
            src = src_dir / f"{cid}_{q}_{hemi}_{fwhm}mm.txt"
            out = out_dir / f"{cid}_{q}_{hemi}_{fwhm}mm_rsl.txt"
            if not out.exists():
                run(["surface-resample", model, mid, src, surfmap, out])


def is_done(work: Path, cid: str) -> bool:
    out_dir = work / "sigmoid_fit" / "resampled"
    return all(
        (out_dir / f"{cid}_{q}_{hemi}_{fwhm}mm_rsl.txt").exists()
        for q in QUANTITIES for hemi in HEMIS for fwhm in FWHMS
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--civet_root", required=True)
    ap.add_argument("--civet_id", required=True)
    ap.add_argument("--prefix", default="adni")
    ap.add_argument("--model_dir", required=True,
                    help="CIVET models/icbm directory holding "
                         "icbm_avg_mid_sym_mc_{left,right}.obj")
    args = ap.parse_args()

    work = Path(args.work_root)
    cid = args.civet_id
    if is_done(work, cid):
        print(f"[SKIP] {cid} already post-processed")
        return

    for hemi in HEMIS:
        smooth(work, cid, hemi)
        resample(work, Path(args.civet_root), cid, args.prefix,
                 Path(args.model_dir), hemi)

    print(f"[OK] {cid} post-processed")


if __name__ == "__main__":
    main()
