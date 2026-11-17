"""Turn per-vertex sigmoid fits into the per-scan feature CSVs the survival
pipeline already reads.

Takes the native-surface outputs of fs_sigmoid_bsc.py, resamples them to
fsaverage6, and aggregates by the FreeSurfer aparc parcellation. Two CSVs come
out, mirroring the schemas of bsc_simple_features_merged.csv and
per_scan_regional.csv so build_spec_cohort.py consumes them with no change.

Three measures travel together, all from the same fit:

    bscsig_*      log growth rate under the published box constraints
    bscsigfree_*  log growth rate with `a` unbounded below
    bscsigratio_* white/gray tissue intensity ratio

The published estimator bounds `a >= min(y)`, which asserts the sigmoid reaches
its gray-side asymptote inside the sampled window. On this cohort that bound
binds for most of the cortex, compressing the measure against a floor near
0.059, so the unbounded variant is carried alongside rather than as an
afterthought. Which of the two is the better instrument is an empirical
question this pipeline exists to answer.

Resampling is a normalized convolution, not a plain smooth, and it has to be.
The native fits are about 93 per cent finite: a sigmoid does not converge
everywhere, and the failures are scattered. mri_surf2surf propagates NaN
through its kernel, so smoothing those maps at 20 mm FWHM turned 7 per cent
missing into 66 per cent missing, and every ROI mean was then an average over
whichever third of the parcel sat furthest from a failed fit. That is a
systematic subsample, not a random one. So the valid data and a validity mask
are smoothed with the same kernel and divided, which is the standard fix and
keeps the kernel identical to what the rest of the pipeline uses.

The prefix is bscsigratio_ rather than the companion pipeline's bscratio_,
because bscratio_ is already used in this project for the scale-invariant
cos(theta) gradient variant and the two must not collide in one cohort table.

Usage:
    python3 fs_sigmoid_features.py --native_root .../sigmoid/native \
        --subjects_dir .../freesurfer --scans cohort/vertex_scans.csv \
        --fs_subjects_dir $FREESURFER_HOME/subjects --out_dir from_cluster/
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot

TRG = "fsaverage6"
HEMIS = ("lh", "rh")
# model_ratio is an intensity ratio, not a rate, so it is not log-transformed
# and not smoothed with the same kernel the reference applies to log c.
MEASURES = {"model_c_log": "bscsig", "model_c_free_log": "bscsigfree",
            "model_ratio": "bscsigratio"}

# The seven bilateral regions pre-specified as the AD signature. FreeSurfer's
# aparc carries entorhinal, which the CIVET atlas does not, so this composite is
# the full seven rather than the six the companion pipeline can manage.
AD_SIGNATURE = ("entorhinal", "parahippocampal", "inferiortemporal",
                "middletemporal", "precuneus", "posteriorcingulate", "fusiform")


MIN_KERNEL_WEIGHT = 0.5    # a target vertex needs this much valid support to keep


def _surf2surf(src: Path, dst: Path, image_id: str, hemi: str, fwhm: float, env) -> None:
    cmd = ["mri_surf2surf", "--srcsubject", image_id, "--trgsubject", TRG,
           "--hemi", hemi, "--sval", str(src), "--tval", str(dst)]
    if fwhm > 0:
        cmd += ["--fwhm-trg", str(fwhm)]
    subprocess.run(cmd, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _save_like(ref: nib.freesurfer.mghformat.MGHImage, vals: np.ndarray, dst: Path) -> None:
    nib.save(nib.MGHImage(vals.astype(np.float32).reshape(-1, 1, 1), ref.affine), str(dst))


def resample(native: Path, image_id: str, hemi: str, meas: str, out: Path,
             fwhm: float, env) -> Path:
    """native surface -> fsaverage6 by normalized convolution.

    The valid values and an indicator of where they are get the same kernel;
    their ratio is the smoothed estimate, kept only where enough of the kernel
    landed on real data. Plain smoothing would spread every NaN instead.
    """
    dst = out / f"{hemi}.{meas}.{TRG}.mgh"
    if dst.exists():
        return dst
    img = nib.load(str(native / f"{hemi}.{meas}.mgh"))
    v = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
    valid = np.isfinite(v)
    num_n, den_n = out / f"_num.{hemi}.mgh", out / f"_den.{hemi}.mgh"
    _save_like(img, np.where(valid, v, 0.0), num_n)
    _save_like(img, valid.astype(np.float64), den_n)
    num_t, den_t = out / f"_numt.{hemi}.mgh", out / f"_dent.{hemi}.mgh"
    _surf2surf(num_n, num_t, image_id, hemi, fwhm, env)
    _surf2surf(den_n, den_t, image_id, hemi, fwhm, env)
    num = np.asarray(nib.load(str(num_t)).dataobj, dtype=np.float64).reshape(-1)
    den = np.asarray(nib.load(str(den_t)).dataobj, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        outv = np.where(den >= MIN_KERNEL_WEIGHT, num / den, np.nan)
    ref = nib.load(str(num_t))
    nib.save(nib.MGHImage(outv.astype(np.float32).reshape(-1, 1, 1), ref.affine), str(dst))
    for f in (num_n, den_n, num_t, den_t):
        f.unlink(missing_ok=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native_root", required=True)
    ap.add_argument("--subjects_dir", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--fs_subjects_dir", required=True)
    ap.add_argument("--work", required=True, help="where resampled maps are cached")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fwhm", type=float, default=20.0,
                    help="target-surface smoothing for the log-rate maps, as in the reference")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    native_root = Path(args.native_root)
    work = Path(args.work)
    fsd = Path(args.fs_subjects_dir)
    scans = pd.read_csv(args.scans)
    stop = None if args.limit is None else args.skip + args.limit
    scans = scans.iloc[args.skip:stop]

    env = dict(os.environ, SUBJECTS_DIR=str(args.subjects_dir))
    link = Path(args.subjects_dir) / TRG
    if not link.exists():
        link.symlink_to(fsd / TRG)

    ann = {}
    for hemi in HEMIS:
        labels, _, names = read_annot(str(fsd / TRG / "label" / f"{hemi}.aparc.annot"))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        ann[hemi] = (labels, names)

    rows = []
    for _, r in scans.iterrows():
        iid = r["image_id"]
        nat = native_root / iid
        if not (nat / "rh.model_c.mgh").exists():
            continue
        out_s = work / iid
        out_s.mkdir(parents=True, exist_ok=True)
        row = {"subject": r["subject"], "visit_code": iid.split("_")[3] if len(iid.split("_")) > 3 else "",
               "acq_date": r["date"], "image_id": iid}
        try:
            for meas, prefix in MEASURES.items():
                fwhm = args.fwhm if meas.endswith("_log") else 0.0
                vals, labs, nms = {}, {}, None
                for hemi in HEMIS:
                    v = np.asarray(nib.load(str(resample(
                        nat, iid, hemi, meas, out_s, fwhm, env))).dataobj,
                        dtype=np.float64).reshape(-1)
                    vals[hemi] = v
                    labs[hemi], nms = ann[hemi]
                both = np.concatenate([vals["lh"], vals["rh"]])
                good = np.isfinite(both)
                row[f"{prefix}_mean"] = float(both[good].mean())
                row[f"{prefix}_std"] = float(both[good].std())
                row[f"{prefix}_median"] = float(np.median(both[good]))
                for p in (10, 25, 50, 75, 90):
                    row[f"{prefix}_p{p}"] = float(np.percentile(both[good], p))

                num = den = 0.0
                for hemi in HEMIS:
                    v, lab = vals[hemi], labs[hemi]
                    for code, name in enumerate(nms):
                        if name in ("unknown", "corpuscallosum", "???"):
                            continue
                        m = (lab == code) & np.isfinite(v)
                        n = int(m.sum())
                        key = f"{prefix}_roi{hemi}_{name}"
                        row[key] = float(v[m].mean()) if n else np.nan
                        if name in AD_SIGNATURE and n:
                            num += float(v[m].sum()); den += n
                row[f"{prefix}_adsig"] = num / den if den else np.nan
        except subprocess.CalledProcessError:
            print(f"[FAIL-resample] {iid}", flush=True)
            continue
        rows.append(row)
        if len(rows) % 25 == 0:
            print(f"  {len(rows)} scans", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.skip}" if args.skip or args.limit else ""
    df.to_csv(out / f"per_scan_sigmoid{tag}.csv", index=False)
    print(f"wrote {len(df)} scan-rows -> {out}/per_scan_sigmoid{tag}.csv")


if __name__ == "__main__":
    main()
