"""Per-subject annualized slope at every fsaverage6 vertex.

For each subject, each hemisphere and each measure (gbsc, thickness, curv),
stacks the sampled per-scan maps in date order and fits an ordinary
least-squares line against time in years since the subject's first window
scan, the same time axis the paper's ROI slopes use. Writes one .mgh per
subject/hemisphere/measure holding the slope, plus a table of which scans
went in.

gBSC is divided by the scan's own cortex-wide mean (both hemispheres) before
the slope is fitted (--normalize global, the default). The global level of the
field moves with scanner and field strength within a subject, by ten per cent
at the median and up to twofold, and a raw vertex slope is then mostly a slope
of that level. The normalized map is the spatial pattern of sharpness relative
to the rest of the cortex, which is the quantity the vertex-wise question is
about. --normalize none keeps the raw field for a sensitivity run.

A scan is dropped if it lacks any of the six maps (a failed recon-all) or if
its gBSC field failed quality control: more than --max_zero_frac of cortex
vertices at exactly zero, or a cortex-wide mean below --min_global (the
volumetric segmentation returned an empty field for a few scans, and such a
scan produces an enormous slope, or a division by zero once normalized). The
subject is kept if at least --min_scans remain; dropped scans go to
slopes/dropped_scans.csv and subjects that fall below the minimum to
slopes/missing.csv. Runs on a login node in a few minutes; no cluster job
needed.

Usage:
    python3 compute_vertex_slopes.py --scans cohort/vertex_scans.csv \
        --surf_root $SCRATCH/.../vertexwise/surf --out_root $SCRATCH/.../vertexwise/slopes
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

MEASURES = ("gbsc", "thickness", "curv")
HEMIS = ("lh", "rh")
TRG = "fsaverage6"


def load_map(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).reshape(-1)


def ols_slope(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slope of Y (n_scans x n_vertices) on t (n_scans,), per column."""
    tc = t - t.mean()
    return (tc @ (Y - Y.mean(axis=0))) / (tc @ tc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True)
    ap.add_argument("--surf_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--min_scans", type=int, default=4)
    ap.add_argument("--normalize", choices=["global", "none"], default="global")
    ap.add_argument("--max_zero_frac", type=float, default=0.05)
    ap.add_argument("--min_global", type=float, default=0.02)
    ap.add_argument("--fs_subjects_dir", default=os.environ.get("SUBJECTS_DIR", ""))
    args = ap.parse_args()
    from nibabel.freesurfer import read_label
    cortex = {hemi: read_label(str(Path(args.fs_subjects_dir) / TRG / "label"
                                   / f"{hemi}.cortex.label")) for hemi in HEMIS}

    scans = pd.read_csv(args.scans).sort_values(["subject", "years"])
    surf_root, out_root = Path(args.surf_root), Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written, missing, dropped, qc = [], [], [], []
    for subj, g in scans.groupby("subject"):
        keep = []
        for iid in g["image_id"]:
            if not (surf_root / iid / f"rh.curv.{TRG}.mgh").exists():
                dropped.append({"subject": subj, "image_id": iid, "reason": "no_surface_maps"})
                keep.append(False)
                continue
            vals = [load_map(surf_root / iid / f"{h}.gbsc.{TRG}.mgh")[cortex[h]] for h in HEMIS]
            zero_frac = max(float((v == 0).mean()) for v in vals)
            gmean = float(np.mean([v.mean() for v in vals]))
            qc.append({"subject": subj, "image_id": iid, "zero_frac": zero_frac, "global_mean": gmean})
            if zero_frac > args.max_zero_frac or gmean < args.min_global:
                dropped.append({"subject": subj, "image_id": iid,
                                "reason": f"qc zero_frac={zero_frac:.2f} global={gmean:.3f}"})
                keep.append(False)
            else:
                keep.append(True)
        have = np.array(keep, dtype=bool)
        n_window = len(g)
        g = g[have]
        if len(g) < args.min_scans:
            missing.append({"subject": subj, "n_window": n_window, "n_have": len(g),
                            "missing": ";".join(dropped_i["image_id"] for dropped_i in dropped
                                                if dropped_i["subject"] == subj)})
            continue
        # slopes are relative to the first retained scan
        t = g["years"].to_numpy(dtype=np.float64)
        t = t - t[0]
        sdir = out_root / subj
        sdir.mkdir(exist_ok=True)
        scale = np.ones(len(g))
        if args.normalize == "global":
            scale = np.array([np.mean([load_map(surf_root / iid / f"{h}.gbsc.{TRG}.mgh")[cortex[h]].mean()
                                       for h in HEMIS]) for iid in g["image_id"]])
        for hemi in HEMIS:
            for meas in MEASURES:
                Y = np.stack([load_map(surf_root / iid / f"{hemi}.{meas}.{TRG}.mgh")
                              for iid in g["image_id"]])
                if meas == "gbsc":
                    Y = Y / scale[:, None]
                slope = ols_slope(t, Y).astype(np.float32)
                img = nib.MGHImage(slope.reshape(-1, 1, 1), np.eye(4))
                nib.save(img, str(sdir / f"{hemi}.{meas}_slope.mgh"))
        written.append({"subject": subj, "n_scans": len(g), "n_window": n_window,
                        "span_years": float(t[-1])})

    pd.DataFrame(written).to_csv(out_root / "subjects_done.csv", index=False)
    pd.DataFrame(missing, columns=["subject", "n_window", "n_have", "missing"]).to_csv(
        out_root / "missing.csv", index=False)
    pd.DataFrame(dropped, columns=["subject", "image_id", "reason"]).to_csv(
        out_root / "dropped_scans.csv", index=False)
    pd.DataFrame(qc).to_csv(out_root / "scan_qc.csv", index=False)
    print(f"slopes written for {len(written)} subjects; {len(dropped)} scans dropped; "
          f"{len(missing)} subjects below {args.min_scans} scans")


if __name__ == "__main__":
    main()
