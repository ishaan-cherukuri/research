"""Per-subject annualized slope at every fsaverage6 vertex.

OLS fit of each measure against years since the subject's first window scan,
the same time axis the ROI slopes use.

gBSC is divided by the scan's own cortex-wide mean first (--normalize global).
The global level of the field moves with scanner within a subject, ten per cent
at the median and up to twofold, so a raw vertex slope is mostly a slope of
that level rather than of anything spatial. --normalize none for a sensitivity
run; the cos variant does not need it. Log-valued measures such as the sigmoid
growth rate take --normalize demean, the additive analogue.

Also writes slope_precision.csv, the per-subject precision of the slope
estimate. A slope's variance is sigma^2 / Sxx: Sxx is exactly known from the
scan times, and sigma is the residual scale, pooled here over every cortex
vertex and both hemispheres so it carries tens of thousands of residuals
rather than the n-2 a per-vertex estimate would give. Both parts are needed:
subjects differ threefold in scan count and span, and a four-scan subject over
one year carries far less information about a slope than a nine-scan subject
over six. make_design.py turns this into regression weights.

A scan is dropped if it lacks any of the six maps or fails QC: more than
--max_zero_frac of cortex at exactly zero, or a cortex-wide mean below
--min_global. The volumetric segmentation returned an empty field for thirteen
scans, and those produce enormous slopes or divide by zero once normalized.

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

HEMIS = ("lh", "rh")
TRG = "fsaverage6"


def load_map(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).reshape(-1)


def ols_slope(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Slope of Y (n_scans x n_vertices) on t (n_scans,), per column."""
    tc = t - t.mean()
    return (tc @ (Y - Y.mean(axis=0))) / (tc @ tc)


def pooled_resid_sd(t: np.ndarray, Y: np.ndarray, slope: np.ndarray) -> float:
    """Residual scale of the per-vertex straight-line fits, pooled over vertices."""
    tc = t - t.mean()
    fit = Y.mean(axis=0) + tc[:, None] * slope
    rss = float(((Y - fit) ** 2).sum())
    dof = (len(t) - 2) * Y.shape[1]
    return float(np.sqrt(rss / dof)) if dof > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", required=True)
    ap.add_argument("--surf_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--min_scans", type=int, default=4)
    ap.add_argument("--measure", default="gbsc",
                    help="dependent measure; its maps are read as "
                         "{hemi}.{measure}.fsaverage6.mgh under --measure_root")
    ap.add_argument("--measure_root", default=None,
                    help="where the dependent measure's maps live; defaults to --surf_root")
    ap.add_argument("--normalize", choices=["global", "demean", "none"], default="global")
    ap.add_argument("--qc", choices=["gbsc", "finite", "none"], default="gbsc",
                    help="gbsc applies the zero-fraction and minimum-level checks "
                         "below; finite only requires the map to be finite, which is "
                         "what a log-valued measure needs")
    ap.add_argument("--max_zero_frac", type=float, default=0.05)
    ap.add_argument("--min_global", type=float, default=0.02)
    ap.add_argument("--fs_subjects_dir", default=os.environ.get("SUBJECTS_DIR", ""))
    args = ap.parse_args()
    from nibabel.freesurfer import read_label
    cortex = {hemi: read_label(str(Path(args.fs_subjects_dir) / TRG / "label"
                                   / f"{hemi}.cortex.label")) for hemi in HEMIS}

    scans = pd.read_csv(args.scans).sort_values(["subject", "years"])
    surf_root, out_root = Path(args.surf_root), Path(args.out_root)
    meas_root = Path(args.measure_root) if args.measure_root else surf_root
    y_meas = args.measure
    measures = (y_meas, "thickness", "curv")

    def y_map(iid: str, hemi: str) -> np.ndarray:
        return load_map(meas_root / iid / f"{hemi}.{y_meas}.{TRG}.mgh")
    out_root.mkdir(parents=True, exist_ok=True)

    written, missing, dropped, qc, prec = [], [], [], [], []
    for subj, g in scans.groupby("subject"):
        keep = []
        for iid in g["image_id"]:
            if not ((surf_root / iid / f"rh.curv.{TRG}.mgh").exists()
                    and (meas_root / iid / f"rh.{y_meas}.{TRG}.mgh").exists()):
                dropped.append({"subject": subj, "image_id": iid, "reason": "no_surface_maps"})
                keep.append(False)
                continue
            vals = [y_map(iid, h)[cortex[h]] for h in HEMIS]
            zero_frac = max(float((v == 0).mean()) for v in vals)
            nonfinite = max(float((~np.isfinite(v)).mean()) for v in vals)
            gmean = float(np.mean([np.nanmean(v) for v in vals]))
            qc.append({"subject": subj, "image_id": iid, "zero_frac": zero_frac,
                       "nonfinite_frac": nonfinite, "global_mean": gmean})
            if args.qc == "gbsc":
                reject = zero_frac > args.max_zero_frac or gmean < args.min_global
            elif args.qc == "finite":
                reject = nonfinite > args.max_zero_frac or not np.isfinite(gmean)
            else:
                reject = False
            if reject:
                dropped.append({"subject": subj, "image_id": iid,
                                "reason": f"qc zero_frac={zero_frac:.2f} "
                                          f"nonfinite={nonfinite:.2f} global={gmean:.3f}"})
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
        level = np.ones(len(g))
        if args.normalize != "none":
            level = np.array([np.mean([np.nanmean(y_map(iid, h)[cortex[h]]) for h in HEMIS])
                              for iid in g["image_id"]])
        rss_parts = []
        for hemi in HEMIS:
            for meas in measures:
                root = meas_root if meas == y_meas else surf_root
                Y = np.stack([load_map(root / iid / f"{hemi}.{meas}.{TRG}.mgh")
                              for iid in g["image_id"]])
                if meas == y_meas:
                    Y = Y / level[:, None] if args.normalize == "global" else Y - level[:, None]
                slope = ols_slope(t, Y)
                if meas == y_meas:
                    rss_parts.append(pooled_resid_sd(t, Y[:, cortex[hemi]],
                                                     slope[cortex[hemi]]))
                img = nib.MGHImage(slope.astype(np.float32).reshape(-1, 1, 1), np.eye(4))
                nib.save(img, str(sdir / f"{hemi}.{meas}_slope.mgh"))
        sxx = float(((t - t.mean()) ** 2).sum())
        resid_sd = float(np.sqrt(np.mean(np.square(rss_parts))))
        prec.append({"subject": subj, "n_scans": len(g), "span_years": float(t[-1]),
                     "sxx": sxx, "resid_sd": resid_sd,
                     "slope_precision": sxx / resid_sd ** 2 if resid_sd > 0 else np.nan})
        written.append({"subject": subj, "n_scans": len(g), "n_window": n_window,
                        "span_years": float(t[-1])})

    pd.DataFrame(written).to_csv(out_root / "subjects_done.csv", index=False)
    pd.DataFrame(missing, columns=["subject", "n_window", "n_have", "missing"]).to_csv(
        out_root / "missing.csv", index=False)
    pd.DataFrame(dropped, columns=["subject", "image_id", "reason"]).to_csv(
        out_root / "dropped_scans.csv", index=False)
    pd.DataFrame(qc).to_csv(out_root / "scan_qc.csv", index=False)
    pr = pd.DataFrame(prec)
    pr.to_csv(out_root / "slope_precision.csv", index=False)
    print(f"slopes written for {len(written)} subjects; {len(dropped)} scans dropped; "
          f"{len(missing)} subjects below {args.min_scans} scans")
    if len(pr):
        q = pr["slope_precision"].quantile([.05, .5, .95]).to_numpy()
        print(f"slope precision 5th/median/95th: {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g} "
              f"(ratio {q[2] / q[0]:.1f}x)")


if __name__ == "__main__":
    main()
