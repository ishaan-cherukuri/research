"""Stack per-subject slope maps and write the mri_glmfit design for Models A and B.

Model A: gBSC slope ~ converter + non-converter + age + sex + APOE4 + 3T.
Model B: Model A plus, at the same vertex, cortical thickness slope and mean
         curvature slope entered as per-vertex regressors (mri_glmfit --pvr).
The contrast in both is converter minus non-converter. Covariates are centred
so that the group columns carry the adjusted group means.

Writes into --out_dir:
  {hemi}.gbsc_slope.4d.mgh, {hemi}.thickness_slope.4d.mgh, {hemi}.curv_slope.4d.mgh
  X.mat, contrast_A.mtx, contrast_B.mtx, subjects_used.csv, design_columns.txt

Usage:
    python3 make_design.py --subjects cohort/vertex_subjects.csv --scans cohort/vertex_scans.csv \
        --slopes_root .../slopes --out_dir .../glm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

MEASURES = ("gbsc", "thickness", "curv")
HEMIS = ("lh", "rh")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--slopes_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--drop_slope_outlier_pct", type=float, default=0.0,
                    help="drop subjects whose cortex-mean |gBSC slope| exceeds this "
                         "percentile; outcome-blind, for measures whose slope "
                         "variance is dominated by a few unstable scans")
    args = ap.parse_args()

    subj = pd.read_csv(args.subjects)
    scans = pd.read_csv(args.scans)
    slopes_root, out = Path(args.slopes_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    done = pd.read_csv(slopes_root / "subjects_done.csv")["subject"]
    subj = subj[subj["subject"].isin(done)].sort_values("subject").reset_index(drop=True)

    # Field strength as a 3 T indicator; where the cohort table has no baseline
    # value, fall back to the IDA field strength of the subject's first scan,
    # and failing that to the acq-15T / acq-3T tag in the BIDS filename.
    first = scans.sort_values(["subject", "years"]).groupby("subject").first()
    from_path = first["t1_path"].str.extract(r"acq-(\d+)T")[0].map(
        lambda v: np.nan if pd.isna(v) else (1.5 if v == "15" else float(v)))
    fs = (subj["field_strength_bl"]
          .fillna(subj["subject"].map(first["field_strength_ida"]))
          .fillna(subj["subject"].map(from_path)))
    if fs.isna().any():
        raise SystemExit(f"no field strength for {subj.loc[fs.isna(), 'subject'].tolist()}")
    subj["field_3t"] = (fs >= 2.5).astype(float)

    conv = subj["event"].to_numpy(dtype=float)
    cov = subj[["age_at_landmark", "female", "apoe4", "field_3t"]].to_numpy(dtype=float)
    cov = cov - cov.mean(axis=0)
    X = np.column_stack([conv, 1.0 - conv, cov])
    cols = ["converter", "nonconverter", "age_c", "female_c", "apoe4_c", "field_3t_c"]
    np.savetxt(out / "X.mat", X, fmt="%.8f")
    (out / "design_columns.txt").write_text("\n".join(cols) + "\n")

    # Contrast rows must span the global columns plus one column per PVR for B.
    cA = np.zeros((1, X.shape[1])); cA[0, 0], cA[0, 1] = 1.0, -1.0
    cB = np.zeros((1, X.shape[1] + 2)); cB[0, 0], cB[0, 1] = 1.0, -1.0
    np.savetxt(out / "contrast_A.mtx", cA, fmt="%.1f")
    np.savetxt(out / "contrast_B.mtx", cB, fmt="%.1f")

    if args.drop_slope_outlier_pct > 0:
        # Ranked on the magnitude of each subject's own slope map, which never
        # looks at converter status, so the exclusion cannot bias the contrast.
        mag = np.array([np.mean([np.abs(np.asarray(nib.load(
            str(slopes_root / s / f"{h}.gbsc_slope.mgh")).dataobj)).mean()
            for h in HEMIS]) for s in subj["subject"]])
        cut = np.percentile(mag, args.drop_slope_outlier_pct)
        keep = mag <= cut
        dropped = subj.loc[~keep, "subject"].tolist()
        print(f"slope-outlier QC: dropped {len(dropped)} subjects above the "
              f"{args.drop_slope_outlier_pct:g}th percentile "
              f"(cut {cut:.4g}): {dropped}")
        subj = subj[keep].reset_index(drop=True)
        conv = subj["event"].to_numpy(dtype=float)
        cov = subj[["age_at_landmark", "female", "apoe4", "field_3t"]].to_numpy(dtype=float)
        cov = cov - cov.mean(axis=0)
        X = np.column_stack([conv, 1.0 - conv, cov])
        np.savetxt(out / "X.mat", X, fmt="%.8f")

    for hemi in HEMIS:
        for meas in MEASURES:
            maps = [np.asarray(nib.load(str(slopes_root / s / f"{hemi}.{meas}_slope.mgh")).dataobj,
                               dtype=np.float32).reshape(-1) for s in subj["subject"]]
            data = np.stack(maps, axis=-1)[:, None, None, :]     # nv x 1 x 1 x nsubj
            nib.save(nib.MGHImage(data, np.eye(4)), str(out / f"{hemi}.{meas}_slope.4d.mgh"))

    subj.to_csv(out / "subjects_used.csv", index=False)
    print(f"design: {len(subj)} subjects ({int(conv.sum())} converters), "
          f"X {X.shape}, stacked maps in {out}")


if __name__ == "__main__":
    main()
