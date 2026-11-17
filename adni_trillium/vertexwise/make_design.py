"""Stack per-subject slope maps and write the design for Models A and B.

Model A is converter against non-converter with the covariates named by
--covariates; Model B adds the thickness and curvature slopes at the same
vertex as per-vertex regressors. Contrast is converter minus non-converter in
both. Covariates are centred so the two group columns carry adjusted means.

--field_subset restricts the sample by scanner field strength. Covarying for
field strength is not the same test: 3 T covers 75 per cent of non-converters
and 50 per cent of converters, so the covariate partly adjusts away the group
itself. Fitting 3 T and 1.5 T separately avoids that, and because the two
strata hold 67 and 66 events they form a replication pair of nearly equal
power rather than a main analysis and a footnote.

--weights turns slope_precision.csv into regression weights. Subjects carry
four to thirteen scans over spans differing several-fold, so an unweighted fit
gives a four-scan subject's slope the same say as a thirteen-scan subject's.

The weights are bounded, and that matters more than it sounds. Inverse-variance
weighting is only optimal when the variances are right, and raw precision here
spans a factor of sixty, which hands the fit to a handful of subjects: on
synthetic data with a realistic spread, unbounded weights took the effective
sample size from 40 to 17. With 133 events in the real cohort, losing more than
half of the effective sample to reclaim some measurement variance is a bad
trade. --max_weight_ratio clips the weights around their median so no subject
counts more than the given multiple of another, which keeps the effective
sample size close to n while still demoting the four-scan-over-one-year
subjects. The effective n is printed so the cost is never invisible.

Clipping alone turned out to be the wrong instrument: with a spread this wide
every subject lands on one of the two bounds and the weight becomes a two-level
factor rather than a precision weight. So the spread is compressed instead, by
raising precision to the power that brings the 5th-to-95th-percentile ratio
down to --max_weight_ratio. That keeps the weights monotone in precision and
saturates nobody; the clip stays only as a backstop for genuine outliers.

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

HEMIS = ("lh", "rh")
COVMAP = {"age": "age_at_landmark", "female": "female",
          "apoe4": "apoe4", "field_3t": "field_3t"}


def field_3t(subj: pd.DataFrame, scans: pd.DataFrame) -> pd.Series:
    """3 T indicator. Where the cohort table has no baseline value, fall back to
    the IDA field strength of the subject's first scan, and failing that to the
    acq-15T / acq-3T tag in the BIDS filename."""
    first = scans.sort_values(["subject", "years"]).groupby("subject").first()
    from_path = first["t1_path"].str.extract(r"acq-(\d+)T")[0].map(
        lambda v: np.nan if pd.isna(v) else (1.5 if v == "15" else float(v)))
    fs = (subj["field_strength_bl"]
          .fillna(subj["subject"].map(first["field_strength_ida"]))
          .fillna(subj["subject"].map(from_path)))
    if fs.isna().any():
        raise SystemExit(f"no field strength for {subj.loc[fs.isna(), 'subject'].tolist()}")
    return (fs >= 2.5).astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--slopes_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--y_measure", default="gbsc",
                    help="dependent measure, matching compute_vertex_slopes.py --measure")
    ap.add_argument("--covariates", default="age,female,apoe4,field_3t",
                    help="comma-separated nuisance terms beside the two group columns; "
                         "'age,female' is the three-term model (group, age, sex)")
    ap.add_argument("--field_subset", choices=["all", "stable", "3t", "15t"], default="all",
                    help="all; stable drops subjects whose field strength changed "
                         "mid-series; 3t and 15t fit one field strength alone")
    ap.add_argument("--weights", action="store_true",
                    help="weight subjects by slope precision from slope_precision.csv")
    ap.add_argument("--max_weight_ratio", type=float, default=4.0,
                    help="largest permitted ratio between any two subject weights; "
                         "clipped around the median, so 4 gives a half-to-double band")
    ap.add_argument("--drop_slope_outlier_pct", type=float, default=0.0,
                    help="drop subjects whose cortex-mean |slope| exceeds this "
                         "percentile; outcome-blind, for measures whose slope "
                         "variance is dominated by a few unstable scans")
    ap.add_argument("--outlier_within_group", action="store_true",
                    help="take the outlier percentile inside each group rather "
                         "than over the pooled sample, so the exclusion removes "
                         "the same fraction of converters and non-converters")
    args = ap.parse_args()

    subj = pd.read_csv(args.subjects)
    scans = pd.read_csv(args.scans)
    slopes_root, out = Path(args.slopes_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = args.y_measure

    done = pd.read_csv(slopes_root / "subjects_done.csv")["subject"]
    subj = subj[subj["subject"].isin(done)].sort_values("subject").reset_index(drop=True)
    subj["field_3t"] = field_3t(subj, scans)

    if args.field_subset == "stable":
        keep = subj["field_strength_changed"].fillna(1) == 0
    elif args.field_subset == "3t":
        keep = subj["field_3t"] == 1
    elif args.field_subset == "15t":
        keep = subj["field_3t"] == 0
    else:
        keep = pd.Series(True, index=subj.index)
    if not keep.all():
        print(f"field subset '{args.field_subset}': kept {int(keep.sum())} of {len(subj)} subjects")
        subj = subj[keep].reset_index(drop=True)

    if args.drop_slope_outlier_pct > 0:
        mag = np.array([np.mean([np.abs(np.asarray(nib.load(
            str(slopes_root / s / f"{h}.{y}_slope.mgh")).dataobj)).mean()
            for h in HEMIS]) for s in subj["subject"]])
        # A pooled percentile is outcome-blind in construction but not in
        # effect. If converters really do change faster, they are enriched in
        # the upper tail of |slope|, so a pooled cut removes more converters
        # than non-converters and takes real signal out with the noise. On this
        # cohort the pooled 97.5th percentile dropped 6 converters out of 11
        # subjects against a 32 per cent base rate, which makes the resulting
        # loss of significance impossible to read. Cutting inside each group
        # removes the same fraction from both and leaves the contrast alone.
        ev = subj["event"].to_numpy(dtype=float)
        if args.outlier_within_group:
            okay = np.zeros(len(mag), dtype=bool)
            for g in (0.0, 1.0):
                m = ev == g
                okay[m] = mag[m] <= np.percentile(mag[m], args.drop_slope_outlier_pct)
            where = "within each group"
        else:
            okay = mag <= np.percentile(mag, args.drop_slope_outlier_pct)
            where = "over the pooled sample"
        n_drop = int((~okay).sum())
        n_drop_conv = int(((~okay) & (ev == 1)).sum())
        print(f"slope-outlier QC: dropped {n_drop} subjects above the "
              f"{args.drop_slope_outlier_pct:g}th percentile taken {where}, "
              f"{n_drop_conv} of them converters "
              f"({100 * n_drop_conv / max(n_drop, 1):.0f}% against a "
              f"{100 * ev.mean():.0f}% base rate): "
              f"{subj.loc[~okay, 'subject'].tolist()}")
        subj = subj[okay].reset_index(drop=True)

    want = [c.strip() for c in args.covariates.split(",") if c.strip()]
    bad = [c for c in want if c not in COVMAP]
    if bad:
        raise SystemExit(f"unknown covariate(s) {bad}; choose from {sorted(COVMAP)}")
    if args.field_subset in ("3t", "15t") and "field_3t" in want:
        # constant inside a single stratum, so it would make X singular
        want.remove("field_3t")
        print("dropped field_3t from the covariates: constant within this subset")

    conv = subj["event"].to_numpy(dtype=float)
    cov = subj[[COVMAP[c] for c in want]].to_numpy(dtype=float)
    cov = cov - cov.mean(axis=0)
    X = np.column_stack([conv, 1.0 - conv, cov])
    cols = ["converter", "nonconverter"] + [f"{c}_c" for c in want]
    np.savetxt(out / "X.mat", X, fmt="%.8f")
    (out / "design_columns.txt").write_text("\n".join(cols) + "\n")

    if args.weights:
        prec = pd.read_csv(slopes_root / "slope_precision.csv").set_index("subject")
        w = subj["subject"].map(prec["slope_precision"]).to_numpy(dtype=float)
        if not np.isfinite(w).all():
            raise SystemExit(f"no slope precision for "
                             f"{subj.loc[~np.isfinite(w), 'subject'].tolist()}")
        p5, p95 = np.percentile(w, [5, 95])
        raw_ratio = p95 / p5
        alpha = min(1.0, np.log(args.max_weight_ratio) / np.log(raw_ratio)) \
            if raw_ratio > args.max_weight_ratio else 1.0
        w = w ** alpha
        half = np.sqrt(args.max_weight_ratio)
        med = np.median(w)
        lo, hi = med / half, med * half
        n_clip = int(((w < lo) | (w > hi)).sum())
        w = np.clip(w, lo, hi) / np.clip(w, lo, hi).mean()
        np.savetxt(out / "weights.txt", w, fmt="%.8f")
        eff = w.sum() ** 2 / (w ** 2).sum()
        print(f"weights: precision 5th-95th ratio {raw_ratio:.0f}x compressed by "
              f"power {alpha:.3f} to {args.max_weight_ratio:g}x, {n_clip} clipped, "
              f"scaled range {w.min():.2f} to {w.max():.2f}, effective n "
              f"{eff:.1f} of {len(w)} ({100 * eff / len(w):.0f}%)")

    # Contrast rows must span the global columns plus one column per PVR for B.
    cA = np.zeros((1, X.shape[1])); cA[0, 0], cA[0, 1] = 1.0, -1.0
    cB = np.zeros((1, X.shape[1] + 2)); cB[0, 0], cB[0, 1] = 1.0, -1.0
    np.savetxt(out / "contrast_A.mtx", cA, fmt="%.1f")
    np.savetxt(out / "contrast_B.mtx", cB, fmt="%.1f")

    for hemi in HEMIS:
        for meas in (y, "thickness", "curv"):
            maps = [np.asarray(nib.load(str(slopes_root / s / f"{hemi}.{meas}_slope.mgh")).dataobj,
                               dtype=np.float32).reshape(-1) for s in subj["subject"]]
            data = np.stack(maps, axis=-1)[:, None, None, :]     # nv x 1 x 1 x nsubj
            nib.save(nib.MGHImage(data, np.eye(4)), str(out / f"{hemi}.{meas}_slope.4d.mgh"))

    subj.to_csv(out / "subjects_used.csv", index=False)
    print(f"design: {len(subj)} subjects ({int(conv.sum())} converters), "
          f"X {X.shape} [{', '.join(cols)}], y={y}, stacked maps in {out}")


if __name__ == "__main__":
    main()
