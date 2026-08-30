"""Construct-validity checks for the sigmoid BSC.

Five checks. The first two are the paper's own; the rest are what this project
needs in order to say whether its negative result survives a correctly
specified measure.

  1. Curvature. T1 intensity runs higher in gyral crowns than in sulcal folds,
     so any intensity-derived measure is at risk of partly measuring folding.
     Reports the vertexwise BSC-curvature correlation before and after
     residualisation. The paper's claim is that it survives residualisation.

  2. Tissue intensity ratio. The measure BSC was proposed to replace. The
     paper's motivating result is that the ratio correlates with curvature
     across most of the cortex while BSC, once residualised, does not.

  3. Image quality. The decisive one for this project. The Atropos BSC tracked
     intensity dispersion (r = 0.239) and SNR (r = 0.191) while tracking age at
     r = -0.007, which is what made the existing manuscript a design-effect
     paper. These correlations are recomputed on the sigmoid measure. If they
     hold, the finding stands against the published measure and gets stronger;
     if they vanish, they were an artifact of the gradient proxy.

  4. Agreement with the Atropos measure. How much of the existing null result
     is attributable to having measured the wrong quantity.

  5. Longitudinal jitter. CIVET is run per scan, so surface placement varies
     between a subject's timepoints and that variance lands directly in the
     slopes the survival models consume. Estimated from rescan pairs closer
     together than --jitter_max_days, where real biological change is small
     relative to measurement noise. Reported as ICC(1,1): the fraction of
     variance that is between-subject rather than within.

Usage:
    python3 run_validity_checks.py --sigmoid from_cluster/bsc_sigmoid_simple_features_sigmoid_v3.csv \
        --atropos from_cluster/bsc_simple_features_v3.csv \
        --t1_scans from_cluster/t1_scan_features.csv \
        --cohort analysis/results/spec_v3/spec_cohort.csv \
        --out_dir analysis/results/sigmoid_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Per-scan quality columns the Atropos-era correlations were reported on.
QC_COLS = ["qc_brain_std", "qc_snr", "qc_brain_mean", "qc_brain_bg_ratio",
           "meta_field_strength_t"]

# Summary columns compared against them, primary measure first.
BSC_COLS = ["bscsig_mean", "bscsig_median", "bscsig_std",
            "bscsigfree_mean", "bscratio_mean"]


def corr(x: pd.Series, y: pd.Series) -> dict:
    """Pearson r with n and p, tolerant of missing values."""
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "r": None, "p": None}
    r, p = stats.pearsonr(x[ok], y[ok])
    return {"n": int(ok.sum()), "r": float(r), "p": float(p)}


def vertexwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-by-column Pearson r between two (n_scans, n_vertices) matrices."""
    az = a - np.nanmean(a, axis=0)
    bz = b - np.nanmean(b, axis=0)
    num = np.nansum(az * bz, axis=0)
    den = np.sqrt(np.nansum(az ** 2, axis=0) * np.nansum(bz ** 2, axis=0))
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def check_curvature(raw_npz: Path, resid_npz: Path, curv_npz: Path) -> dict:
    """Check 1: vertexwise BSC-curvature correlation, before and after."""
    curv = np.load(curv_npz)["data"]
    out = {}
    for label, path in (("before", raw_npz), ("after", resid_npz)):
        if not path.exists():
            out[label] = None
            continue
        r = vertexwise_corr(np.load(path)["data"], curv)
        r = r[np.isfinite(r)]
        out[label] = {
            "n_vertices": int(r.size),
            "mean_abs_r": float(np.abs(r).mean()),
            "median_r": float(np.median(r)),
            "frac_abs_r_gt_0.1": float((np.abs(r) > 0.1).mean()),
            "frac_abs_r_gt_0.3": float((np.abs(r) > 0.3).mean()),
        }
    return out


def check_ratio_curvature(ratio_npz: Path, curv_npz: Path) -> dict:
    """Check 2: the tissue intensity ratio against curvature."""
    if not ratio_npz.exists():
        return None
    r = vertexwise_corr(np.load(ratio_npz)["data"], np.load(curv_npz)["data"])
    r = r[np.isfinite(r)]
    return {
        "n_vertices": int(r.size),
        "mean_abs_r": float(np.abs(r).mean()),
        "median_r": float(np.median(r)),
        "frac_negative": float((r < 0).mean()),
        "frac_abs_r_gt_0.3": float((np.abs(r) > 0.3).mean()),
    }


def check_image_quality(scans: pd.DataFrame) -> dict:
    out = {}
    for b in BSC_COLS:
        if b not in scans.columns:
            continue
        out[b] = {q: corr(scans[b], scans[q])
                  for q in QC_COLS if q in scans.columns}
        if "age_at_scan" in scans.columns:
            out[b]["age_at_scan"] = corr(scans[b], scans["age_at_scan"])
    return out


def check_agreement(scans: pd.DataFrame) -> dict:
    out = {}
    for a in ("bsc_dir_mean", "bsc_mag_mean", "bsc_dir_median"):
        if a not in scans.columns:
            continue
        out[a] = {b: corr(scans[b], scans[a])
                  for b in BSC_COLS if b in scans.columns}
    return out


def check_jitter(scans: pd.DataFrame, col: str, max_days: int) -> dict:
    """Check 5: ICC(1,1) over subjects contributing a short-interval rescan pair.

    Restricting to close-together pairs is what makes this a measurement-noise
    estimate rather than a biology estimate. A low ICC means slopes built on
    this measure are largely surface-placement noise.
    """
    if col not in scans.columns:
        return None

    df = scans[["subject", "acq_date", col]].dropna().copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).sort_values(["subject", "acq_date"])

    pairs = []
    for subj, g in df.groupby("subject"):
        d = g["acq_date"].to_numpy()
        v = g[col].to_numpy()
        for i in range(len(g) - 1):
            gap = (d[i + 1] - d[i]) / np.timedelta64(1, "D")
            if gap <= max_days:
                pairs.append((subj, v[i], v[i + 1], gap))
                break

    if len(pairs) < 10:
        return {"n_pairs": len(pairs), "icc": None,
                "note": "too few short-interval rescan pairs"}

    p = pd.DataFrame(pairs, columns=["subject", "v1", "v2", "gap_days"])
    vals = p[["v1", "v2"]].to_numpy()
    grand = vals.mean()
    subj_means = vals.mean(axis=1)
    n = len(p)

    # One-way random effects ICC(1,1).
    ms_between = 2 * ((subj_means - grand) ** 2).sum() / (n - 1)
    ms_within = ((vals - subj_means[:, None]) ** 2).sum() / n
    icc = (ms_between - ms_within) / (ms_between + ms_within)

    return {
        "n_pairs": n,
        "median_gap_days": float(p["gap_days"].median()),
        "icc": float(icc),
        "within_subject_sd": float(np.sqrt(ms_within)),
        "between_subject_sd": float(np.sqrt(max(ms_between - ms_within, 0) / 2)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sigmoid", required=True)
    ap.add_argument("--atropos", default=None)
    ap.add_argument("--t1_scans", required=True)
    ap.add_argument("--cohort", default=None,
                    help="spec_cohort.csv, for age and site")
    ap.add_argument("--cache_dir", default=None,
                    help="<work_root>/cache, for the vertexwise checks 1 and 2")
    ap.add_argument("--curvature_npz", default=None)
    ap.add_argument("--jitter_max_days", type=int, default=180)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    scans = pd.read_csv(args.sigmoid)
    print(f"Sigmoid features: {len(scans)} scans")

    t1 = pd.read_csv(args.t1_scans)
    keep = ["image_id"] + [c for c in QC_COLS if c in t1.columns]
    scans = scans.merge(t1[keep], on="image_id", how="left")

    if args.atropos:
        at = pd.read_csv(args.atropos)
        acols = [c for c in at.columns if c.startswith("bsc_")]
        scans = scans.merge(at[["image_id"] + acols], on="image_id", how="left")
        print(f"  merged Atropos BSC on {scans['bsc_dir_mean'].notna().sum()} scans")

    if args.cohort:
        co = pd.read_csv(args.cohort)
        cc = [c for c in ("age_at_landmark", "site") if c in co.columns]
        scans = scans.merge(co[["subject"] + cc], on="subject", how="left")
        # Ages are recorded at the landmark, so shift by each scan's offset from
        # the subject's first scan to get an age at acquisition.
        if "age_at_landmark" in scans.columns:
            d = pd.to_datetime(scans["acq_date"], errors="coerce")
            off = (d - d.groupby(scans["subject"]).transform("min")).dt.days / 365.25
            scans["age_at_scan"] = scans["age_at_landmark"] + off

    results = {
        "n_scans": int(len(scans)),
        "n_subjects": int(scans["subject"].nunique()),
        "image_quality": check_image_quality(scans),
        "atropos_agreement": check_agreement(scans) if args.atropos else None,
        "longitudinal_jitter": {
            c: check_jitter(scans, c, args.jitter_max_days)
            for c in ("bscsig_mean", "bscsigfree_mean", "bsc_dir_mean")
            if c in scans.columns
        },
    }

    if args.cache_dir and args.curvature_npz:
        cache = Path(args.cache_dir)
        results["curvature"] = check_curvature(
            cache / "model_c_log_20mm.npz",
            cache / "model_c_log_20mm_resid.npz",
            Path(args.curvature_npz),
        )
        results["ratio_vs_curvature"] = check_ratio_curvature(
            cache / "model_ratio_20mm.npz", Path(args.curvature_npz))
    else:
        print("[NOTE] no --cache_dir/--curvature_npz; "
              "skipping vertexwise curvature checks 1 and 2")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "validity_checks.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n[DONE] {out}")
    iq = results["image_quality"].get("bscsig_mean", {})
    print("\nbscsig_mean vs:")
    for k, v in iq.items():
        if v["r"] is not None:
            print(f"  {k:24s} r={v['r']:+.4f}  p={v['p']:.3g}  n={v['n']}")


if __name__ == "__main__":
    main()
