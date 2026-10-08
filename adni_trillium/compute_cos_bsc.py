"""Scale-invariant boundary sharpness: cos(theta) = |grad I . n| / |grad I|.

The gradient measure the paper evaluates, gBSC = |grad I . n|, scales linearly
with image intensity: double the receive gain and it doubles. That is the most
likely reason it tracks signal-to-noise ratio and brain intensity dispersion so
faithfully, and why its cortex-wide level jumps when a subject changes scanner.

Dividing by the full gradient magnitude removes that. Because n is a unit
vector, Cauchy-Schwarz gives |grad I . n| <= |grad I|, so the ratio is bounded
in [0, 1] with no division instability, and it is invariant to any
multiplicative rescaling of image intensity. Geometrically it is the cosine of
the angle between the intensity gradient and the tissue-probability boundary:
1 where the intensity steps cleanly across the interface, lower where the
gradient wanders off the boundary normal.

This recomputes per-scan features on that definition, both global summaries and
per-parcel means, in one pass so the expensive DKT parcellation is done once.
Schema mirrors extract_bsc_features.py and compute_regional_bsc.py so the
existing cohort builder and survival models can consume it unchanged.

Usage (one chunk of subjects, for GNU Parallel fan-out):
    python3 compute_cos_bsc.py --manifest manifest.csv --bsc_root $SCRATCH/.../bsc \
        --out_csv out/cos_part0.csv [--skip N] [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import numpy as np
import pandas as pd
from antspynet.utilities import desikan_killiany_tourville_labeling

from compute_regional_bsc import AD_SIGNATURE, propagate_labels

PERCENTILES = (10, 25, 50, 75, 90)
BINS = (2, 2, 2)


def cos_map(bsc_dir: np.ndarray, bsc_mag: np.ndarray, band: np.ndarray):
    """cos(theta) inside the boundary band; NaN where there is no gradient.

    Both stored maps are already zeroed outside the band, so the band mask is
    applied explicitly rather than inferred from non-zero values.
    """
    inband = (band > 0) & (bsc_mag > 0)
    out = np.full(bsc_dir.shape, np.nan, dtype=np.float32)
    out[inband] = bsc_dir[inband] / bsc_mag[inband]
    # |grad I . n| <= |grad I| analytically; clip only guards float error.
    np.clip(out, 0.0, 1.0, out=out)
    return out, inband


def global_stats(cos: np.ndarray, inband: np.ndarray) -> dict:
    v = cos[inband]
    v = v[np.isfinite(v)]
    row = {"Nboundary_cos": int(v.size)}
    if v.size == 0:
        row.update({f"cos_{k}": np.nan for k in
                    ("mean", "std", "median", *[f"p{p}" for p in PERCENTILES])})
        row.update({f"cos_bin_{i}_mean": np.nan for i in range(int(np.prod(BINS)))})
        return row
    row["cos_mean"] = float(v.mean())
    row["cos_std"] = float(v.std())
    row["cos_median"] = float(np.median(v))
    for p, q in zip(PERCENTILES, np.quantile(v, [p / 100 for p in PERCENTILES])):
        row[f"cos_p{p}"] = float(q)

    bx, by, bz = BINS
    idx = np.argwhere(inband)
    shape = cos.shape
    xs = np.clip((idx[:, 0] * bx) // shape[0], 0, bx - 1)
    ys = np.clip((idx[:, 1] * by) // shape[1], 0, by - 1)
    zs = np.clip((idx[:, 2] * bz) // shape[2], 0, bz - 1)
    bid = xs * (by * bz) + ys * bz + zs
    vals = cos[idx[:, 0], idx[:, 1], idx[:, 2]]
    for i in range(bx * by * bz):
        sel = vals[bid == i]
        sel = sel[np.isfinite(sel)]
        row[f"cos_bin_{i}_mean"] = float(sel.mean()) if sel.size else np.nan
    return row


def region_stats(cos: np.ndarray, inband: np.ndarray, labels: np.ndarray, codes) -> dict:
    out = {}
    for code in codes:
        m = inband & (labels == code)
        v = cos[m]
        v = v[np.isfinite(v)]
        out[code] = (float(v.mean()) if v.size else np.nan, int(v.size))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bsc_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_label_attempts", type=int, default=3)
    ap.add_argument("--min_coverage", type=float, default=0.15)
    ap.add_argument("--max_mm", type=float, default=5.0)
    args = ap.parse_args()

    root = Path(args.bsc_root)
    man = pd.read_csv(args.manifest)
    if "image_id" not in man.columns:
        # plain manifests key scans by the triple; the derivatives tree is named
        # by the same convention (see run_bsc_batch.py::make_image_id)
        man["image_id"] = (man["subject"].astype(str) + "_"
                           + man["visit_code"].astype(str) + "_"
                           + man["acq_date"].astype(str))
    subs = sorted(man["subject"].unique())
    stop = None if args.limit is None else args.skip + args.limit
    subs = subs[args.skip:stop]

    rows = []
    for si, subj in enumerate(subs, 1):
        g = man[man["subject"] == subj]
        scans = [r for _, r in g.iterrows()
                 if (root / r["image_id"] / "bsc_dir_map.nii.gz").exists()]
        if not scans:
            continue

        # Reference selection by boundary-band coverage, as in the regional
        # pipeline: DKT labelling fails outright on a sizeable minority of
        # baselines and a bad reference propagates to every timepoint.
        base_t1 = base_lab = None
        ref_idx, best_cov = None, -1.0
        for attempt in range(min(args.max_label_attempts, len(scans))):
            cand = root / scans[attempt]["image_id"]
            try:
                t1 = ants.image_read(str(cand / "t1w_preproc.nii.gz"))
                dkt = desikan_killiany_tourville_labeling(
                    t1, do_preprocessing=True, return_probability_images=False,
                    do_lobar_parcellation=False)
                lab_i = dkt if isinstance(dkt, ants.ANTsImage) else dkt["segmentation_image"]
                bb_i = ants.image_read(str(cand / "boundary_band_mask.nii.gz"))
                prop_i = propagate_labels(lab_i.numpy(), bb_i.spacing, args.max_mm)
                bbn = bb_i.numpy() > 0
                cov = float((bbn & (prop_i > 0)).sum() / max(bbn.sum(), 1))
            except Exception as e:
                print(f"[WARN] {subj} labeling attempt {attempt}: {e}", flush=True)
                continue
            if cov > best_cov:
                base_t1, base_lab, ref_idx, best_cov = t1, lab_i, attempt, cov
            if cov >= args.min_coverage:
                break
        if base_lab is None:
            print(f"[ERROR] {subj}: all labeling attempts failed", flush=True)
            continue

        codes = [int(c) for c in np.unique(base_lab.numpy()) if c >= 1000]

        for k, r in enumerate(scans):
            d = root / r["image_id"]
            try:
                if k == ref_idx:
                    lab = base_lab
                else:
                    mov = ants.image_read(str(d / "t1w_preproc.nii.gz"))
                    reg = ants.registration(fixed=mov, moving=base_t1,
                                            type_of_transform="Rigid")
                    lab = ants.apply_transforms(
                        fixed=mov, moving=base_lab,
                        transformlist=reg["fwdtransforms"],
                        interpolator="genericLabel")
                bd = ants.image_read(str(d / "bsc_dir_map.nii.gz")).numpy()
                bm = ants.image_read(str(d / "bsc_mag_map.nii.gz")).numpy()
                bb_img = ants.image_read(str(d / "boundary_band_mask.nii.gz"))
                bb = bb_img.numpy()
                cos, inband = cos_map(bd, bm, bb)
                prop = propagate_labels(lab.numpy(), bb_img.spacing, args.max_mm)
                st = region_stats(cos, inband, prop, codes)
                covered = float(((bb > 0) & (prop > 0)).sum() / max((bb > 0).sum(), 1))
            except Exception as e:
                print(f"[ERROR] {r['image_id']}: {e}", flush=True)
                continue

            row = {"subject": subj, "visit_code": r["visit_code"],
                   "acq_date": r["acq_date"], "image_id": r["image_id"],
                   "band_coverage": covered, "ref_scan_idx": ref_idx,
                   "ref_coverage": best_cov}
            row.update(global_stats(cos, inband))
            for code, (mu, n) in st.items():
                row[f"cos_roi{code}"] = mu
                row[f"ncos_roi{code}"] = n

            # Same pre-specified AD-signature composite, boundary-voxel weighted.
            num = den = 0.0
            for c in AD_SIGNATURE.values():
                for code in (c, c + 1000):
                    v = st.get(code)
                    if v and v[1] > 0 and np.isfinite(v[0]):
                        num += v[0] * v[1]
                        den += v[1]
            row["cos_adsig"] = num / den if den else np.nan
            row["n_adsig"] = int(den)
            rows.append(row)

        if si % 10 == 0:
            print(f"[{si}/{len(subs)}] {subj}", flush=True)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} scan-rows)")


if __name__ == "__main__":
    main()
