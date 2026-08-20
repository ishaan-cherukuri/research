"""Per-region BSC from the Desikan-Killiany-Tourville parcellation (spec section 5).

The BSC maps already computed by run_bsc_batch.py live in each scan's native
space, so a regional summary needs only a voxelwise cortical label map to
intersect with the boundary band. No surface reconstruction is involved, which
is why this does not need FreeSurfer.

Labels are generated once per subject on their baseline scan and propagated to
the subject's other timepoints by rigid registration, rather than being
regenerated independently at every visit. The feature of interest is a slope,
so independent per-timepoint labelling would inject parcellation jitter
directly into the quantity being modelled. Same subject, same head, so a rigid
transform is sufficient, and it cuts labelling cost by the mean scan count.

Outputs one row per scan per region, with the mean directional and magnitude
BSC over boundary voxels inside that region plus the voxel count backing it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Standard FreeSurfer aparc codes, 1000-series left / 2000-series right.
AD_SIGNATURE = {
    "entorhinal": 1006, "fusiform": 1007, "inferiortemporal": 1009,
    "middletemporal": 1015, "parahippocampal": 1016,
    "posteriorcingulate": 1023, "precuneus": 1025,
}


def propagate_labels(labels, spacing, max_mm):
    """Extend cortical parcels inward to cover the gray-white boundary band.

    DKT labels the cortical gray ribbon, but the BSC boundary band straddles the
    gray-white interface and therefore sits mostly in unlabelled white matter:
    measured on a real scan, 82 percent of band voxels carry label 0 and under
    10 percent fall inside a parcel. Intersecting the two directly would sample
    a thin, biased sliver of each region.

    Each voxel is instead assigned the label of its nearest cortical voxel, with
    assignment refused beyond max_mm so that deep white matter is not attributed
    to a parcel it does not border.
    """
    cort = np.where(labels >= 1000, labels, 0)
    if not cort.any():
        return cort
    dist, idx = distance_transform_edt(cort == 0, sampling=spacing,
                                       return_distances=True, return_indices=True)
    prop = cort[tuple(idx)]
    prop[dist > max_mm] = 0
    return prop


def region_stats(bsc_dir, bsc_mag, band, labels, codes):
    """Mean BSC inside each label, restricted to boundary-band voxels."""
    out = {}
    inband = band > 0
    for code in codes:
        m = inband & (labels == code)
        n = int(m.sum())
        out[code] = (float(bsc_dir[m].mean()) if n else np.nan,
                     float(bsc_mag[m].mean()) if n else np.nan,
                     n)
    return out


def main():
    import ants
    from antspynet.utilities import desikan_killiany_tourville_labeling

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bsc_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_label_attempts", type=int, default=3)
    ap.add_argument("--min_coverage", type=float, default=0.15)
    ap.add_argument("--max_mm", type=float, default=5.0,
                    help="max distance a band voxel may sit from its parcel")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["image_id"] = (man["subject"] + "_" + man["visit_code"] + "_"
                       + man["acq_date"].astype(str))
    man = man.sort_values(["subject", "acq_date"])

    subs = sorted(man["subject"].unique())
    stop = None if args.limit is None else args.skip + args.limit
    subs = subs[args.skip:stop]

    root = Path(args.bsc_root)
    rows = []
    for si, subj in enumerate(subs, 1):
        g = man[man["subject"] == subj]
        scans = [r for _, r in g.iterrows()
                 if (root / r["image_id"] / "bsc_dir_map.nii.gz").exists()]
        if not scans:
            continue

        # Pick the labelling reference by QC rather than assuming the first scan
        # works. DKT labelling fails outright on roughly one baseline in seven,
        # and because labels propagate from the reference, a bad reference takes
        # the whole subject down with it. Try successive scans until one yields
        # adequate boundary-band coverage, then keep the best attempt.
        base_t1 = base_lab = None
        ref_idx, best_cov = None, -1.0
        for attempt in range(min(args.max_label_attempts, len(scans))):
            cand_dir = root / scans[attempt]["image_id"]
            try:
                t1 = ants.image_read(str(cand_dir / "t1w_preproc.nii.gz"))
                dkt = desikan_killiany_tourville_labeling(
                    t1, do_preprocessing=True, return_probability_images=False,
                    do_lobar_parcellation=False)
                lab_i = dkt if isinstance(dkt, ants.ANTsImage) else dkt["segmentation_image"]
                bb_i = ants.image_read(str(cand_dir / "boundary_band_mask.nii.gz"))
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
        if best_cov < args.min_coverage:
            print(f"[WARN] {subj}: best coverage {best_cov:.3f} below "
                  f"{args.min_coverage}", flush=True)

        codes = [int(c) for c in np.unique(base_lab.numpy()) if c >= 1000]

        for k, r in enumerate(scans):
            d = root / r["image_id"]
            try:
                if k == ref_idx:
                    lab = base_lab
                else:
                    mov = ants.image_read(str(d / "t1w_preproc.nii.gz"))
                    # Rigid is adequate within subject and avoids deforming the
                    # cortical ribbon, which would bias the boundary band.
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
                prop = propagate_labels(lab.numpy(), bb_img.spacing, args.max_mm)
                st = region_stats(bd, bm, bb, prop, codes)
                covered = float(((bb > 0) & (prop > 0)).sum() / max((bb > 0).sum(), 1))
            except Exception as e:
                print(f"[ERROR] {r['image_id']}: {e}", flush=True)
                continue

            row = {"subject": subj, "visit_code": r["visit_code"],
                   "acq_date": r["acq_date"], "image_id": r["image_id"],
                   "band_coverage": covered, "ref_scan_idx": ref_idx,
                   "ref_coverage": best_cov}
            for code, (dm, mg, n) in st.items():
                row[f"bscdir_roi{code}"] = dm
                row[f"bscmag_roi{code}"] = mg
                row[f"n_roi{code}"] = n
            # Pre-specified AD-signature composite: boundary-voxel-weighted mean
            # across the seven bilateral regions, so large regions do not get
            # diluted by small ones.
            num_d = num_m = den = 0.0
            for c in AD_SIGNATURE.values():
                for code in (c, c + 1000):
                    v = st.get(code)
                    if v and v[2] > 0 and np.isfinite(v[0]):
                        num_d += v[0] * v[2]
                        num_m += v[1] * v[2]
                        den += v[2]
            row["bscdir_adsig"] = num_d / den if den else np.nan
            row["bscmag_adsig"] = num_m / den if den else np.nan
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
