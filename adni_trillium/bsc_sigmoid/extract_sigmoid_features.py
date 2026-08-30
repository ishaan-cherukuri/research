"""Turn per-vertex sigmoid outputs into the per-scan feature CSVs the existing
survival pipeline already knows how to read.

Three stages:

  1. Gather. Every scan's resampled left/right vertex files are concatenated
     into one (n_scans, n_vertices) matrix per quantity and cached as .npy, so
     the ~170k small text files are read once rather than once per analysis.

  2. Residualise against mean curvature. The paper regresses BSC on CIVET's
     mean curvature at each vertex and keeps the residual, because T1 intensity
     is higher in gyral crowns than sulcal fold: without this, BSC partly
     measures folding. The regression is fit within site, since curvature's
     effect on intensity runs through acquisition. Sites with fewer than
     --min_site_n scans are left unresidualised and flagged rather than fit on
     too little data.

  3. Summarise. Two CSVs are written whose schemas mirror the existing
     bsc_simple_features_v3.csv and per_scan_regional.csv, so
     build_spec_cohort.py picks the columns up with no code change: load_scans()
     in build_landmark_cohort.py treats every non-metadata column as a feature,
     keyed on image_id.

Column prefixes are deliberately distinct from the Atropos pipeline's bsc_*, so
both measures can sit in one cohort table and be compared directly:

    bscsig_*      log growth rate, reference bounds  <- the published measure
    bscsigfree_*  log growth rate, a unbounded below <- sensitivity variant
    bscratio_*    tissue intensity ratio

Usage:
    python3 extract_sigmoid_features.py --work_root $SCRATCH/bsc_sigmoid \
        --manifest from_cluster/manifest_v3.csv \
        --atlas_left icbm_atlas_left.txt --atlas_right icbm_atlas_right.txt \
        --atlas_labels atlas_labels.csv --out_dir from_cluster/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Vertex quantity -> output column prefix. Only the 20 mm smoothing is carried
# into features; the 10 mm files stay on disk for the sensitivity analysis.
QUANTITIES = {
    "model_c_log": "bscsig",
    "model_c_free_log": "bscsigfree",
    "model_ratio": "bscratio",
}
FWHM = 20
HEMIS = ["left", "right"]
PERCENTILES = [10, 25, 50, 75, 90]

# Cortical regions making up the AD signature, by atlas region name. Matched
# against the --atlas_labels name column, case-insensitively, as substrings.
# This mirrors AD_SIGNATURE in compute_regional_bsc.py, with one unavoidable
# gap: CIVET's surface atlas has no entorhinal parcel, so the composite here is
# six regions where the volumetric version has seven. Not interchangeable with
# it; compare within a measure, not across.
AD_SIGNATURE_NAMES = [
    "parahippocampal", "fusiform", "temporal_inf", "temporal_mid",
    "cingulum_post", "precuneus",
]

SCAN_META = ["subject", "visit_code", "acq_date", "image_id", "diagnosis"]


def make_image_id(row: pd.Series) -> str:
    return f"{row['subject']}_{row['visit_code']}_{row['acq_date']}"


def gather(work: Path, cids: list[str], quantity: str,
           cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate resampled left+right vertex data into (n_scans, n_vertices).

    Scans missing either hemisphere become an all-NaN row and are reported by
    the caller; dropping them here would silently change cohort size.
    """
    if cache.exists():
        d = np.load(cache)
        return d["data"], d["present"]

    rsl = work / "sigmoid_fit" / "resampled"
    data = None
    present = np.zeros(len(cids), dtype=bool)

    for i, cid in enumerate(cids):
        paths = [rsl / f"{cid}_{quantity}_{h}_{FWHM}mm_rsl.txt" for h in HEMIS]
        if not all(p.exists() for p in paths):
            continue
        vals = np.concatenate([np.loadtxt(p, dtype=np.float32) for p in paths])
        if data is None:
            data = np.full((len(cids), vals.size), np.nan, dtype=np.float32)
        elif vals.size != data.shape[1]:
            print(f"[WARN] {cid} has {vals.size} vertices, "
                  f"expected {data.shape[1]}; skipped")
            continue
        data[i] = vals
        present[i] = True

    if data is None:
        raise RuntimeError(f"No resampled data found for quantity {quantity}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, data=data, present=present)
    return data, present


def residualise_curvature(data: np.ndarray, curv: np.ndarray,
                          site: np.ndarray, min_site_n: int) -> tuple[np.ndarray, dict]:
    """Vertexwise regression of BSC on mean curvature, within site, keeping the
    residual plus the overall mean so the output stays on the original scale."""
    out = np.full_like(data, np.nan)
    info = {"sites_residualised": 0, "sites_skipped": 0, "scans_skipped": 0}

    for s in np.unique(site[~pd.isna(site)]):
        idx = np.where(site == s)[0]
        rows = idx[np.isfinite(data[idx]).any(axis=1)
                   & np.isfinite(curv[idx]).any(axis=1)]
        if rows.size < min_site_n:
            out[idx] = data[idx]
            info["sites_skipped"] += 1
            info["scans_skipped"] += int(idx.size)
            continue

        y, x = data[rows], curv[rows]
        ok = np.isfinite(y) & np.isfinite(x)
        n = ok.sum(axis=0)

        ym = np.where(n > 0, np.nansum(np.where(ok, y, 0), axis=0) / np.maximum(n, 1), np.nan)
        xm = np.where(n > 0, np.nansum(np.where(ok, x, 0), axis=0) / np.maximum(n, 1), np.nan)
        yc = np.where(ok, y - ym, 0.0)
        xc = np.where(ok, x - xm, 0.0)

        sxx = (xc ** 2).sum(axis=0)
        beta = np.divide((xc * yc).sum(axis=0), sxx,
                         out=np.zeros_like(sxx), where=sxx > 0)
        # Add ym back so residualised BSC keeps the units and rough location of
        # the raw measure; a mean-zero residual would make the summary stats
        # below uninterpretable.
        out[rows] = y - beta * (x - xm)
        info["sites_residualised"] += 1

    return out, info


def summarise(data: np.ndarray, prefix: str) -> pd.DataFrame:
    """Per-scan distribution summary across vertices."""
    with np.errstate(all="ignore"):
        cols = {
            f"{prefix}_mean": np.nanmean(data, axis=1),
            f"{prefix}_std": np.nanstd(data, axis=1),
            f"{prefix}_median": np.nanmedian(data, axis=1),
        }
        for p in PERCENTILES:
            cols[f"{prefix}_p{p}"] = np.nanpercentile(data, p, axis=1)
    return pd.DataFrame(cols)


def load_atlas(left: Path, right: Path, n_vertices: int) -> np.ndarray:
    labels = np.concatenate([
        np.loadtxt(left, dtype=np.int32), np.loadtxt(right, dtype=np.int32)
    ])
    if labels.size != n_vertices:
        raise ValueError(f"Atlas has {labels.size} vertices, "
                         f"data has {n_vertices}")
    return labels


def regional_means(data: np.ndarray, labels: np.ndarray, prefix: str,
                   label_names: dict[int, str]) -> pd.DataFrame:
    """Mean per atlas region, plus the AD-signature composite."""
    cols = {}
    codes = [c for c in np.unique(labels) if c != 0]
    with np.errstate(all="ignore"):
        for code in codes:
            cols[f"{prefix}_roi{code}"] = np.nanmean(data[:, labels == code], axis=1)

        sig = [c for c in codes
               if any(k in label_names.get(int(c), "").lower()
                      for k in AD_SIGNATURE_NAMES)]
        if sig:
            mask = np.isin(labels, sig)
            cols[f"{prefix}_adsig"] = np.nanmean(data[:, mask], axis=1)
        else:
            print("[WARN] no atlas regions matched the AD signature; "
                  "adsig column omitted")
    return pd.DataFrame(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--id_map", default=None,
                    help="civet_id_map.csv (default: <work_root>/civet_id_map.csv)")
    ap.add_argument("--curvature_npz", default=None,
                    help="Cached (n_scans, n_vertices) mean curvature matrix. "
                         "Without it, residualisation is skipped and the raw "
                         "measure is written.")
    ap.add_argument("--atlas_left", required=True)
    ap.add_argument("--atlas_right", required=True)
    ap.add_argument("--atlas_labels", required=True,
                    help="CSV with columns code,name for the surface atlas")
    ap.add_argument("--site_csv", default=None,
                    help="CSV with columns image_id,site. Defaults to the "
                         "manifest's dataset column, which is a coarse proxy.")
    ap.add_argument("--min_site_n", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_tag", default="sigmoid_v3")
    args = ap.parse_args()

    work = Path(args.work_root)
    id_map = pd.read_csv(args.id_map or work / "civet_id_map.csv")

    man = pd.read_csv(args.manifest)
    man["image_id"] = man.apply(make_image_id, axis=1)
    man = man.merge(id_map[["image_id", "civet_id"]], on="image_id", how="inner")
    print(f"Manifest scans with a CIVET id: {len(man)}")

    cids = man["civet_id"].tolist()

    if args.site_csv:
        sites = pd.read_csv(args.site_csv).set_index("image_id")["site"]
        site = man["image_id"].map(sites).to_numpy()
    else:
        site = man["dataset"].to_numpy() if "dataset" in man.columns \
            else np.full(len(man), "all")
        print("[NOTE] no --site_csv; residualising within manifest dataset")

    curv = None
    if args.curvature_npz:
        curv = np.load(args.curvature_npz)["data"]

    labels_df = pd.read_csv(args.atlas_labels)
    label_names = dict(zip(labels_df["code"].astype(int),
                           labels_df["name"].astype(str)))

    meta = man[["subject", "visit_code", "acq_date", "image_id"]].copy()
    if "diagnosis" in man.columns:
        meta["diagnosis"] = man["diagnosis"].to_numpy()

    simple_parts, regional_parts = [meta.reset_index(drop=True)], \
        [meta[["image_id"]].reset_index(drop=True)]
    present_any = None
    atlas = None

    for quantity, prefix in QUANTITIES.items():
        print(f"\n=== {quantity} -> {prefix}_* ===")
        cache = work / "cache" / f"{quantity}_{FWHM}mm.npz"
        data, present = gather(work, cids, quantity, cache)
        print(f"  {present.sum()}/{len(cids)} scans present")
        present_any = present if present_any is None else (present_any | present)

        if curv is not None:
            if curv.shape != data.shape:
                raise ValueError(f"Curvature matrix {curv.shape} does not match "
                                 f"data {data.shape}")
            data, info = residualise_curvature(data, curv, site, args.min_site_n)
            print(f"  residualised: {info}")
            # Kept so run_validity_checks.py can compare the vertexwise
            # BSC-curvature correlation before and after.
            resid_cache = cache.with_name(cache.stem + "_resid.npz")
            if not resid_cache.exists():
                np.savez_compressed(resid_cache, data=data, present=present)
        else:
            print("  [NOTE] no curvature supplied; writing unresidualised values")

        if atlas is None:
            atlas = load_atlas(Path(args.atlas_left), Path(args.atlas_right),
                               data.shape[1])

        simple_parts.append(summarise(data, prefix))
        regional_parts.append(regional_means(data, atlas, prefix, label_names))

    simple = pd.concat(simple_parts, axis=1)
    simple["n_vertices_present"] = np.where(present_any, atlas.size, 0)
    simple = simple[present_any]

    regional = pd.concat(regional_parts, axis=1)[present_any]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / f"bsc_sigmoid_simple_features_{args.out_tag}.csv"
    p2 = out_dir / f"per_scan_regional_sigmoid_{args.out_tag}.csv"
    simple.to_csv(p1, index=False)
    regional.to_csv(p2, index=False)

    print(f"\n[DONE] {p1}  ({len(simple)} scans, {simple.shape[1]} cols)")
    print(f"[DONE] {p2}  ({len(regional)} scans, {regional.shape[1]} cols)")


if __name__ == "__main__":
    main()
