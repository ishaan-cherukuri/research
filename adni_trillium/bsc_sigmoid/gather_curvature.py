"""Collect CIVET mean curvature into the matrix extract_sigmoid_features.py expects.

The paper residualises BSC against mean curvature at each vertex, so curvature
has to be on the same mesh and in the same scan order as the BSC data. CIVET
already writes a resampled mean curvature per hemisphere; this concatenates
left and right and stacks scans in manifest order, matching gather() in
extract_sigmoid_features.py exactly.

The filename pattern is a flag because it moves between CIVET versions. Under
2.1.0 the resampled mean curvature is

    <civet_root>/<cid>/surfaces/<prefix>_<cid>_native_mc_rsl_<hemi>.txt

If a run produced unresampled curvature only, pass the pattern for it, but be
aware the residualisation then mixes meshes and is not valid.

Usage:
    python3 gather_curvature.py --civet_root $SCRATCH/civet_out \
        --work_root $SCRATCH/bsc_sigmoid --manifest from_cluster/manifest_v3.csv \
        --out $SCRATCH/bsc_sigmoid/cache/curvature.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HEMIS = ["left", "right"]
DEFAULT_PATTERN = "{civet_root}/{cid}/surfaces/{prefix}_{cid}_native_mc_rsl_{hemi}.txt"


def make_image_id(row: pd.Series) -> str:
    return f"{row['subject']}_{row['visit_code']}_{row['acq_date']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--civet_root", required=True)
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--id_map", default=None)
    ap.add_argument("--prefix", default="adni")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work = Path(args.work_root)
    id_map = pd.read_csv(args.id_map or work / "civet_id_map.csv")

    man = pd.read_csv(args.manifest)
    man["image_id"] = man.apply(make_image_id, axis=1)
    man = man.merge(id_map[["image_id", "civet_id"]], on="image_id", how="inner")
    cids = man["civet_id"].tolist()

    data = None
    present = np.zeros(len(cids), dtype=bool)
    for i, cid in enumerate(cids):
        paths = [Path(args.pattern.format(civet_root=args.civet_root, cid=cid,
                                          prefix=args.prefix, hemi=h))
                 for h in HEMIS]
        if not all(p.exists() for p in paths):
            continue
        vals = np.concatenate([np.loadtxt(p, dtype=np.float32) for p in paths])
        if data is None:
            data = np.full((len(cids), vals.size), np.nan, dtype=np.float32)
        elif vals.size != data.shape[1]:
            print(f"[WARN] {cid} curvature has {vals.size} vertices, "
                  f"expected {data.shape[1]}; skipped")
            continue
        data[i] = vals
        present[i] = True

    if data is None:
        raise RuntimeError("No curvature files found; check --pattern")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=data, present=present)
    print(f"[DONE] {out}: {present.sum()}/{len(cids)} scans, "
          f"{data.shape[1]} vertices")


if __name__ == "__main__":
    main()
