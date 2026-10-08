"""Cluster-wise p-values corrected across both hemispheres.

vertex_glm.py runs each hemisphere with the same seed, so permutation i is the
same shuffle on lh and rh. The corrected null for the largest cluster is the
per-iteration maximum over the two hemispheres, and each observed cluster's
cluster-wise p is the fraction of iterations whose maximum is at least as
large (with the observed data counted once).

Writes, per model:
  cluster_summary.csv            all observed clusters with cluster-wise p
  {hemi}.sig_clusters.mgh        t inside clusters with cwp < alpha, 0 elsewhere
  {hemi}.cwp.mgh                 -log10 cluster-wise p painted on each cluster

Usage: python3 combine_clusters.py --glm_dir GLM --model B [--alpha 0.05]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def load(path):
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).reshape(-1)


def save(path, vals):
    nib.save(nib.MGHImage(vals.astype(np.float32).reshape(-1, 1, 1), np.eye(4)), str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm_dir", required=True)
    ap.add_argument("--model", choices=["A", "B"], required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    glm = Path(args.glm_dir)
    dirs = {h: glm / f"py{args.model}_{h}" for h in ("lh", "rh")}

    null = np.maximum(np.load(dirs["lh"] / "perm_max_area.npy"),
                      np.load(dirs["rh"] / "perm_max_area.npy"))
    nperm = null.size
    frames = []
    for h, d in dirs.items():
        obs = pd.read_csv(d / "clusters_obs.csv")
        obs["cwp"] = [(1 + (null >= a).sum()) / (1 + nperm) for a in obs["area_mm2"]]
        t = load(d / "t.mgh")
        cid = load(d / "cluster_id.mgh").astype(int)
        sig = np.zeros_like(t)
        cwp_map = np.zeros_like(t)
        for _, r in obs.iterrows():
            m = cid == r["cluster"]
            cwp_map[m] = -np.log10(r["cwp"])
            if r["cwp"] < args.alpha:
                sig[m] = t[m]
        save(glm / f"{h}.model{args.model}.sig_clusters.mgh", sig)
        save(glm / f"{h}.model{args.model}.cwp.mgh", cwp_map)
        frames.append(obs)
    summary = pd.concat(frames).sort_values("cwp").reset_index(drop=True)
    summary["model"] = args.model
    summary.to_csv(glm / f"model{args.model}_cluster_summary.csv", index=False)
    thr = float(np.percentile(null, 100 * (1 - args.alpha)))
    n_sig = int((summary["cwp"] < args.alpha).sum())
    print(f"model {args.model}: {len(summary)} observed clusters, {n_sig} survive "
          f"cwp<{args.alpha} (critical area {thr:.0f} mm2 over {nperm} permutations)")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
