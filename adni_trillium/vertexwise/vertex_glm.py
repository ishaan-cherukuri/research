"""Vertex-wise group GLM on fsaverage6 with cluster-wise permutation correction.

mri_glmfit cannot permute a design that has per-vertex regressors, which is
exactly what Model B needs, so both models run through this script and Model A
is cross-checked against mri_glmfit-sim separately (slurm/03_glm.sh).

Model A  y_v ~ 1 + converter + age + sex + APOE4 + 3T
Model B  y_v ~ 1 + converter + age + sex + APOE4 + 3T + thickness_slope_v + curv_slope_v

y_v is the gBSC slope at vertex v. The statistic is the t for the converter
coefficient. The null distribution follows Freedman and Lane (1983) as set out
by Winkler et al. (2014): residuals of the reduced model (everything except
the converter column) are permuted with one shuffle per iteration shared by
every vertex, the full model is refitted, and the largest suprathreshold
cluster area is recorded. The same seed on both hemispheres gives identical
permutation sequences, so combine_clusters.py can take the per-iteration
maximum across hemispheres, correcting over both surfaces at once.

Outputs in --out_dir:
  t.mgh             converter t at each cortex vertex (0 outside cortex)
  beta.mgh          converter coefficient (gBSC slope units per year)
  sig.mgh           signed -log10 uncorrected p
  cluster_id.mgh    observed cluster labels at the cluster-forming threshold
  clusters_obs.csv  one row per observed cluster: area, size, peak, aparc label
  perm_max_area.npy largest cluster area under each permutation
  info.json

Usage:
    python3 vertex_glm.py --glm_dir GLM --model B --hemi lh --out_dir GLM/pyB_lh \
        --nperm 10000 --seed 0 --cft 0.01 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot, read_geometry, read_label
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import t as t_dist

TRG = "fsaverage6"


# ----------------------------------------------------------------------------
# mesh helpers
# ----------------------------------------------------------------------------
def load_mesh(fs_subjects_dir: Path, hemi: str):
    surf = fs_subjects_dir / TRG / "surf" / f"{hemi}.white"
    coords, faces = read_geometry(str(surf))
    nv = coords.shape[0]
    # vertex area: one third of every incident triangle
    a, b, c = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = np.zeros(nv)
    for k in range(3):
        np.add.at(area, faces[:, k], tri_area / 3.0)
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    adj = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(nv, nv)).tocsr()
    adj = ((adj + adj.T) > 0).astype(np.int8).tocsr()
    cortex = np.zeros(nv, dtype=bool)
    cortex[read_label(str(fs_subjects_dir / TRG / "label" / f"{hemi}.cortex.label"))] = True
    labels, _, names = read_annot(str(fs_subjects_dir / TRG / "label" / f"{hemi}.aparc.annot"))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    return adj, area, cortex, labels, names


def clusters(mask: np.ndarray, adj: sparse.csr_matrix):
    """Connected components of the vertices where mask is True. Returns labels
    (-1 off-cluster, else 0..k-1) and k."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.full(mask.shape, -1, dtype=np.int32), 0
    sub = adj[idx][:, idx]
    k, comp = connected_components(sub, directed=False)
    out = np.full(mask.shape, -1, dtype=np.int32)
    out[idx] = comp
    return out, k


def max_cluster_area(mask: np.ndarray, adj, area: np.ndarray) -> float:
    lab, k = clusters(mask, adj)
    if k == 0:
        return 0.0
    return float(np.bincount(lab[lab >= 0], weights=area[lab >= 0], minlength=k).max())


# ----------------------------------------------------------------------------
# per-vertex least squares with a possibly vertex-specific design
# ----------------------------------------------------------------------------
def load_4d(path: Path) -> np.ndarray:
    d = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    return d.reshape(d.shape[0], -1)          # nv x nsubj


def build_design(glm_dir: Path, model: str, hemi: str, cortex: np.ndarray):
    """Return M (nv_c x n x p), the index of the converter column, and the
    reduced-model design Z (nv_c x n x pz). Model A designs are broadcast from
    a single matrix. Only cortex vertices are kept."""
    X = np.loadtxt(glm_dir / "X.mat")
    cols = (glm_dir / "design_columns.txt").read_text().split()
    conv = X[:, cols.index("converter")]
    covs = X[:, [i for i, c in enumerate(cols) if c not in ("converter", "nonconverter")]]
    n = X.shape[0]
    glob = np.column_stack([np.ones(n), conv, covs])   # intercept, group, covariates
    g_col = 1
    if model == "A":
        M = glob[None, :, :]
    else:
        pv = [load_4d(glm_dir / f"{hemi}.{m}_slope.4d.mgh")[cortex] for m in ("thickness", "curv")]
        pv = np.stack(pv, axis=-1)                       # nv_c x n x 2
        pv = pv - pv.mean(axis=1, keepdims=True)
        M = np.concatenate([np.broadcast_to(glob, (pv.shape[0], n, glob.shape[1])), pv], axis=-1)
    keep = [i for i in range(M.shape[-1]) if i != g_col]
    Z = M[..., keep]
    return M, g_col, Z


def pinv_batch(A: np.ndarray):
    """(A'A)^-1 A' and diag((A'A)^-1) for a stack of designs A (nv x n x p)."""
    AtA = np.matmul(np.swapaxes(A, -1, -2), A)
    inv = np.linalg.inv(AtA)
    pinv = np.matmul(inv, np.swapaxes(A, -1, -2))
    return pinv, np.diagonal(inv, axis1=-2, axis2=-1)


def fit_t(M, pinv, g_var, g_col, Y):
    """t for column g_col of a batched OLS fit of Y (nv x n)."""
    n, p = M.shape[-2], M.shape[-1]
    beta = np.matmul(pinv, Y[..., None])[..., 0]               # nv x p
    resid = Y - np.matmul(M, beta[..., None])[..., 0]
    sigma2 = (resid * resid).sum(axis=-1) / (n - p)
    se = np.sqrt(sigma2 * g_var[..., g_col])
    return beta[..., g_col] / se, beta[..., g_col]


# ----------------------------------------------------------------------------
# permutation worker (module-level state shared through fork)
# ----------------------------------------------------------------------------
_S: dict = {}


def _perm_chunk(args):
    seed, start, count = args
    S = _S
    out = np.empty(count)
    n = S["eZ"].shape[1]
    for k in range(count):
        rng = np.random.default_rng([seed, start + k])
        perm = rng.permutation(n)
        t_perm, _ = fit_t(S["M"], S["pinv"], S["g_var"], S["g_col"], S["eZ"][:, perm])
        supra = np.zeros(S["nv"], dtype=bool)
        supra[S["cidx"]] = np.abs(t_perm) > S["tcrit"]
        out[k] = max_cluster_area(supra, S["adj"], S["area"])
    return start, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm_dir", required=True)
    ap.add_argument("--model", choices=["A", "B"], required=True)
    ap.add_argument("--hemi", choices=["lh", "rh"], required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fs_subjects_dir", default=os.environ.get("SUBJECTS_DIR", ""))
    ap.add_argument("--nperm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cft", type=float, default=0.01, help="cluster-forming p (two-sided)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    glm_dir, out = Path(args.glm_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fsd = Path(args.fs_subjects_dir)
    t0 = time.time()

    adj, area, cortex, aparc, aparc_names = load_mesh(fsd, args.hemi)
    nv = cortex.size
    cidx = np.flatnonzero(cortex)
    Y = load_4d(glm_dir / f"{args.hemi}.gbsc_slope.4d.mgh")[cortex]
    M, g_col, Z = build_design(glm_dir, args.model, args.hemi, cortex)
    n, p = M.shape[-2], M.shape[-1]
    pinv, g_var = pinv_batch(M)
    pinvZ, _ = pinv_batch(Z)
    df = n - p
    tcrit = float(t_dist.isf(args.cft / 2, df))

    # observed fit
    t_obs, beta_obs = fit_t(M, pinv, g_var, g_col, Y)
    p_obs = 2 * t_dist.sf(np.abs(t_obs), df)
    supra = np.zeros(nv, dtype=bool)
    supra[cidx] = np.abs(t_obs) > tcrit
    lab, k = clusters(supra, adj)

    def to_full(vals, fill=0.0):
        a = np.full(nv, fill); a[cidx] = vals; return a
    def save(name, vals):
        nib.save(nib.MGHImage(vals.astype(np.float32).reshape(-1, 1, 1), np.eye(4)),
                 str(out / name))
    save("t.mgh", to_full(t_obs))
    save("beta.mgh", to_full(beta_obs))
    save("sig.mgh", to_full(-np.log10(np.clip(p_obs, 1e-300, 1)) * np.sign(t_obs)))
    save("cluster_id.mgh", (lab + 1).astype(float))     # 0 = none, 1..k

    rows = []
    t_full = to_full(t_obs)
    for c in range(k):
        vs = np.flatnonzero(lab == c)
        peak = vs[np.argmax(np.abs(t_full[vs]))]
        rows.append({"hemi": args.hemi, "cluster": c + 1, "n_vertices": int(vs.size),
                     "area_mm2": float(area[vs].sum()), "peak_t": float(t_full[peak]),
                     "peak_vertex": int(peak), "sign": "pos" if t_full[peak] > 0 else "neg",
                     "peak_aparc": aparc_names[aparc[peak]] if aparc[peak] >= 0 else "unknown"})
    obs = pd.DataFrame(rows, columns=["hemi", "cluster", "n_vertices", "area_mm2", "peak_t",
                                      "peak_vertex", "sign", "peak_aparc"])
    obs.to_csv(out / "clusters_obs.csv", index=False)
    print(f"[{args.model} {args.hemi}] n={n} p={p} df={df} tcrit={tcrit:.3f} "
          f"observed clusters={k} (setup {time.time() - t0:.0f}s)", flush=True)

    # residuals of the reduced model, permuted below
    eZ = Y - np.matmul(Z, np.matmul(pinvZ, Y[..., None]))[..., 0]
    _S.update(dict(M=M, pinv=pinv, g_var=g_var, g_col=g_col, eZ=eZ, nv=nv, cidx=cidx,
                   tcrit=tcrit, adj=adj, area=area))

    chunk = max(1, args.nperm // (args.workers * 4))
    tasks = [(args.seed, s, min(chunk, args.nperm - s)) for s in range(0, args.nperm, chunk)]
    null = np.empty(args.nperm)
    t1 = time.time()
    with Pool(args.workers) as pool:
        for start, vals in pool.imap_unordered(_perm_chunk, tasks):
            null[start:start + len(vals)] = vals
    np.save(out / "perm_max_area.npy", null)

    info = dict(model=args.model, hemi=args.hemi, n=n, p=p, df=df, tcrit=tcrit, cft=args.cft,
                nperm=args.nperm, seed=args.seed, n_cortex_vertices=int(cidx.size),
                observed_clusters=k, perm_seconds=round(time.time() - t1, 1))
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[{args.model} {args.hemi}] {args.nperm} permutations in {time.time() - t1:.0f}s; "
          f"null max area median {np.median(null):.0f} mm2, 95th {np.percentile(null, 95):.0f}",
          flush=True)


if __name__ == "__main__":
    main()
