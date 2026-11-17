"""Vertex-wise group GLM on fsaverage6 with cluster-wise permutation correction.

mri_glmfit cannot permute a design with per-vertex regressors, which is what
Model B needs, so both models run here and Model A is cross-checked against
mri_glmfit-sim in slurm/03_glm.sh.

    Model A  y ~ 1 + converter + covariates
    Model B  Model A + thickness_slope and curv_slope at the same vertex

--mask_labels restricts the search space to named aparc parcels, which lowers
the critical cluster area because the permutation maximum is taken over less
cortex. Only worth it for a region set fixed in advance.

--tail neg or pos tests one side. --weights applies subject weights from
make_design.py, by scaling both sides of the fit by their square root, so
everything downstream including the permutation runs in the whitened space.

Null distribution follows Freedman-Lane (1983; Winkler 2014): residuals of the
reduced model are permuted, one shuffle per iteration shared across vertices,
the full model refitted, largest suprathreshold cluster area recorded. The same
seed on both hemispheres gives identical permutation sequences, so
combine_clusters.py can take the per-iteration max across hemispheres and
correct over both surfaces at once.

Usage:
    python3 vertex_glm.py --glm_dir GLM --model B --hemi lh --out_dir GLM/pyB_lh \
        --nperm 10000 --seed 0 --cft 0.01 --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.freesurfer import read_annot, read_geometry, read_label
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import t as t_dist

TRG = "fsaverage6"


def load_mesh(fs_subjects_dir: Path, hemi: str):
    surf = fs_subjects_dir / TRG / "surf" / f"{hemi}.white"
    coords, faces = read_geometry(str(surf))
    nv = coords.shape[0]
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


def load_4d(path: Path) -> np.ndarray:
    d = np.asarray(nib.load(str(path)).dataobj, dtype=np.float64)
    return d.reshape(d.shape[0], -1)          # nv x nsubj


def build_design(glm_dir: Path, model: str, hemi: str, mask: np.ndarray,
                 w: np.ndarray | None = None):
    """Return M (nv_m x n x p), the index of the converter column, and the
    reduced-model design Z (nv_m x n x pz). Model A designs are broadcast from
    a single matrix. Only in-mask vertices are kept."""
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
        pv = [load_4d(glm_dir / f"{hemi}.{m}_slope.4d.mgh")[mask] for m in ("thickness", "curv")]
        pv = np.stack(pv, axis=-1)                       # nv_c x n x 2
        pv = pv - pv.mean(axis=1, keepdims=True)
        M = np.concatenate([np.broadcast_to(glob, (pv.shape[0], n, glob.shape[1])), pv], axis=-1)
    if w is not None:
        M = M * np.sqrt(w)[:, None]
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


# Module-level so Pool workers inherit it through fork rather than pickling: the
# permutation state is hundreds of megabytes and pickling it per task would cost
# more than the fit. The context is named rather than left to the default, which
# is spawn on macOS and becomes forkserver on Linux in Python 3.14; under either
# of those the workers start with this dict empty.
_S: dict = {}


def suprathreshold(t: np.ndarray, tcrit: float, tail: str) -> np.ndarray:
    if tail == "neg":
        return t < -tcrit
    if tail == "pos":
        return t > tcrit
    return np.abs(t) > tcrit


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
        supra[S["cidx"]] = suprathreshold(t_perm, S["tcrit"], S["tail"])
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
    ap.add_argument("--cft", type=float, default=0.01, help="cluster-forming p")
    ap.add_argument("--tail", choices=["two", "neg", "pos"], default="two",
                    help="neg tests converters blurring faster, the hypothesised direction")
    ap.add_argument("--y_measure", default="gbsc")
    ap.add_argument("--mask_labels", default="",
                    help="comma-separated aparc parcel names to restrict the search to; "
                         "empty searches the whole cortex")
    ap.add_argument("--weights", default="",
                    help="path to make_design.py's weights.txt for a weighted fit")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    glm_dir, out = Path(args.glm_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fsd = Path(args.fs_subjects_dir)
    t0 = time.time()

    adj, area, cortex, aparc, aparc_names = load_mesh(fsd, args.hemi)
    nv = cortex.size
    mask = cortex.copy()
    want = [v.strip() for v in args.mask_labels.split(",") if v.strip()]
    if want:
        unknown = [v for v in want if v not in aparc_names]
        if unknown:
            raise SystemExit(f"not aparc parcel names: {unknown}")
        codes = [aparc_names.index(v) for v in want]
        mask &= np.isin(aparc, codes)
        print(f"[{args.hemi}] mask {', '.join(want)}: {int(mask.sum())} vertices, "
              f"{area[mask].sum():.0f} mm2 of {area[cortex].sum():.0f} cortex")
    cidx = np.flatnonzero(mask)
    w = np.loadtxt(args.weights) if args.weights else None
    Y = load_4d(glm_dir / f"{args.hemi}.{args.y_measure}_slope.4d.mgh")[mask]
    M, g_col, Z = build_design(glm_dir, args.model, args.hemi, mask, w)
    if w is not None:
        Y = Y * np.sqrt(w)
    n, p = M.shape[-2], M.shape[-1]
    pinv, g_var = pinv_batch(M)
    pinvZ, _ = pinv_batch(Z)
    df = n - p
    tcrit = float(t_dist.isf(args.cft / (2 if args.tail == "two" else 1), df))

    # observed fit
    t_obs, beta_obs = fit_t(M, pinv, g_var, g_col, Y)
    p_obs = 2 * t_dist.sf(np.abs(t_obs), df)
    supra = np.zeros(nv, dtype=bool)
    supra[cidx] = suprathreshold(t_obs, tcrit, args.tail)
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
                   tcrit=tcrit, tail=args.tail, adj=adj, area=area))

    chunk = max(1, args.nperm // (args.workers * 4))
    tasks = [(args.seed, s, min(chunk, args.nperm - s)) for s in range(0, args.nperm, chunk)]
    null = np.empty(args.nperm)
    t1 = time.time()
    with get_context("fork").Pool(args.workers) as pool:
        for start, vals in pool.imap_unordered(_perm_chunk, tasks):
            null[start:start + len(vals)] = vals
    np.save(out / "perm_max_area.npy", null)

    info = dict(model=args.model, hemi=args.hemi, n=n, p=p, df=df, tcrit=tcrit, cft=args.cft,
                tail=args.tail, y_measure=args.y_measure, mask_labels=want,
                weighted=bool(args.weights), nperm=args.nperm, seed=args.seed,
                n_search_vertices=int(cidx.size),
                search_area_mm2=round(float(area[mask].sum()), 1),
                observed_clusters=k, perm_seconds=round(time.time() - t1, 1))
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(f"[{args.model} {args.hemi}] {args.nperm} permutations in {time.time() - t1:.0f}s; "
          f"null max area median {np.median(null):.0f} mm2, 95th {np.percentile(null, 95):.0f}",
          flush=True)


if __name__ == "__main__":
    main()
