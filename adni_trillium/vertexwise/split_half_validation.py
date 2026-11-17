"""Select the cluster in one half of the subjects, measure it in the other.

Every result so far shares one flaw: the cluster is chosen and evaluated on the
same 416 subjects. Within-sample effect sizes at a selected cluster are biased
upward no matter how the stratum is sliced, so the field-strength strata cannot
settle whether the effect is real either, being subsets of the selection
sample.

This removes the circularity. Subjects are split in half, stratified on outcome
and field strength. The cluster is defined in the first half exactly as the
pooled analysis defines it, by fitting the model, thresholding at the same
cluster-forming p and taking the largest suprathreshold cluster inside the
AD-signature mask across both hemispheres. Its mean slope is then measured in
the second half, which had no part in choosing it, and the group difference
there is tested once.

A held-out effect size near the within-sample one means the cluster is real. A
held-out effect size near zero means the within-sample one was selection bias,
and no amount of covariate adjustment in the pooled sample would have revealed
it. No permutations are needed: the held-out test is a single scalar contrast.

Usage:
    python3 split_half_validation.py --glm_dir .../glm_grad_all --slopes_root .../slopes_prec \
        --model B --nsplit 200 --fs_subjects_dir $FREESURFER_HOME/subjects
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from vertex_glm import clusters, fit_t, load_4d, load_mesh, pinv_batch, suprathreshold

SIGNATURE = ("entorhinal", "parahippocampal", "inferiortemporal", "middletemporal",
             "precuneus", "posteriorcingulate", "fusiform")
HEMIS = ("lh", "rh")


def bounded_weights(prec: np.ndarray, max_ratio: float = 4.0) -> np.ndarray:
    p5, p95 = np.percentile(prec, [5, 95])
    ratio = p95 / p5
    alpha = min(1.0, np.log(max_ratio) / np.log(ratio)) if ratio > max_ratio else 1.0
    w = prec ** alpha
    half, med = np.sqrt(max_ratio), np.median(prec ** alpha)
    w = np.clip(w, med / half, med * half)
    return w / w.mean()


def design(conv, cov, pv, w):
    """Stacked per-vertex design, weighted. pv is nv x n x k or None."""
    n = len(conv)
    glob = np.column_stack([np.ones(n), conv, cov])
    if pv is None:
        M = glob[None, :, :]
    else:
        pv = pv - pv.mean(axis=1, keepdims=True)
        M = np.concatenate(
            [np.broadcast_to(glob, (pv.shape[0], n, glob.shape[1])), pv], axis=-1)
    return M * np.sqrt(w)[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm_dir", required=True)
    ap.add_argument("--slopes_root", required=True)
    ap.add_argument("--model", choices=["A", "B"], default="B")
    ap.add_argument("--y_measure", default="gbsc")
    ap.add_argument("--cft", type=float, default=0.01)
    ap.add_argument("--tail", default="neg")
    ap.add_argument("--nsplit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fs_subjects_dir", required=True)
    args = ap.parse_args()

    glm, slopes = Path(args.glm_dir), Path(args.slopes_root)
    subj = pd.read_csv(glm / "subjects_used.csv")
    prec = pd.read_csv(slopes / "slope_precision.csv").set_index("subject")
    precision = subj["subject"].map(prec["slope_precision"]).to_numpy(dtype=float)
    conv_all = subj["event"].to_numpy(dtype=float)
    f3_all = subj["field_3t"].to_numpy(dtype=float)
    cov_all = subj[["age_at_landmark", "female"]].to_numpy(dtype=float)
    n = len(subj)

    mesh, Y, PV = {}, {}, {}
    for hemi in HEMIS:
        adj, area, cortex, aparc, names = load_mesh(Path(args.fs_subjects_dir), hemi)
        codes = [names.index(s) for s in SIGNATURE]
        mask = cortex & np.isin(aparc, codes)
        mesh[hemi] = (adj, area, mask)
        Y[hemi] = load_4d(glm / f"{hemi}.{args.y_measure}_slope.4d.mgh")[mask]
        if args.model == "B":
            PV[hemi] = np.stack(
                [load_4d(glm / f"{hemi}.{m}_slope.4d.mgh")[mask]
                 for m in ("thickness", "curv")], axis=-1)
        print(f"[{hemi}] signature mask {int(mask.sum())} vertices, "
              f"{area[mask].sum():.0f} mm2", flush=True)

    rng = np.random.default_rng(args.seed)
    # stratify on the four outcome-by-field cells so both halves match on both
    cells = [np.flatnonzero((conv_all == c) & (f3_all == f))
             for c in (0.0, 1.0) for f in (0.0, 1.0)]
    rows = []
    for split in range(args.nsplit):
        A = np.concatenate([rng.permutation(ix)[: len(ix) // 2] for ix in cells])
        B = np.setdiff1d(np.arange(n), A)
        wA, wB = bounded_weights(precision[A]), bounded_weights(precision[B])

        best = None
        for hemi in HEMIS:
            adj, area, mask = mesh[hemi]
            cidx = np.flatnonzero(mask)
            M = design(conv_all[A], cov_all[A] - cov_all[A].mean(axis=0),
                       PV[hemi][:, A, :] if args.model == "B" else None, wA)
            pinv, gvar = pinv_batch(M)
            t, _ = fit_t(M, pinv, gvar, 1, Y[hemi][:, A] * np.sqrt(wA))
            df = len(A) - M.shape[-1]
            tcrit = float(stats.t.isf(args.cft / (2 if args.tail == "two" else 1), df))
            supra = np.zeros(mask.size, dtype=bool)
            supra[cidx] = suprathreshold(t, tcrit, args.tail)
            lab, k = clusters(supra, adj)
            if k == 0:
                continue
            areas = np.bincount(lab[lab >= 0], weights=area[lab >= 0], minlength=k)
            c = int(areas.argmax())
            if best is None or areas[c] > best[0]:
                best = (float(areas[c]), hemi, lab == c, mask)

        if best is None:
            rows.append({"split": split, "area_mm2": 0.0, "hemi": "", "d_train": np.nan,
                         "d_test": np.nan, "t_test": np.nan, "p_test": np.nan})
            continue
        area_mm2, hemi, sel, mask = best
        roi = Y[hemi][sel[mask], :].mean(axis=0)        # per-subject mean inside the cluster

        def contrast(ix, w):
            X = np.column_stack([np.ones(len(ix)), conv_all[ix],
                                 cov_all[ix] - cov_all[ix].mean(axis=0)])
            sw = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(X * sw[:, None], roi[ix] * sw, rcond=None)
            r = roi[ix] * sw - (X * sw[:, None]) @ beta
            dof = len(ix) - X.shape[1]
            s2 = (r @ r) / dof
            se = np.sqrt(s2 * np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))[1, 1])
            a, b = roi[ix][conv_all[ix] == 1], roi[ix][conv_all[ix] == 0]
            sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                         / (len(a) + len(b) - 2))
            return beta[1] / se, 2 * stats.t.sf(abs(beta[1] / se), dof), (a.mean() - b.mean()) / sp

        t_tr, _, d_tr = contrast(A, wA)
        t_te, p_te, d_te = contrast(B, wB)
        rows.append({"split": split, "area_mm2": area_mm2, "hemi": hemi,
                     "d_train": d_tr, "t_train": t_tr,
                     "d_test": d_te, "t_test": t_te, "p_test": p_te})
        if (split + 1) % 25 == 0:
            print(f"  {split + 1}/{args.nsplit}", flush=True)

    r = pd.DataFrame(rows)
    out = glm / f"split_half_model{args.model}.csv"
    r.to_csv(out, index=False)
    ok = r.dropna(subset=["d_test"])
    print(f"\n{len(ok)} of {args.nsplit} splits formed a cluster in the training half")
    print(f"training-half  d: median {ok['d_train'].median():+.3f} "
          f"[{ok['d_train'].quantile(.025):+.3f}, {ok['d_train'].quantile(.975):+.3f}]")
    print(f"held-out half  d: median {ok['d_test'].median():+.3f} "
          f"[{ok['d_test'].quantile(.025):+.3f}, {ok['d_test'].quantile(.975):+.3f}]")
    print(f"held-out d in the hypothesised direction: "
          f"{100 * (ok['d_test'] < 0).mean():.0f}% of splits")
    print(f"held-out p < 0.05 in that direction: "
          f"{100 * ((ok['p_test'] < 0.05) & (ok['d_test'] < 0)).mean():.0f}% of splits")
    print(f"shrinkage, held-out over training: "
          f"{ok['d_test'].median() / ok['d_train'].median():.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
