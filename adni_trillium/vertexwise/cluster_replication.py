"""Is a surviving cluster's effect present in both field-strength strata?

Cluster extent is not an effect size. It mixes the size of the group
difference with how spatially coherent the noise happens to be, so a cluster
that shrinks in a subsample has not necessarily weakened. This reduces the
cluster to one number per subject, the mean slope inside it, and asks the
group question once per stratum. One test each, no multiple comparisons, and
the strata are compared on the same footing.

The cluster is selected on the pooled sample, so the pooled estimate here is
inflated by that selection and is reported only as the reference the strata are
measured against. The strata inherit the same selection, which is what makes
them comparable to each other; the question is not whether either is unbiased
but whether the effect lives in both or only in the pooled sample.

The interaction model is the direct test. If gBSC slope differs by field
strength and field strength differs by outcome, a pooled group effect can be
manufactured with no within-stratum effect at all, and the interaction term is
where that shows up.

Usage:
    python3 cluster_replication.py --glm_dir .../glm_grad_all --model B --hemi rh \
        --cluster 1 --slopes_root .../slopes_prec --subjects cohort/vertex_subjects.csv \
        --scans cohort/vertex_scans.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

from make_design import COVMAP, field_3t


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float64).reshape(-1)


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted least squares. Returns beta, t and p for every column."""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(Xw.T @ Xw)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    return beta, t, 2 * stats.t.sf(np.abs(t), dof), dof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glm_dir", required=True, help="the pooled run the cluster comes from")
    ap.add_argument("--model", default="B")
    ap.add_argument("--hemi", default="rh")
    ap.add_argument("--cluster", type=int, default=1)
    ap.add_argument("--slopes_root", required=True)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--scans", required=True)
    ap.add_argument("--y_measure", default="gbsc")
    args = ap.parse_args()

    glm, slopes = Path(args.glm_dir), Path(args.slopes_root)
    cid = load(glm / f"py{args.model}_{args.hemi}" / "cluster_id.mgh").astype(int)
    mask = cid == args.cluster
    if not mask.any():
        raise SystemExit(f"cluster {args.cluster} not found in {glm}")

    subj = pd.read_csv(args.subjects)
    scans = pd.read_csv(args.scans)
    done = pd.read_csv(slopes / "subjects_done.csv")["subject"]
    subj = subj[subj["subject"].isin(done)].sort_values("subject").reset_index(drop=True)
    subj["field_3t"] = field_3t(subj, scans)

    subj["roi_slope"] = [
        float(load(slopes / s / f"{args.hemi}.{args.y_measure}_slope.mgh")[mask].mean())
        for s in subj["subject"]]
    prec = pd.read_csv(slopes / "slope_precision.csv").set_index("subject")
    subj["precision"] = subj["subject"].map(prec["slope_precision"])

    print(f"cluster {args.cluster} of model {args.model} {args.hemi} in {glm.name}: "
          f"{int(mask.sum())} vertices")
    print(f"{len(subj)} subjects, {int(subj['event'].sum())} converters\n")

    def weights(d: pd.DataFrame) -> np.ndarray:
        # the same bounded precision weights make_design.py builds
        w = d["precision"].to_numpy(dtype=float)
        p5, p95 = np.percentile(w, [5, 95])
        ratio = p95 / p5
        alpha = min(1.0, np.log(4.0) / np.log(ratio)) if ratio > 4.0 else 1.0
        w = w ** alpha
        half, med = 2.0, np.median(w ** 1.0)
        w = np.clip(w, med / half, med * half)
        return w / w.mean()

    strata = [("all", subj),
              ("field-stable", subj[subj["field_strength_changed"].fillna(1) == 0]),
              ("3T only", subj[subj["field_3t"] == 1]),
              ("1.5T only", subj[subj["field_3t"] == 0])]

    print(f"{'stratum':14s} {'n':>5s} {'conv':>5s} {'beta':>11s} {'t':>7s} "
          f"{'p':>9s} {'d':>7s}")
    for name, d in strata:
        d = d.reset_index(drop=True)
        conv = d["event"].to_numpy(dtype=float)
        cov = d[[COVMAP[c] for c in ("age", "female")]].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(d)), conv, cov - cov.mean(axis=0)])
        y = d["roi_slope"].to_numpy(dtype=float)
        beta, t, p, _ = wls(X, y, weights(d))
        a, b = y[conv == 1], y[conv == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
        d_coh = (a.mean() - b.mean()) / sp
        print(f"{name:14s} {len(d):5d} {int(conv.sum()):5d} {beta[1]:11.3e} "
              f"{t[1]:7.2f} {p[1]:9.4f} {d_coh:7.3f}")

    # group x field interaction on the pooled sample
    conv = subj["event"].to_numpy(dtype=float)
    f3 = subj["field_3t"].to_numpy(dtype=float)
    cov = subj[[COVMAP[c] for c in ("age", "female")]].to_numpy(dtype=float)
    cc, ff = conv - conv.mean(), f3 - f3.mean()
    X = np.column_stack([np.ones(len(subj)), cc, ff, cc * ff, cov - cov.mean(axis=0)])
    y = subj["roi_slope"].to_numpy(dtype=float)
    beta, t, p, dof = wls(X, y, weights(subj))
    names = ["intercept", "converter", "field_3t", "converter x field_3t", "age", "female"]
    print(f"\ninteraction model (pooled, df={dof})")
    for i, nm in enumerate(names):
        print(f"  {nm:22s} beta {beta[i]:11.3e}  t {t[i]:6.2f}  p {p[i]:.4f}")

    print(f"\nfield strength by outcome: 3T is "
          f"{100 * subj.loc[subj['event'] == 1, 'field_3t'].mean():.1f}% of converters "
          f"and {100 * subj.loc[subj['event'] == 0, 'field_3t'].mean():.1f}% of "
          f"non-converters")


if __name__ == "__main__":
    main()
