"""Longitudinal ComBat harmonization (Beer et al., 2020, NeuroImage 220:117129).

Spec section 4. Standard ComBat assumes independent observations, which is wrong
here: every subject contributes several scans, and the within-subject
correlation is the signal we care about. Longitudinal ComBat adds a subject
random intercept so that batch effects are estimated after accounting for
repeated measures, rather than partly absorbing them.

For feature v, scan j of subject i in batch b:

    y_ijv = alpha_v + X_ij beta_v + b_iv + gamma_bv + delta_bv * eps_ijv

alpha is the grand mean, X the biological covariates we want preserved (age at
scan, sex, diagnosis at scan), b_iv the subject random intercept, and gamma/delta
the additive and multiplicative batch effects to be removed. Batch parameters are
shrunk toward their across-batch means by empirical Bayes, which is what makes
ComBat stable when some scanners contribute few scans.

Fit and apply are separate so harmonization can be estimated on training folds
only and applied to held-out folds, per the spec's leakage warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _design(df: pd.DataFrame, covars: list[str]) -> np.ndarray:
    """Covariate design matrix with an intercept, categoricals dummy-coded."""
    parts = [np.ones((len(df), 1))]
    for c in covars:
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith("category"):
            d = pd.get_dummies(s.astype(str), drop_first=True).to_numpy(float)
            if d.shape[1]:
                parts.append(d)
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(float)
            v = np.nan_to_num(v, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0)
            parts.append(v.reshape(-1, 1))
    return np.hstack(parts)


def _fit_lmm_ols(y, X, subj_idx, n_subj, n_iter=15):
    """Fit y ~ X + (1|subject) by alternating least squares.

    A full REML mixed model per feature per fold is unnecessary here: with a
    single random intercept, alternating between fixed effects and shrunken
    subject means converges quickly and avoids a statsmodels fit for every one
    of the features on every fold.
    """
    b = np.zeros(n_subj)
    beta = np.zeros(X.shape[1])
    tau2, sig2 = 1.0, 1.0
    for _ in range(n_iter):
        beta, *_ = np.linalg.lstsq(X, y - b[subj_idx], rcond=None)
        r = y - X @ beta
        # Shrunken subject means (James-Stein style), the random-intercept BLUP.
        cnt = np.bincount(subj_idx, minlength=n_subj).astype(float)
        sums = np.bincount(subj_idx, weights=r, minlength=n_subj)
        shrink = tau2 / (tau2 + sig2 / np.maximum(cnt, 1))
        b = shrink * (sums / np.maximum(cnt, 1))
        resid = r - b[subj_idx]
        sig2 = max(float(np.mean(resid ** 2)), 1e-12)
        tau2 = max(float(np.mean(b ** 2)), 1e-12)
    return beta, b, np.sqrt(sig2)


def fit(df: pd.DataFrame, features: list[str], batch_col: str,
        covars: list[str], subject_col: str, eb: bool = True) -> dict:
    """Estimate harmonization parameters. Returns a dict for apply().

    Batch enters the model as a fixed effect estimated jointly with the subject
    random intercept, which is the point of the longitudinal variant. Estimating
    the random intercept first and reading batch off the residuals biases the
    batch effect toward zero, because a subject scanned entirely on one scanner
    has their batch offset absorbed into their own intercept.
    """
    batches = pd.Index(sorted(df[batch_col].astype(str).unique()))
    bcode = batches.get_indexer(df[batch_col].astype(str))
    subs = pd.Index(sorted(df[subject_col].unique()))
    scode = subs.get_indexer(df[subject_col])

    Xc = _design(df, covars)
    # Full dummy coding, no reference level; the batch block is centered after
    # fitting so that harmonization shifts batches to the grand mean rather than
    # to whichever level happened to be dropped.
    B = np.zeros((len(df), len(batches)))
    B[np.arange(len(df)), np.clip(bcode, 0, None)] = 1.0
    X = np.hstack([Xc, B])
    nc = Xc.shape[1]

    gamma = np.full((len(batches), len(features)), np.nan)
    delta = np.full((len(batches), len(features)), np.nan)
    betas, sigmas = {}, {}

    for vi, f in enumerate(features):
        y = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 20:
            continue
        coef, b, sigma = _fit_lmm_ols(y[ok], X[ok], scode[ok], len(subs))
        betas[f], sigmas[f] = coef[:nc], sigma

        g_raw = coef[nc:]
        w = np.bincount(bcode[ok], minlength=len(batches)).astype(float)
        g_raw = g_raw - np.average(g_raw, weights=np.maximum(w, 1e-9))
        gamma[:, vi] = g_raw / sigma

        resid = y[ok] - X[ok] @ coef - b[scode[ok]]
        for k in range(len(batches)):
            m = bcode[ok] == k
            if m.sum() >= 3:
                sd = float(resid[m].std(ddof=1)) / sigma
                delta[k, vi] = sd if sd > 0 else 1.0

    nb = np.bincount(bcode[bcode >= 0], minlength=len(batches)).astype(float)
    if eb:
        gamma, delta = _empirical_bayes(gamma, delta, nb)
    gamma[~np.isfinite(gamma)] = 0.0
    delta[~np.isfinite(delta) | (delta <= 0)] = 1.0

    return {"batches": batches, "features": features, "covars": covars,
            "batch_col": batch_col, "subject_col": subject_col,
            "gamma": gamma, "delta": delta, "betas": betas, "sigmas": sigmas}


def _empirical_bayes(gamma, delta, nb):
    """Shrink per-batch parameters toward the across-feature consensus.

    Following ComBat, the prior for batch b pools across features. The estimate
    gamma_hat is a mean of n_b standardized residuals, so its sampling variance
    is about 1/n_b, and the posterior weight must scale with batch size. Using a
    fixed variance of 1 instead collapses every feature onto the batch mean,
    which discards exactly the per-feature structure being corrected.
    """
    g, d = gamma.copy(), delta.copy()
    for k in range(g.shape[0]):
        row_g, row_d = g[k], d[k]
        okg, okd = np.isfinite(row_g), np.isfinite(row_d)
        n_k = max(float(nb[k]), 1.0)
        if okg.sum() > 1:
            gbar, tau2 = row_g[okg].mean(), row_g[okg].var(ddof=1)
            w = (n_k * tau2) / (n_k * tau2 + 1.0) if tau2 > 0 else 0.0
            g[k, okg] = w * row_g[okg] + (1 - w) * gbar
        if okd.sum() > 1:
            dbar, dvar = row_d[okd].mean(), row_d[okd].var(ddof=1)
            w = (n_k * dvar) / (n_k * dvar + 1.0) if dvar > 0 else 0.0
            d[k, okd] = w * row_d[okd] + (1 - w) * dbar
    d[~np.isfinite(d) | (d <= 0)] = 1.0
    g[~np.isfinite(g)] = 0.0
    return g, d


def apply(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove estimated batch effects. Unseen batches are left untouched."""
    out = df.copy()
    batches, features = params["batches"], params["features"]
    bcode = batches.get_indexer(df[params["batch_col"]].astype(str))
    seen = bcode >= 0
    bsafe = np.clip(bcode, 0, None)
    X = _design(df, params["covars"])

    subs = pd.Index(sorted(df[params["subject_col"]].unique()))
    scode = subs.get_indexer(df[params["subject_col"]])

    for vi, f in enumerate(features):
        if f not in params["betas"]:
            continue
        y = pd.to_numeric(df[f], errors="coerce").to_numpy(float)
        beta, sigma = params["betas"][f], params["sigmas"][f]
        g = np.where(seen, params["gamma"][bsafe, vi], 0.0) * sigma
        d = np.where(seen, params["delta"][bsafe, vi], 1.0)

        fixed = X @ beta
        # Strip the batch offset first, then take the subject mean, so the
        # intercept is not itself contaminated by which scanner the subject used.
        r = y - fixed - g
        ok = np.isfinite(r)
        cnt = np.bincount(scode[ok], minlength=len(subs)).astype(float)
        sums = np.bincount(scode[ok], weights=r[ok], minlength=len(subs))
        b = np.divide(sums, np.maximum(cnt, 1), where=cnt > 0)

        out[f] = fixed + b[scode] + (r - b[scode]) / d
    return out


def batch_diagnostics(df: pd.DataFrame, features: list[str],
                      batch_col: str) -> pd.DataFrame:
    """Kruskal-Wallis test of each feature against batch (spec section 4.3)."""
    from scipy import stats
    rows = []
    for f in features:
        groups = [pd.to_numeric(g[f], errors="coerce").dropna().to_numpy()
                  for _, g in df.groupby(batch_col) if g[f].notna().sum() >= 3]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            h, p = stats.kruskal(*groups)
        except ValueError:
            continue
        rows.append({"feature": f, "H": float(h), "p": float(p)})
    out = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    if len(out):
        m = len(out)
        raw = out["p"] * m / (out.index + 1)
        out["q"] = raw[::-1].cummin()[::-1].clip(upper=1.0)
    return out
