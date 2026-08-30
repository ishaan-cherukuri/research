"""Calibration and time-dependent discrimination for the NI: Reports revision.

The manuscript claims that clinical covariates are sufficient. A sufficiency
claim is stronger if the covariate model is also calibrated, so this script adds
the integrated Brier score alongside the time-dependent AUC that the methods
already promise.

Brier scores need a survival function rather than a risk score, so the models
here are Cox with L2 penalization and Random Survival Forest, both of which
produce one directly. XGBoost AFT returns a predicted time and is left out: it
is the flagship for discrimination, not for calibration, and inventing a
survival curve for it would be a modelling choice presented as a measurement.

Folds, preprocessing and seed match run_spec_models exactly.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_spec_models import (SEED, cindex, feature_sets, preprocess, surv_y)

warnings.filterwarnings("ignore")

CONFIGS = [
    ("F0_covariates", "Clinical covariates"),
    ("F8_cov_stdmri", "Covariates + standard MRI"),
    ("F9b_cov_regional_stdmri", "Covariates + regional BSC + standard MRI"),
]
HORIZONS = [1.0, 2.0, 3.0]


def fit_surv(model, Xtr, ytr, Xte, times):
    """Return (risk score, survival probabilities at `times`) for the test rows."""
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    if model == "cox_l2":
        m = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
    else:
        m = RandomSurvivalForest(n_estimators=1000, min_samples_split=10,
                                 min_samples_leaf=15, max_features="sqrt",
                                 random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    fns = m.predict_survival_function(Xte)
    surv = np.row_stack([[fn(t) for t in times] for fn in fns])
    return m.predict(Xte), surv


def run_config(df, cols, model):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    from sksurv.metrics import (cumulative_dynamic_auc, integrated_brier_score,
                                brier_score)

    per_fold = []
    for a, b in skf.split(df, df["event"]):
        tr, te = df.iloc[a], df.iloc[b]
        Xtr, Xte = preprocess(tr, te, cols)
        ytr, yte = surv_y(tr), surv_y(te)

        # Brier scores are only defined inside the follow-up both folds cover.
        tmax = min(ytr["time"].max(), yte["time"][yte["event"]].max())
        hz = [h for h in HORIZONS if h < tmax]
        if not hz:
            continue
        grid = np.linspace(min(hz) * 0.5, max(hz), 40)
        grid = grid[grid < tmax]

        risk, surv_grid = fit_surv(model, Xtr, ytr, Xte, grid)
        _, surv_hz = fit_surv(model, Xtr, ytr, Xte, hz)

        rec = {"cindex": cindex(yte, risk),
               "ibs": float(integrated_brier_score(ytr, yte, surv_grid, grid))}
        _, bs = brier_score(ytr, yte, surv_hz, np.array(hz))
        for h, v in zip(hz, bs):
            rec[f"brier_{h:g}y"] = float(v)
        try:
            auc, _ = cumulative_dynamic_auc(ytr, yte, risk, np.array(hz))
            for h, v in zip(hz, auc):
                rec[f"auc_{h:g}y"] = float(v)
        except Exception:
            pass
        per_fold.append(rec)

    keys = sorted(set().union(*[set(r) for r in per_fold]))
    return {k: float(np.mean([r[k] for r in per_fold if k in r])) for k in keys} | \
           {"n_folds": len(per_fold)}


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    ap.add_argument("--cohort", default=str(root / "results/spec_v3_harmonized/spec_cohort.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/nireports"))
    args = ap.parse_args()

    df = pd.read_csv(args.cohort)
    fs = pd.to_numeric(df.get("field_strength_bl"), errors="coerce")
    df["field_strength_bin"] = np.where(fs < 2.25, 0.0, 1.0)
    sets = feature_sets(df)

    rows = []
    for model in ["cox_l2", "rsf"]:
        for key, label in CONFIGS:
            cols = [c for c in sets[key] if c in df.columns]
            r = run_config(df, cols, model)
            r |= {"model": model, "config": key, "label": label,
                  "n_features": len(cols)}
            rows.append(r)
            print(f"{model:8s} {label:42s} C={r['cindex']:.3f} "
                  f"IBS={r['ibs']:.4f} "
                  f"AUC1y={r.get('auc_1y', float('nan')):.3f} "
                  f"AUC2y={r.get('auc_2y', float('nan')):.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    front = ["model", "config", "label", "n_features", "cindex", "ibs"]
    d = d[front + [c for c in d.columns if c not in front]]
    d.to_csv(out / "calibration.csv", index=False)
    with open(out / "calibration.json", "w") as f:
        json.dump({"seed": SEED, "horizons": HORIZONS, "results": rows}, f, indent=2)
    print(f"\nwrote {out/'calibration.csv'}")


if __name__ == "__main__":
    main()
