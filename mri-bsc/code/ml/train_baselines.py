
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from code.ml.train_rsf_combined import (
    T1_FEATURE_COLS,
    load_and_merge,
    select_slope_features_train_only,
    signed_log1p_df,
    fit_winsor_limits,
    apply_winsor_limits,
    minmax_scale_train_test,
)

def aft_cv(df, slope_cols, y_all, top_k, penalize_regex, penalty_factor,
           fitter_cls, model_name,
           winsor_low=0.01, winsor_high=0.99):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, test_scores = [], []
    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())

    for fold_i, (tr_idx, te_idx) in enumerate(
            skf.split(df.index, y_all["event"]), 1):
        tr_idx = df.index[tr_idx]
        te_idx = df.index[te_idx]
        y_tr = y_all.loc[tr_idx]
        y_te = y_all.loc[te_idx]

        sel = select_slope_features_train_only(
            X_slopes_all.loc[tr_idx], top_k, penalize_regex,
            penalty_factor, winsor_low, winsor_high, quiet=True,
        )

        X_tr_log = signed_log1p_df(X_slopes_all.loc[tr_idx, sel])
        X_te_log = signed_log1p_df(X_slopes_all.loc[te_idx, sel])
        lims = fit_winsor_limits(X_tr_log, winsor_low, winsor_high)
        X_tr_r = apply_winsor_limits(X_tr_log, lims)
        X_te_r = apply_winsor_limits(X_te_log, lims)
        X_tr_s, X_te_s, _ = minmax_scale_train_test(X_tr_r, X_te_r)

        df_tr = X_tr_s.copy()
        df_tr["time_years"] = y_tr["time_years"].values
        df_tr["event"]      = y_tr["event"].values
        df_te = X_te_s.copy()
        df_te["time_years"] = y_te["time_years"].values
        df_te["event"]      = y_te["event"].values

        model = fitter_cls(penalizer=0.01)
        model.fit(df_tr, duration_col="time_years", event_col="event")

        raw_pred_tr = model.predict_median(df_tr).values.astype(float)
        raw_pred_te = model.predict_median(df_te).values.astype(float)

        for arr in (raw_pred_tr, raw_pred_te):
            finite_max = np.nanmax(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 1.0
            arr[~np.isfinite(arr)] = finite_max

        pred_tr = -raw_pred_tr
        pred_te = -raw_pred_te

        c_tr = float(concordance_index_censored(
            y_tr["event"].astype(bool).values, y_tr["time_years"].values, pred_tr)[0])
        c_te = float(concordance_index_censored(
            y_te["event"].astype(bool).values, y_te["time_years"].values, pred_te)[0])

        train_scores.append(c_tr)
        test_scores.append(c_te)
        print(f"  [{model_name}] Fold {fold_i}  train={c_tr:.4f}  test={c_te:.4f}  "
              f"gap={c_tr-c_te:.4f}")

    gaps = [tr - te for tr, te in zip(train_scores, test_scores)]
    print(f"  [{model_name}] Mean train={np.mean(train_scores):.4f}  "
          f"test={np.mean(test_scores):.4f}  gap={np.mean(gaps):.4f}\n")
    return {"train": train_scores, "test": test_scores, "gap": gaps}

def rsf_slopes_cv(df, slope_cols, y_all, top_k, penalize_regex, penalty_factor,
                  n_estimators=1000, winsor_low=0.01, winsor_high=0.99):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, test_scores = [], []
    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())

    for fold_i, (tr_idx, te_idx) in enumerate(
            skf.split(df.index, y_all["event"]), 1):
        tr_idx = df.index[tr_idx]
        te_idx = df.index[te_idx]
        y_tr = y_all.loc[tr_idx]
        y_te = y_all.loc[te_idx]

        sel = select_slope_features_train_only(
            X_slopes_all.loc[tr_idx], top_k, penalize_regex,
            penalty_factor, winsor_low, winsor_high, quiet=True,
        )

        X_tr_log = signed_log1p_df(X_slopes_all.loc[tr_idx, sel])
        X_te_log = signed_log1p_df(X_slopes_all.loc[te_idx, sel])
        lims     = fit_winsor_limits(X_tr_log, winsor_low, winsor_high)
        X_tr_r   = apply_winsor_limits(X_tr_log, lims)
        X_te_r   = apply_winsor_limits(X_te_log, lims)
        X_tr_s, X_te_s, _ = minmax_scale_train_test(X_tr_r, X_te_r)

        y_tr_surv = Surv.from_dataframe("event", "time_years", y_tr)
        y_te_surv = Surv.from_dataframe("event", "time_years", y_te)

        rsf = RandomSurvivalForest(
            n_estimators=n_estimators, min_samples_split=10,
            min_samples_leaf=15, max_features="sqrt",
            n_jobs=-1, random_state=42, verbose=0,
        )
        rsf.fit(X_tr_s, y_tr_surv)

        c_tr = float(rsf.score(X_tr_s, y_tr_surv))
        c_te = float(rsf.score(X_te_s, y_te_surv))
        train_scores.append(c_tr)
        test_scores.append(c_te)
        print(f"  [RSF-slopes] Fold {fold_i}  train={c_tr:.4f}  test={c_te:.4f}  "
              f"gap={c_tr-c_te:.4f}")

    gaps = [tr - te for tr, te in zip(train_scores, test_scores)]
    print(f"  [RSF-slopes] Mean train={np.mean(train_scores):.4f}  "
          f"test={np.mean(test_scores):.4f}  gap={np.mean(gaps):.4f}\n")
    return {"train": train_scores, "test": test_scores, "gap": gaps}

MODEL_COLORS = {
    "Weibull-20":         "#a87090",
    "LogLogistic-20":     "#c07060",
    "LogNormal-20":       "#c4a030",
    "RSF-20\n(BSC only)": "#7090b8",
    "RSF-59\n(BSC+T1)":   "#568a6a",
}

def create_multimodel_boxplot(all_cv, out_path):
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10.5,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "figure.facecolor":  "white",
    })

    PANEL_BG   = "#eaeaee"
    MEAN_COLOR = "#d85820"
    MED_COLOR  = "#111111"

    labels    = list(all_cv.keys())
    positions = list(range(1, len(labels) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Model Performance Comparison (5-Fold Cross-Validation)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    panels = [
        ("train", "Train C-Index",                  "Random (0.5)",   0.5),
        ("test",  "Test C-Index",                   "Random (0.5)",   0.5),
        ("gap",   "Overfitting Gap\n(Train − Test)", "No overfitting", 0.0),
    ]

    for ax, (key, ylabel, ref_label, ref_val) in zip(axes, panels):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, linestyle="--", linewidth=0.6, color="white", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)

        for pos, label in zip(positions, labels):
            data  = all_cv[label][key]
            color = MODEL_COLORS.get(label, "#7090b8")

            bp = ax.boxplot(
                data, positions=[pos],
                patch_artist=True, widths=0.58,
                showfliers=True,
                flierprops=dict(marker="o", markersize=5, linewidth=0,
                                markerfacecolor="none",
                                markeredgecolor="#666666", markeredgewidth=0.9),
                medianprops=dict(color=MED_COLOR, linewidth=2.0),
                whiskerprops=dict(color="#555555", linewidth=1.0),
                capprops=dict(color="#555555", linewidth=1.0),
                boxprops=dict(linewidth=0.7, edgecolor="#555555"),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.82)

            mean_val = np.mean(data)
            ax.scatter([pos], [mean_val], marker="D", color=MEAN_COLOR,
                       s=50, zorder=6, linewidths=0.5, edgecolors="#993300")

        ax.axhline(ref_val, color="#999999", linestyle="--",
                   linewidth=1.1, alpha=0.85, zorder=1)
        ax.text(0.02, ref_val, f" {ref_label}",
                transform=ax.get_yaxis_transform(),
                fontsize=8.0, color="#777777", va="bottom", ha="left")

        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Model Configuration", fontsize=10.5, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        ax.set_xlim(0.3, len(labels) + 0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved multi-model boxplot → {out_path}")

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slopes",          required=True)
    p.add_argument("--survival",        required=True)
    p.add_argument("--t1_features",     required=True)
    p.add_argument("--out_dir",         default="data/ml/results/baselines")
    p.add_argument("--figs_dir",        default="figs")
    p.add_argument("--top_k",           type=int,   default=20)
    p.add_argument("--penalize_regex",  default="nboundary")
    p.add_argument("--penalty_factor",  type=float, default=0.1)
    p.add_argument("--n_estimators",    type=int,   default=1000)
    args = p.parse_args()

    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = Path(args.figs_dir); figs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading data ...")
    print("=" * 80)
    df, t1_cols = load_and_merge(args.slopes, args.survival, args.t1_features)
    _slope_candidates = [c for c in df.columns
                         if c not in ["subject", "event", "time_years"] + t1_cols]
    slope_cols = df[_slope_candidates].select_dtypes(include="number").columns.tolist()
    y_all = df[["event", "time_years"]].copy().reset_index(drop=True)
    df    = df.reset_index(drop=True)

    all_cv = {}

    for label, cls, name in [
        ("Weibull-20",     WeibullAFTFitter,     "Weibull"),
        ("LogLogistic-20", LogLogisticAFTFitter, "LogLogistic"),
        ("LogNormal-20",   LogNormalAFTFitter,   "LogNormal"),
    ]:
        print("\n" + "=" * 80)
        print(f"{label}  (BSC slopes top-{args.top_k})")
        print("=" * 80)
        all_cv[label] = aft_cv(
            df, slope_cols, y_all, args.top_k,
            args.penalize_regex, args.penalty_factor,
            fitter_cls=cls, model_name=name,
        )
        with open(out_dir / f"cv_{label.lower().replace('-','_')}.json", "w") as f:
            json.dump(all_cv[label], f, indent=2)

    print("\n" + "=" * 80)
    print(f"RSF  (BSC slopes only, top-{args.top_k})")
    print("=" * 80)
    all_cv["RSF-20\n(BSC only)"] = rsf_slopes_cv(
        df, slope_cols, y_all, args.top_k,
        args.penalize_regex, args.penalty_factor,
        n_estimators=args.n_estimators,
    )
    with open(out_dir / "cv_rsf_slopes.json", "w") as f:
        json.dump(all_cv["RSF-20\n(BSC only)"], f, indent=2)

    rsf_cv_path = Path("data/ml/results/rsf_combined/rsf_combined_cv.json")
    if rsf_cv_path.exists():
        print("\nLoading pre-computed RSF combined CV results ...")
        with open(rsf_cv_path) as f:
            all_cv["RSF-59\n(BSC+T1)"] = json.load(f)
    else:
        print("WARNING: rsf_combined_cv.json not found — run train_rsf_combined.py first")

    with open(out_dir / "all_models_cv.json", "w") as f:
        json.dump(all_cv, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  {'Model':<26}  {'Train':>8}  {'±':>6}  {'Test':>8}  {'±':>6}  {'Gap':>8}")
    print("  " + "-" * 72)
    for name, cv in all_cv.items():
        flat = name.replace("\n", " ")
        print(f"  {flat:<26}  "
              f"{np.mean(cv['train']):>8.4f}  {np.std(cv['train']):>6.4f}  "
              f"{np.mean(cv['test']):>8.4f}  {np.std(cv['test']):>6.4f}  "
              f"{np.mean(cv['gap']):>8.4f}")
    print("=" * 80)

    create_multimodel_boxplot(all_cv, figs_dir / "model_comparison_boxplot.png")

    print("\nDone. Results in:", out_dir)

if __name__ == "__main__":
    main()
