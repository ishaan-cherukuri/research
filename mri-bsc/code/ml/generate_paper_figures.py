"""
Generate the two paper figures from saved results, writing both to a single
directory (default: research/figs/) so they can be reviewed side by side.

  1. model_comparison_boxplot_xgb.png
     6-configuration 5-fold CV comparison (3 panels: train / test / gap).
     Reads the saved CV JSONs; refits nothing.

  2. kaplan_meier_curves_xgb.png
     Kaplan-Meier by predicted-risk tertile for the XGBoost AFT model.
     Refits XGBoost on the same 70/30 split (seed 42) because the training-set
     predictions needed for the left panel are not persisted by the trainer.

Note on risk direction: the survival:aft objective predicts log survival TIME,
so a HIGH prediction means LOW risk. Risk ordering here is taken from -pred,
matching the concordance computation in train_xgb_survival_combined.py.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import train_test_split

from train_xgb_survival_combined import (
    apply_winsor_limits,
    fit_winsor_limits,
    load_and_merge,
    minmax_scale_train_test,
    select_slope_features_train_only,
    signed_log1p_df,
    train_xgb_model,
)

MODEL_COLORS = {
    "Weibull-20":         "#a87090",
    "LogLogistic-20":     "#c07060",
    "LogNormal-20":       "#c4a030",
    "RSF-20\n(BSC only)": "#7090b8",
    "RSF-59\n(BSC+T1)":   "#568a6a",
    "XGB-59\n(BSC+T1)":   "#2f6f4f",
}

KM_COLORS = {"high": "#d62728", "mid": "#ff9e2c", "low": "#2ca02c"}


def build_boxplot(all_cv, out_path):
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10.5,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "figure.facecolor": "white",
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
        ("train", "Train C-Index",                   "Random (0.5)",   0.5),
        ("test",  "Test C-Index",                    "Random (0.5)",   0.5),
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
                patch_artist=True, widths=0.58, showfliers=True,
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

            ax.scatter([pos], [np.mean(data)], marker="D", color=MEAN_COLOR,
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
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved boxplot -> {out_path}")

    print("\n  5-fold CV summary (mean +/- sd):")
    for label in labels:
        te = all_cv[label]["test"]
        tr = all_cv[label]["train"]
        flat = label.replace("\n", " ")
        print(f"    {flat:24s} train={np.mean(tr):.4f}  "
              f"test={np.mean(te):.4f}+/-{np.std(te):.4f}  "
              f"gap={np.mean(all_cv[label]['gap']):.4f}")


def build_km(y_train, y_test, pred_train, pred_test, out_path):
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        10.5,
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.32)

    stats = {}

    for ax, y_data, pred, title in [
        (axes[0], y_train, pred_train, "Training Set"),
        (axes[1], y_test,  pred_test,  "Test Set"),
    ]:
        # AFT predicts log survival time: negate so that larger == higher risk.
        risk = -np.asarray(pred, dtype=float)

        p33, p67 = np.percentile(risk, [33.33, 66.67])
        high = risk >= p67
        low  = risk <= p33
        mid  = ~(high | low)

        time  = y_data["time_years"].values
        event = y_data["event"].values

        medians = {}
        for mask, key, label_stub in [
            (high, "high", "High Risk"),
            (mid,  "mid",  "Medium Risk"),
            (low,  "low",  "Low Risk"),
        ]:
            n, e = int(mask.sum()), int(event[mask].sum())
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask],
                    label=f"{label_stub} (n={n}, events={e})")
            kmf.plot_survival_function(ax=ax, color=KM_COLORS[key],
                                       linewidth=2.5, ci_show=True)
            medians[key] = (n, e, kmf.median_survival_time_)

        lr = logrank_test(time[high], time[low], event[high], event[low])
        stats[title] = {"medians": medians,
                        "logrank_p": float(lr.p_value),
                        "chi2": float(lr.test_statistic)}

        p_txt = ("Log-rank p<0.0001" if lr.p_value < 1e-4
                 else f"Log-rank p={lr.p_value:.4f}")

        ax.set_xlabel("Time (years)", fontsize=12, fontweight="bold")
        ax.set_ylabel("P(Remain MCI)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title}\nKaplan-Meier by Predicted Risk",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.text(0.98, 0.02, p_txt, transform=ax.transAxes, fontsize=10,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved KM curves -> {out_path}")

    print("\n  KM summary (risk = -prediction, so High Risk = shortest predicted time):")
    for title, s in stats.items():
        print(f"    {title}:")
        for key in ("high", "mid", "low"):
            n, e, med = s["medians"][key]
            med_txt = "not reached" if not np.isfinite(med) else f"{med:.2f} yr"
            print(f"      {key:5s} n={n:3d}  events={e:3d}  median={med_txt}")
        print(f"      log-rank (high vs low): chi2={s['chi2']:.2f}  p={s['logrank_p']:.3e}")

    return stats


def refit_xgb_for_km(args):
    df, t1_cols_used = load_and_merge(args.slopes, args.survival, args.t1_features)

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().mean() > 0.8]
    y_all = df[["time_years", "event"]].copy()

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=y_all["event"]
    )
    y_train, y_test = y_all.loc[idx_train], y_all.loc[idx_test]

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())
    selected = select_slope_features_train_only(
        X_slopes_all.loc[idx_train],
        top_k=args.top_k,
        penalize_regex=args.penalize_regex,
        penalty_factor=args.penalty_factor,
        winsor_q_low=args.winsor_low,
        winsor_q_high=args.winsor_high,
        quiet=True,
    )

    X_t1_all = df[t1_cols_used].copy()
    X_t1_all = X_t1_all.fillna(X_t1_all.loc[idx_train].median())
    X_all = pd.concat([X_slopes_all[selected], X_t1_all], axis=1)

    X_train_log = signed_log1p_df(X_all.loc[idx_train])
    X_test_log  = signed_log1p_df(X_all.loc[idx_test])
    limits = fit_winsor_limits(X_train_log, args.winsor_low, args.winsor_high)
    X_train_scaled, X_test_scaled, _ = minmax_scale_train_test(
        apply_winsor_limits(X_train_log, limits),
        apply_winsor_limits(X_test_log,  limits),
    )

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    _, metrics, pred_train, pred_test = train_xgb_model(
        X_train_scaled, X_test_scaled, y_train, y_test, params, args.n_estimators
    )
    return y_train, y_test, pred_train, pred_test, metrics


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slopes",       default="data/index/bsc_longitudinal_slopes.csv")
    p.add_argument("--survival",     default="data/ml/survival/time_to_conversion.csv")
    p.add_argument("--t1_features",  default="../raw_t1_analysis/features_all_456.csv")
    p.add_argument("--baselines_cv", default="data/ml/results/baselines/all_models_cv.json")
    p.add_argument("--xgb_cv",       default="data/ml/results/xgb_combined/xgb_cv_results.json")
    p.add_argument("--figs_dir",     default="../figs",
                   help="Where both figures are written (default: research/figs/)")
    p.add_argument("--top_k",          type=int,   default=20)
    p.add_argument("--penalize_regex",             default="nboundary")
    p.add_argument("--penalty_factor", type=float, default=0.10)
    p.add_argument("--winsor_low",     type=float, default=0.01)
    p.add_argument("--winsor_high",    type=float, default=0.99)
    p.add_argument("--n_estimators",   type=int,   default=500)
    p.add_argument("--max_depth",      type=int,   default=4)
    p.add_argument("--learning_rate",  type=float, default=0.05)
    args = p.parse_args()

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to: {figs_dir.resolve()}\n")

    print("=" * 80)
    print("FIGURE 1/2  Model comparison boxplot")
    print("=" * 80)
    all_cv = json.load(open(args.baselines_cv))
    xgb_cv = json.load(open(args.xgb_cv))
    all_cv["XGB-59\n(BSC+T1)"] = {k: xgb_cv[k] for k in ("train", "test", "gap")}
    build_boxplot(all_cv, figs_dir / "model_comparison_boxplot_xgb.png")

    print("\n" + "=" * 80)
    print("FIGURE 2/2  Kaplan-Meier by XGBoost predicted risk")
    print("=" * 80)
    y_train, y_test, pred_train, pred_test, metrics = refit_xgb_for_km(args)
    stats = build_km(y_train, y_test, pred_train, pred_test,
                     figs_dir / "kaplan_meier_curves_xgb.png")

    with open(figs_dir / "figure_stats_xgb.json", "w") as f:
        json.dump({"xgb_split_metrics": metrics,
                   "km": {k: {"logrank_p": v["logrank_p"], "chi2": v["chi2"],
                              "medians": {kk: [vv[0], vv[1],
                                               None if not np.isfinite(vv[2]) else vv[2]]
                                          for kk, vv in v["medians"].items()}}
                          for k, v in stats.items()}}, f, indent=2)
    print(f"\n  Wrote caption numbers -> {figs_dir / 'figure_stats_xgb.json'}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
