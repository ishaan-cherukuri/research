"""Figures for the ALZ-26-0928 revision.

fig_landmark_models.png   cross-validated test C-index by feature block and
                          model under the landmark design
fig_design_effect.png     the same 557 subjects scored under the submitted
                          follow-up definition and under the landmark one
fig_landmark_km.png       Kaplan-Meier curves by predicted-risk tertile, held
                          out across folds, for covariates and for covariates
                          plus imaging
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold

from run_revision_models import (fit_predict, fold_columns, get_blocks,
                                 prep_fold)

BLUE, ORANGE, PURPLE = "#2f6f9f", "#c8642a", "#7a5aa8"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"
TERTILE = ["#bfd4e6", "#6c9cc4", "#1f4f77"]

BLOCK_LABEL = {
    "cov": "Clinical covariates",
    "bsc": "BSC slopes",
    "t1x": "T1 cross-sectional",
    "t1long": "T1 longitudinal",
    "t1x+t1long": "T1 morphometry",
    "bsc+t1x+t1long": "BSC + T1",
    "cov+bsc": "Covariates + BSC",
    "cov+t1x+t1long": "Covariates + T1",
    "cov+bsc+t1x+t1long": "Covariates + BSC + T1",
}
MODEL_LABEL = {"coxnet": "Penalized Cox", "weibull": "Weibull AFT",
               "lognormal": "Log-normal AFT", "loglogistic": "Log-logistic AFT",
               "rsf": "RSF", "xgb": "XGBoost AFT"}


def style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def fig_models(cv_csv: Path, out: Path) -> None:
    df = pd.read_csv(cv_csv)
    order = ["cov", "bsc", "t1x", "t1long", "t1x+t1long", "bsc+t1x+t1long",
             "cov+bsc", "cov+t1x+t1long", "cov+bsc+t1x+t1long"]
    df = df[df["features"].isin(order)]
    best = (df.sort_values("test_mean", ascending=False)
              .groupby("features", as_index=False).first())
    best["rank"] = best["features"].map({f: i for i, f in enumerate(order)})
    best = best.sort_values("rank", ascending=False)

    colors = [BLUE if "cov" in f else ORANGE for f in best["features"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)
    y = np.arange(len(best))
    ax.barh(y, best["test_mean"], xerr=best["test_sd"], height=0.62,
            color=colors, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 3},
            zorder=3)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.503, len(best) - 0.35, "chance", color=MUTED, fontsize=7.5, va="center")

    for yi, (_, r) in zip(y, best.iterrows()):
        ax.text(r["test_mean"] + r["test_sd"] + 0.008, yi,
                f"{r['test_mean']:.3f}  ({MODEL_LABEL[r['model']]})",
                va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels([BLOCK_LABEL[f] for f in best["features"]], fontsize=8.5)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("5-fold cross-validated test C-index (best model per feature set)",
                  fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def fig_design(design_json: Path, out: Path) -> None:
    rows = json.load(open(design_json))
    df = pd.DataFrame(rows)
    df["key"] = df["features"] + " | " + df["model"]
    keys = [k for k in df["key"].unique()]
    piv = df.pivot(index="key", columns="design", values="test_mean").loc[keys]

    labels = [f"{BLOCK_LABEL.get(k.split(' | ')[0], k.split(' | ')[0])}\n"
              f"({MODEL_LABEL[k.split(' | ')[1]]})" for k in piv.index]
    y = np.arange(len(piv))[::-1]
    h = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=300)
    ax.barh(y + h / 2 + 0.02, piv["legacy"], height=h, color=ORANGE,
            label="Follow-up from baseline scan (as submitted)", zorder=3)
    ax.barh(y - h / 2 - 0.02, piv["landmark24"], height=h, color=BLUE,
            label="Follow-up from last MRI used (landmark)", zorder=3)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)

    for yi, k in zip(y, piv.index):
        ax.text(piv.loc[k, "legacy"] + 0.006, yi + h / 2 + 0.02,
                f"{piv.loc[k,'legacy']:.3f}", va="center", fontsize=7.5, color=INK)
        ax.text(piv.loc[k, "landmark24"] + 0.006, yi - h / 2 - 0.02,
                f"{piv.loc[k,'landmark24']:.3f}", va="center", fontsize=7.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.40, 0.85)
    ax.set_xlabel("5-fold cross-validated test C-index, same 557 subjects", fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), ncol=2, handlelength=1.4)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def out_of_fold_risk(df, blocks, feature_set, model, top_k, folds, seed):
    y_e, y_t = df["event"].to_numpy(), df["time_years"].to_numpy()
    risk = np.full(len(df), np.nan)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(df, y_e):
        cols = fold_columns(df, blocks, feature_set, tr, top_k)
        Xtr, Xte = prep_fold(df.iloc[tr][cols], df.iloc[te][cols])
        _, rte = fit_predict(model, Xtr, y_e[tr], y_t[tr], Xte, seed)
        risk[te] = rte
    return risk


def fig_km(cohort_csv: Path, out: Path, top_k: int, folds: int, seed: int) -> None:
    df = pd.read_csv(cohort_csv)
    blocks = get_blocks(df)
    panels = [(["cov"], "coxnet", "Clinical covariates only"),
              (["cov", "bsc", "t1x", "t1long"], "coxnet", "Covariates + BSC + T1")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), dpi=300, sharey=True)
    for ax, (fs, model, title) in zip(axes, panels):
        risk = out_of_fold_risk(df, blocks, fs, model, top_k, folds, seed)
        tert = pd.qcut(risk, 3, labels=["Low", "Medium", "High"])
        for name, color in zip(["Low", "Medium", "High"], TERTILE):
            m = tert == name
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[m, "time_years"], df.loc[m, "event"], label=name)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, lw=2,
                                       ci_alpha=0.12)
        lo, hi = tert == "Low", tert == "High"
        lr = logrank_test(df.loc[hi, "time_years"], df.loc[lo, "time_years"],
                          df.loc[hi, "event"], df.loc[lo, "event"])
        p = lr.p_value
        ptxt = "p < 0.0001" if p < 1e-4 else f"p = {p:.3f}"
        ax.text(0.03, 0.06, f"High vs low: {ptxt}", transform=ax.transAxes,
                fontsize=8, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("Years from landmark", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, 8)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, title="Predicted risk",
                  title_fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.15))
        style(ax)
    axes[0].set_ylabel("Probability of remaining MCI", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="analysis/results/landmark24")
    p.add_argument("--design_json", default="analysis/results/design_comparison.json")
    p.add_argument("--figs_dir", default="/Users/ishu/research/mri-bsc/figs/revision")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    figs = Path(args.figs_dir)
    figs.mkdir(parents=True, exist_ok=True)
    res = Path(args.results_dir)
    fig_models(res / "cv_model_comparison.csv", figs / "fig_landmark_models.png")
    fig_design(Path(args.design_json), figs / "fig_design_effect.png")
    fig_km(res / "landmark_cohort.csv", figs / "fig_landmark_km.png",
           args.top_k, args.folds, args.seed)


if __name__ == "__main__":
    main()
