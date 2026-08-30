"""Figures for the ALZ-26-0928 V3 revision manuscript.

Five figures, PNG only. The architecture schematic is supplied separately by the
author and dropped in as Figure 3, so the numbering here skips it.

  fig1  Cross-validated C-index by feature set and model family
  fig2  Kaplan-Meier by predicted-risk tertile
  (fig3 is the architecture schematic, not produced here)
  fig4  Effect of the follow-up definition
  fig5  Fold-matched incremental value
  fig6  What BSC covaries with

Every figure reads from a file under results/spec_v3_harmonized/, so the numbers
in the plots and the numbers in the manuscript cannot drift apart.

Figures 1 and 2 follow the styling in
mri-bsc/code/ml/generate_model_comparison_boxplot.py so they match the figures
already published in this series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/spec_v3_harmonized"
CLUSTER = ROOT.parent / "from_cluster"
from nireports_fignames import target

# Risk tertiles, high to low. Line style backs up the colour, because the orange
# and green are too close for red-blind readers to separate reliably.
TIERS = [(2, "High Risk", "#C4362B", "solid"),
         (1, "Medium Risk", "#E8850E", (0, (6, 2))),
         (0, "Low Risk", "#3F8F3F", (0, (2, 1.6)))]

BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#cfcfcf"


def save(fig, name):
    out = target(name)
    if out is None:
        plt.close(fig)
        print(f"  skipped {name}, not included in the manuscript")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name} ({name})")


def clean_rc():
    """Plain look for the small figures, after the darkgrid style has been used."""
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


# ============================================================ figure 1
def fig1_model_comparison():
    """Cross-validated C-index for every feature set and model family.

    Lives in make_model_comparison.py, which reads the same table3_cindex.csv
    that Table 3 is built from. Kept here so the whole figure set still builds
    from one entry point. The earlier version of this figure plotted fold-level
    train, test and gap boxes for six hand-picked configurations; it was
    replaced so that the figure and the table can no longer disagree.
    """
    from make_model_comparison import main as build
    build()
    clean_rc()


# ============================================================ figure 2
def fig2_kaplan_meier():
    """Kaplan-Meier by predicted-risk tertile, two contrasting models.

    Risk scores are out-of-fold, so no subject is stratified by a model fitted
    on them.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    d = pd.read_csv(RES / "oof_risk.csv")
    configs = list(dict.fromkeys(d.config))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, len(configs), figsize=(14, 5.5), sharey=True)
    for ax, cfg in zip(np.atleast_1d(axes), configs):
        g = d[d.config == cfg].copy()
        g["tertile"] = pd.qcut(g.risk, 3, labels=[0, 1, 2]).astype(int)
        for t, name, colour, ls in TIERS:
            sub = g[g.tertile == t]
            km = KaplanMeierFitter()
            km.fit(sub.time_years, sub.event,
                   label=f"{name} (n={len(sub)}, events={int(sub.event.sum())})")
            km.plot_survival_function(ax=ax, ci_show=True, color=colour,
                                      linewidth=2.0, linestyle=ls, ci_alpha=0.15)

        lr = multivariate_logrank_test(g.time_years, g.tertile, g.event)
        ptxt = ("Log-rank p<0.0001" if lr.p_value < 1e-4
                else f"Log-rank p={lr.p_value:.4f}")
        ax.text(0.97, 0.97, ptxt, transform=ax.transAxes, ha="right", va="top",
                fontsize=11, bbox=dict(boxstyle="round,pad=0.4",
                                       facecolor="#F6E7C1",
                                       edgecolor="#C9A227", linewidth=0.9))
        ax.set_title(cfg, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (years)", fontsize=13, fontweight="bold")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1.12)
        ax.legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=1.0,
                  edgecolor="#9a9a9a", fancybox=False)
    np.atleast_1d(axes)[0].set_ylabel(
        "Probability of Remaining MCI\n(Not Converting to AD)",
        fontsize=13, fontweight="bold")

    plt.suptitle("Kaplan-Meier Survival Curves by Predicted Risk",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    save(fig, "fig2_kaplan_meier")
    clean_rc()


# ============================================================ figure 4
def fig4_design_effect():
    """Slope graph: what the follow-up definition alone does to performance."""
    d = pd.read_csv(RES / "design_comparison.csv")
    p = d.pivot_table(index=["feature_set", "model"], columns="design",
                      values="cv_cindex_mean").reset_index()
    label = {"F0_covariates": "Clinical covariates",
             "F2_bsc_slopes": "BSC slopes, global",
             "F3_regional_slopes": "BSC slopes, regional",
             "F5_std_mri_slopes": "Thickness + hippocampus"}
    colour = {"F0_covariates": BLUE, "F2_bsc_slopes": VERM,
              "F3_regional_slopes": VIOLET, "F5_std_mri_slopes": GREEN}

    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    for fs in label:
        sub = p[p.feature_set == fs]
        for _, r in sub.iterrows():
            ax.plot([0, 1], [r["legacy"], r["corrected"]], color=colour[fs],
                    linewidth=1.1, alpha=0.75, solid_capstyle="round", zorder=3)
            ax.plot([0, 1], [r["legacy"], r["corrected"]], "o", color=colour[fs],
                    markersize=3.2, markeredgecolor="white", markeredgewidth=0.5,
                    zorder=4)
        ax.text(1.06, sub["corrected"].mean(), label[fs], color=colour[fs],
                fontsize=6.9, va="center", ha="left")

    ax.axhline(0.5, color=RULE, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.5, 0.506, "Random (0.5)", fontsize=6.2, color=MUTED,
            va="bottom", ha="center")
    ax.set_xlim(-0.12, 1.02)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original\ndesign", "Corrected\ndesign"], fontsize=7.5)
    ax.set_ylabel("Cross-validated C-index")
    ax.set_ylim(0.40, 0.86)
    ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.60)
    save(fig, "fig4_design_effect")


# ============================================================ figure 5
def fig5_increment():
    """Forest plot of the fold-matched nested comparisons."""
    d = pd.read_csv(RES / "incr_xgb/increment_tests.csv")
    order = ["BSC slopes over covariates", "regional BSC over covariates",
             "AD-signature composite over covariates",
             "standard MRI slopes over covariates",
             "BSC slopes over covariates + standard MRI",
             "regional BSC over covariates + standard MRI",
             "AD-signature over covariates + standard MRI"]
    short = ["BSC global", "BSC regional", "AD-signature", "Standard MRI",
             "BSC global", "BSC regional", "AD-signature"]
    d = d.set_index("comparison").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ys = np.arange(len(d))[::-1].astype(float)
    ys[:4] += 0.45
    for y, (_, r), s in zip(ys, d.iterrows(), short):
        # Standard MRI is the reference row: an established measure behaving the
        # same way is what makes the BSC nulls interpretable.
        c = GREEN if "standard MRI slopes" in r["comparison"] else BLUE
        ax.plot([r.delta_ci_lo, r.delta_ci_hi], [y, y], color=c, linewidth=1.3,
                solid_capstyle="round", zorder=3)
        ax.plot(r.fold_matched_delta, y, "o", color=c, markersize=4.2,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Change in C-index")
    ax.set_ylim(-0.8, ys.max() + 0.9)
    ax.text(0.0, ys.max() + 0.75, "Added to clinical covariates", fontsize=6.6,
            color=MUTED, ha="center")
    ax.text(0.0, 2.72, "Added to covariates and standard MRI", fontsize=6.6,
            color=MUTED, ha="center")
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    save(fig, "fig5_increment")


# ============================================================ figure 6
def fig6_construct_validity():
    """Scatter panels: BSC against biological then image-quality measures."""
    from scipy import stats
    bsc = pd.read_csv(CLUSTER / "bsc_simple_features_merged.csv")
    t1 = pd.read_csv(CLUSTER / "t1_scan_features.csv")
    coh = pd.read_csv(RES / "spec_cohort.csv")[["subject", "age_at_landmark",
                                                "landmark_date"]]
    rd = pd.read_csv(CLUSTER / "manifest_realdates.csv")
    rd["image_id"] = (rd.subject + "_" + rd.visit_code + "_"
                      + pd.to_datetime(rd.acq_date).dt.strftime("%Y-%m-%d"))
    rd["dt"] = pd.to_datetime(rd.real_date.fillna(rd.acq_date), errors="coerce")
    rd = rd.merge(coh, on="subject", how="inner")
    rd["age"] = (rd.age_at_landmark
                 + (rd.dt - pd.to_datetime(rd.landmark_date)).dt.days / 365.25)

    d = (bsc[["image_id", "bsc_dir_mean"]]
         .merge(rd[["image_id", "age"]], on="image_id")
         .merge(t1[["image_id", "qc_snr", "qc_brain_std", "seg_bpf"]],
                on="image_id").dropna())

    panels = [("age", "Age at scan (years)", "Biological"),
              ("seg_bpf", "Brain parenchymal fraction", "Biological"),
              ("qc_snr", "Signal-to-noise ratio", "Image quality"),
              ("qc_brain_std", "Brain intensity SD", "Image quality")]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), sharey=True)
    for ax, (col, xlabel, kind) in zip(axes, panels):
        c = MUTED if kind == "Biological" else VERM
        r, pv = stats.pearsonr(d[col], d.bsc_dir_mean)
        ax.scatter(d[col], d.bsc_dir_mean, s=1.4, color=c, alpha=0.16,
                   linewidths=0, rasterized=True, zorder=3)
        z = np.polyfit(d[col], d.bsc_dir_mean, 1)
        xs = np.linspace(d[col].quantile(.005), d[col].quantile(.995), 50)
        ax.plot(xs, np.polyval(z, xs), color=c, linewidth=1.4, zorder=4)
        ptxt = (f"$p$ = {pv:.2f}" if pv > 0.01
                else f"$p$ < $10^{{{int(np.floor(np.log10(pv)))}}}$")
        ax.text(0.05, 0.95, f"$r$ = {r:+.3f}\n{ptxt}", transform=ax.transAxes,
                fontsize=6.8, va="top", ha="left", color=c)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_title(kind, fontsize=7, color=c, loc="left", x=0.02)
        ax.set_xlim(d[col].quantile(.002), d[col].quantile(.998))
        if d[col].max() > 5000:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("BSC directional mean")
    axes[0].set_ylim(d.bsc_dir_mean.quantile(.002), d.bsc_dir_mean.quantile(.998))
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig6_construct_validity")


if __name__ == "__main__":
    print(f"building figures into {OUT}")
    clean_rc()
    for fn in (fig1_model_comparison, fig2_kaplan_meier, fig4_design_effect,
               fig5_increment, fig6_construct_validity):
        fn()
    print("figure 3 (architecture) is supplied separately and not built here")
