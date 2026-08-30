"""Figure 7 for the NeuroImage: Reports revision: the construct-validity upgrade.

Figure 6 already shows that BSC tracks image quality and not age across scans.
This figure adds the two things that turn that correlation into an argument:
that the association is at least as strong inside a single brain over time as it
is across people, and that a model given nothing but acquisition structure
discriminates converters about as well as BSC does.

  Panel A  The BSC-to-SNR association estimated three ways, per standard
           deviation, with 95 per cent intervals.
  Panel B  Every scan with its own subject's mean removed from both axes, which
           is the within-subject association drawn directly.
  Panel C  Cross-validated C-index for acquisition structure alone, set beside
           the BSC feature sets and the clinical covariates.

Reads only from results/nireports/ and the frozen cohort, so the figure and the
manuscript cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/nireports"
COH = ROOT / "results/spec_v3_harmonized"
CLUSTER = ROOT.parent / "from_cluster"
OUT = ROOT.parent.parent / "mri-bsc/paper/neuroimage_clinical"

BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#cfcfcf"


def clean_rc():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def panel_a(ax, cv: dict):
    """BSC on SNR, estimated cross-sectionally, between subjects and within."""
    xs = cv["cross_sectional"]
    r = next(d["r"] for d in xs if d["predictor"] == "qc_snr")
    n = next(d["n_scans"] for d in xs if d["predictor"] == "qc_snr")
    # Fisher z interval for the raw correlation, so all three carry an interval.
    se = 1.0 / np.sqrt(n - 3)
    z = np.arctanh(r)
    xr_lo, xr_hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

    # Single-predictor estimates, so all three points mean the same thing:
    # the change in BSC per standard deviation of signal-to-noise ratio.
    b = cv["within_subject"]["unadjusted"]["qc_snr"]["between"]
    w = cv["within_subject"]["unadjusted"]["qc_snr"]["within"]

    rows = [("Across scans\n(cross-sectional)", r, xr_lo, xr_hi, MUTED),
            ("Across subjects\n(between)", b["coef"], b["ci_lo"], b["ci_hi"], BLUE),
            ("Within subject\n(over time)", w["coef"], w["ci_lo"], w["ci_hi"], VERM)]

    for i, (lab, c, lo, hi, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([lo, hi], [y, y], color=col, linewidth=1.6, solid_capstyle="round",
                zorder=3)
        ax.plot([c], [y], "o", color=col, markersize=5, zorder=4)
        ax.text(hi + 0.03, y, f"{c:+.2f}", va="center", fontsize=6.8, color=col)
    ax.axvline(0, color=RULE, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
    ax.set_xlim(-0.1, 0.75)
    ax.set_xlabel("BSC per SD of signal-to-noise ratio")
    ax.set_title("A  BSC tracks image quality within a brain", loc="left",
                 fontsize=8, color=INK)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_b(ax, w: pd.DataFrame):
    """Every scan, with each subject's own mean removed from both axes."""
    ax.scatter(w.qc_snr, w.bsc_dir_mean, s=1.6, color=VERM, alpha=0.18,
               linewidths=0, rasterized=True, zorder=3)
    z = np.polyfit(w.qc_snr, w.bsc_dir_mean, 1)
    xs = np.linspace(w.qc_snr.quantile(.005), w.qc_snr.quantile(.995), 50)
    ax.plot(xs, np.polyval(z, xs), color=VERM, linewidth=1.5, zorder=4)
    ax.axhline(0, color=RULE, linewidth=0.7, zorder=1)
    ax.axvline(0, color=RULE, linewidth=0.7, zorder=1)
    ax.set_xlim(w.qc_snr.quantile(.004), w.qc_snr.quantile(.996))
    ax.set_ylim(w.bsc_dir_mean.quantile(.004), w.bsc_dir_mean.quantile(.996))
    ax.set_xlabel("Signal-to-noise, subject mean removed")
    ax.set_ylabel("BSC, subject mean removed")
    ax.set_title(f"B  {len(w):,} scans, {w.subject.nunique()} subjects",
                 loc="left", fontsize=8, color=INK)


def panel_c(ax, cv: dict, table3: pd.DataFrame):
    """Acquisition structure alone, against the feature sets it should not match."""
    sc = cv["scanner_signal"]
    best = table3.set_index("feature_set")[
        ["xgb_aft", "rsf", "cox_l2", "cox_lasso", "aft_weibull",
         "aft_lognormal", "aft_loglogistic"]].max(axis=1)

    rows = [("Scanner identity alone\n(site, vendor, field strength)",
             sc["cindex_scanner_only"], VIOLET),
            ("BSC slopes, global", best["F2_bsc_slopes"], VERM),
            ("BSC slopes, regional", best["F3_regional_slopes"], VERM),
            ("Thickness + hippocampus", best["F5_std_mri_slopes"], GREEN),
            ("Clinical covariates", best["F0_covariates"], BLUE)]

    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62,
            zorder=3)
    for i, (lab, v, c) in enumerate(rows):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=6.8, color=c)
    ax.axvline(0.5, color=INK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.text(0.5, len(rows) - 0.35, " chance", fontsize=6.3, color=MUTED,
            va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.8)
    ax.set_xlim(0.4, 0.87)
    ax.set_xlabel("Cross-validated C-index (best model family)")
    ax.set_title("C  Acquisition structure predicts conversion", loc="left",
                 fontsize=8, color=INK)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    clean_rc()
    cv = json.loads((RES / "construct_validity.json").read_text())
    table3 = pd.read_csv(COH / "table3_cindex_regen.csv")
    within = pd.read_csv(RES / "within_subject_centred.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5),
                             gridspec_kw={"width_ratios": [1.05, 0.85, 1.25]})
    panel_a(axes[0], cv)
    panel_b(axes[1], within)
    panel_c(axes[2], cv, table3)
    fig.tight_layout(w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'fig7.png'}")


if __name__ == "__main__":
    main()
