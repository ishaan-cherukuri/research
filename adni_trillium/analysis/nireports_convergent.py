"""Figure 7: convergent evidence that gBSC measures acquisition, not biology.

Folds the old four-panel scatter and the old three-panel construct-validity
figure into one verdict. Top row is what gBSC fails to track, bottom row is what
it does track, and the colour carries the argument: grey for a null biological
association, orange for image quality, purple for the scanner label.

  A  gBSC against age at scan, per scan.
  B  gBSC slopes against amyloid PET burden, per subject, with the result of the
     full false-discovery-rate screen across all five pathology targets.
  C  The gBSC-to-SNR association estimated three ways. The within-subject
     estimate removes every time-invariant subject characteristic and is the
     largest of the three.
  D  Cross-validated C-index at each feature set's best model family, with
     scanner identity alone included. It sits above global gBSC slopes.

Counts: the per-scan analyses use 2,386 scans with complete quality metrics,
contributed by 412 of the 417 cohort subjects. Five subjects contribute no scan
with a complete set of quality metrics. Panel B is per subject and uses the 278
with amyloid PET.

Every number is read from results/nireports/ or results/biomarker_v1/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from nireports_construct_validity import per_scan_table
from nireports_fignames import target

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/nireports"
COH = ROOT / "results/spec_v3_harmonized"
BIO = ROOT / "results/biomarker_v1"
CLUSTER = ROOT.parent / "from_cluster"

BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE, NULLC = "#1a1a1a", "#6b6b6b", "#cfcfcf", "#8a8a8a"


def clean_rc():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.8,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def scatter(ax, x, y, colour, xlabel, ylabel, title, note):
    ax.scatter(x, y, s=1.5, color=colour, alpha=0.16, linewidths=0,
               rasterized=True, zorder=3)
    z = np.polyfit(x, y, 1)
    xs = np.linspace(np.quantile(x, .005), np.quantile(x, .995), 50)
    ax.plot(xs, np.polyval(z, xs), color=colour, linewidth=1.4, zorder=4)
    ax.set_xlim(np.quantile(x, .004), np.quantile(x, .996))
    ax.set_ylim(np.quantile(y, .004), np.quantile(y, .996))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=7.8, color=INK)
    ax.text(0.035, 0.955, note, transform=ax.transAxes, fontsize=6.4,
            va="top", ha="left", color=colour)
    ax.yaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def main():
    clean_rc()
    cv = json.loads((RES / "construct_validity.json").read_text())
    cohort = pd.read_csv(COH / "spec_cohort.csv")
    scans = per_scan_table(CLUSTER, cohort)
    bio = pd.read_csv(BIO / "cohort_with_biomarkers.csv")
    screen = pd.read_csv(BIO / "construct_validity_biomarkers.csv")
    t3 = pd.read_csv(COH / "table3_cindex_regen.csv").set_index("feature_set")
    fam = ["xgb_aft", "rsf", "cox_l2", "cox_lasso", "aft_weibull",
           "aft_lognormal", "aft_loglogistic"]

    fig = plt.figure(figsize=(7.2, 4.35))
    gs = GridSpec(2, 12, figure=fig, hspace=0.85, wspace=2.7,
                  left=0.075, right=0.985, top=0.855, bottom=0.105)

    # ---------------------------------------------------------- row headings
    fig.text(0.075, 0.960, "What gBSC does not track (biology)", fontsize=8.4,
             color=NULLC, fontweight="bold", ha="left")
    fig.text(0.075, 0.445, "What gBSC does track (acquisition)", fontsize=8.4,
             color=VERM, fontweight="bold", ha="left")

    # ---------------------------------------------------------- A: age
    xs = {d["predictor"]: d for d in cv["cross_sectional"]}
    a = xs["age"]
    ax = fig.add_subplot(gs[0, 0:6])
    scatter(ax, scans["age"].to_numpy(), scans["bsc_dir_mean"].to_numpy(),
            NULLC, "Age at scan (years)", "gBSC directional mean",
            "A  No age association",
            f"r = {a['r']:+.3f}".replace("-", "−") + f"\np = {a['p']:.2f}\n"
            f"{len(scans):,} scans, {scans.subject.nunique()} of "
            f"{len(cohort)} subjects")

    # ---------------------------------------------------------- B: pathology
    pb = bio[["bsc_dir_mean_slope", "pet_CENTILOIDS"]].dropna()
    row = screen[(screen.target == "pet_CENTILOIDS")
                 & (screen.feature == "bsc_dir_mean_slope")].iloc[0]
    n_sig = int((screen["q"] < 0.05).sum())
    n_tested = int(screen.groupby("target").size().iloc[0])
    ax = fig.add_subplot(gs[0, 6:12])
    scatter(ax, pb["pet_CENTILOIDS"].to_numpy(),
            pb["bsc_dir_mean_slope"].to_numpy(), NULLC,
            "Amyloid PET (centiloids)", "gBSC directional slope",
            "B  No pathology association",
            f"$\\rho$ = {row.rho:+.3f}".replace("-", "−")
            + f"\np = {row.p:.2f}\n{len(pb)} subjects with amyloid PET")
    ax.text(0.035, 0.045, f"{n_sig} of {n_tested} gBSC slopes survive FDR against "
            "amyloid PET,\nCSF A$\\beta$42, CSF p-tau, their ratio or tau PET",
            transform=ax.transAxes, fontsize=6.2, ha="left", va="bottom",
            color=NULLC, style="italic", zorder=6,
            bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.6))

    # ---------------------------------------------------------- C: SNR
    u = cv["within_subject"]["unadjusted"]["qc_snr"]
    r_cs = xs["qc_snr"]["r"]
    n_cs = xs["qc_snr"]["n_scans"]
    se = 1.0 / np.sqrt(n_cs - 3)
    z = np.arctanh(r_cs)
    rows = [("Across scans", r_cs, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)),
            ("Between subjects", u["between"]["coef"], u["between"]["ci_lo"],
             u["between"]["ci_hi"]),
            ("Within subject", u["within"]["coef"], u["within"]["ci_lo"],
             u["within"]["ci_hi"])]
    ax = fig.add_subplot(gs[1, 0:5])
    for i, (lab, m, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        lw = 2.0 if lab.startswith("Within") else 1.4
        ms = 5.6 if lab.startswith("Within") else 4.2
        ax.plot([lo, hi], [y, y], color=VERM, linewidth=lw,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [y], "o", color=VERM, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.text(hi + 0.028, y, f"{m:+.2f}".replace("-", "−"), fontsize=6.5,
                va="center", color=VERM)
    ax.axvline(0, color=RULE, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.8)
    ax.set_xlim(-0.05, 0.62)
    ax.set_xlabel("gBSC per SD of signal-to-noise ratio")
    ax.set_title("C  Tracks image quality, most within-brain", loc="left",
                 fontsize=7.8, color=INK)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ---------------------------------------------------------- D: C-index
    sc = cv["scanner_signal"]["cindex_scanner_only"]
    bars = [("Clinical covariates", float(t3.loc["F0_covariates", fam].max()), BLUE),
            ("Thickness + hippocampus", float(t3.loc["F5_std_mri_slopes", fam].max()), GREEN),
            ("gBSC slopes, regional", float(t3.loc["F3_regional_slopes", fam].max()), VERM),
            ("Scanner identity alone", sc, VIOLET),
            ("gBSC slopes, global", float(t3.loc["F2_bsc_slopes", fam].max()), VERM)]
    ax = fig.add_subplot(gs[1, 5:12])
    y = np.arange(len(bars))[::-1]
    for yy, (lab, v, c) in zip(y, bars):
        hl = lab.startswith("Scanner")
        ax.barh(yy, v, color=c, height=0.68, zorder=3,
                edgecolor=INK if hl else "none", linewidth=1.0 if hl else 0)
        ax.text(v + 0.006, yy, f"{v:.3f}", va="center", fontsize=6.6, color=c,
                fontweight="bold" if hl else "normal")
    ax.axvline(0.5, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
    ax.set_ylim(-0.62, 4.78)
    ax.text(0.503, 4.50, "chance", fontsize=6.2, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_xlim(0.45, 0.87)
    ax.set_xlabel("Cross-validated C-index (best model family)")
    ax.set_title("D  Scanner identity out-predicts gBSC", loc="left",
                 fontsize=7.8, color=INK)
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    out = target("nireports_convergent")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
