"""Figure 1 for the oxford manuscript: cross-validated C-index, every feature
set against every model family.

Reads results/spec_v3_harmonized/table3_cindex_regen.csv, which is regenerated
directly from cv_results.csv and is the same file the manuscript's Table 2 is
built from, so the figure and the table cannot drift apart. The older
table3_cindex.csv was edited by hand after the run and disagrees with it in the
two BSC-only rows; do not read from that file.

One row per feature set, one marker per model family. Hue groups the families
into the three kinds of model in the study and marker shape separates the
families inside a hue, so identity never rests on colour alone. The connector
spans the range across families, which is the quantity a reader wants when
asking how much of a result is the feature set and how much is the fitter.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results/spec_v3_harmonized"
from nireports_fignames import target

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#cfcfcf"

# (column, display name, hue, marker). Hue is the kind of model, marker
# separates families within a kind.
FAMILIES = [
    ("xgb_aft", "XGB AFT", BLUE, "o"),
    ("rsf", "RSF", BLUE, "s"),
    ("cox_l2", "Cox L2", VERM, "^"),
    ("cox_lasso", "Cox Lasso", VERM, "v"),
    ("aft_weibull", "Weibull", GREEN, "D"),
    ("aft_lognormal", "LogNorm", GREEN, "P"),
    ("aft_loglogistic", "LogLog", GREEN, "X"),
]

# Rows above the rule are single feature blocks, rows below add covariates.
N_SINGLE = 6


def clean_rc():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7.4, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False,
        "figure.dpi": 150, "savefig.dpi": 300,
    })


def main():
    clean_rc()
    d = pd.read_csv(RES / "table3_cindex_regen.csv")
    # The manuscript calls the measure gBSC; the stored table predates the
    # rename, so map the display labels rather than rewriting the results file.
    d["label"] = d["label"].str.replace("BSC", "gBSC", regex=False)
    cols = [c for c, *_ in FAMILIES]

    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    ys = list(range(len(d)))[::-1]

    for y, (_, r) in zip(ys, d.iterrows()):
        vals = r[cols].astype(float)
        # Range connector first, so the markers sit on top of it.
        ax.plot([vals.min(), vals.max()], [y, y], color=RULE, linewidth=3.0,
                solid_capstyle="round", zorder=2)
        for (col, _, hue, marker) in FAMILIES:
            ax.plot(r[col], y, marker, color=hue, markersize=4.6,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.text(vals.max() + 0.012, y, f"{vals.max():.3f}", fontsize=6.6,
                color=MUTED, va="center", ha="left", zorder=5)

    ax.axvline(0.5, color=INK, linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.5, -0.72, "Random (0.5)", fontsize=6.4, color=MUTED,
            ha="center", va="bottom")

    # Separate the single blocks from the covariate combinations.
    ax.axhline(len(d) - N_SINGLE - 0.5, color=RULE, linewidth=0.6, zorder=1)
    ax.text(0.381, len(d) - 0.35, "Single feature blocks", fontsize=6.6,
            color=MUTED, ha="left", va="bottom", style="italic")
    ax.text(0.381, len(d) - N_SINGLE - 0.42, "Added to clinical covariates",
            fontsize=6.6, color=MUTED, ha="left", va="top", style="italic")

    labels = [f"{lb} ({n})" for lb, n in zip(d.label, d.n_features)]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, len(d) + 0.45)
    ax.set_xlim(0.375, 0.85)
    ax.set_xlabel("Cross-validated C-index (mean of five stratified folds)")
    ax.xaxis.grid(True, color=RULE, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], color=hue, marker=marker, linestyle="none",
                      markersize=4.6, markeredgecolor="white",
                      markeredgewidth=0.5, label=name)
               for _, name, hue, marker in FAMILIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.235),
              ncol=7, frameon=False, handletextpad=0.25, columnspacing=1.25)

    out = target("fig1_model_comparison")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
