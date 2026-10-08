"""Figure 6: fold-matched incremental value of each imaging feature set.

The clearest single statement of the null. Every interval crosses zero, and the
increment gBSC produces is indistinguishable from the one produced by standard
volumetric measures over the same folds, which is why the standard-MRI row is
drawn in a second colour rather than alongside the gBSC rows.

No significance markers are drawn. Nothing is significant, and that is the point.

Every value is read from results/spec_v3_harmonized/incr_xgb/increment_tests.csv.
Nothing is hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

from nireports_fignames import target

RES = Path(__file__).resolve().parent / "results/spec_v3_harmonized/incr_xgb"

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#cfcfcf"

# (baseline set, augmented set, row label, colour). The standard-MRI row is the
# comparator: an established measure put through the identical test.
ROWS = [
    ("header", "Added to clinical covariates", None, None),
    ("F0_covariates", "F6_cov_bsc", "gBSC slopes, global", VERM),
    ("F0_covariates", "F7_cov_regional", "gBSC slopes, regional", VERM),
    ("F0_covariates", "F4b_cov_adsig", "AD-signature composite", VERM),
    ("F0_covariates", "F8_cov_stdmri", "Standard MRI slopes", GREEN),
    ("header", "Added to covariates + standard MRI", None, None),
    ("F8_cov_stdmri", "F9_cov_bsc_stdmri", "gBSC slopes, global", VERM),
    ("F8_cov_stdmri", "F9b_cov_regional_stdmri", "gBSC slopes, regional", VERM),
    ("F8_cov_stdmri", "F9c_cov_adsig_stdmri", "AD-signature composite", VERM),
]


def main():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 7.4, "axes.labelsize": 7.6,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.2,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False,
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })
    d = pd.read_csv(RES / "increment_tests.csv")
    key = d.set_index(["baseline", "augmented"])

    fig, ax = plt.subplots(figsize=(3.46, 2.95))
    y = len(ROWS) - 1
    yticks, ylabels = [], []
    for base, aug, label, colour in ROWS:
        if base == "header":
            # White backing so the zero line and gridlines do not run
            # through the group heading.
            ax.text(-0.0435, y, aug, fontsize=7.0, color=INK, style="italic",
                    va="center", ha="left", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
            yticks.append(y)
            ylabels.append("")
            y -= 1
            continue
        r = key.loc[(base, aug)]
        lo, hi, m = r.delta_ci_lo, r.delta_ci_hi, r.fold_matched_delta
        ax.plot([lo, hi], [y, y], color=colour, linewidth=1.5,
                solid_capstyle="round", zorder=3)
        ax.plot([m], [y], "o", color=colour, markersize=4.4,
                markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        ax.text(0.0665, y, f"{m:+.3f}".replace("-", "−"), fontsize=6.6,
                color=colour, va="center", ha="right")
        yticks.append(y)
        ylabels.append("    " + label)
        y -= 1

    ax.axvline(0, color=INK, linewidth=0.9, zorder=2)
    for g in (-0.02, 0.02, 0.04):
        ax.axvline(g, color=RULE, linewidth=0.5, zorder=1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_xlim(-0.045, 0.068)
    ax.set_xticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_xticklabels(["−0.04", "−0.02", "0", "+0.02", "+0.04"])
    for g in (-0.04,):
        ax.axvline(g, color=RULE, linewidth=0.5, zorder=1)
    ax.set_xlabel("Change in C-index")
    ax.set_axisbelow(True)

    # Legend distinguishing the biomarker under test from the established
    # comparator held to the identical standard.
    ax.plot([], [], "o-", color=VERM, markersize=4.4, linewidth=1.5,
            label="gBSC feature set")
    ax.plot([], [], "o-", color=GREEN, markersize=4.4, linewidth=1.5,
            label="Established comparator")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.19),
              ncol=2, handlelength=1.4, columnspacing=1.2, fontsize=6.6)

    fig.tight_layout()
    out = target("nireports_forest")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
