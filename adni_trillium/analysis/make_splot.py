"""Cohort overview figure: cross-sectional age histogram plus longitudinal S-curve.

Follows mri-bsc/code/figs/plot_sample_overview.py so the figure matches the one
in the submitted version, rebuilt for the corrected 417-subject cohort.

Left panel is the age distribution at the first scan, split by sex, with
converters overlaid as hatched bars. Right panel gives each subject one
horizontal line spanning their observation window, sorted by age at first scan,
which produces the S shape and makes the enrollment structure visible.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
COH = ROOT / "results/spec_v3_harmonized/spec_cohort.csv"
from nireports_fignames import target

F_COLOR, M_COLOR = "#F8766D", "#00BFC4"
BG_COLOR, GRID_COLOR = "#EBEBEB", "white"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG_COLOR, "figure.facecolor": "white",
    "axes.grid": True, "grid.color": GRID_COLOR, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.bottom": False, "ytick.left": False,
})

d = pd.read_csv(COH)
# The landmark is the last scan in the observation window, so the first scan sits
# one window-span earlier.
d["age_start"] = d.age_at_landmark - d.window_span_years
d["age_end"] = d.age_at_landmark
d["sex"] = np.where(d.female == 1, "F", "M")
d = d.dropna(subset=["age_start", "age_end", "sex"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- left: age at first scan, converters hatched over the stacked bars
ax = axes[0]
lo, hi = int(d.age_start.min()), int(d.age_start.max()) + 2
bins = np.arange(lo, hi + 1, 2)
f_stable = d[(d.sex == "F") & (d.event == 0)].age_start
m_stable = d[(d.sex == "M") & (d.event == 0)].age_start
f_conv = d[(d.sex == "F") & (d.event == 1)].age_start
m_conv = d[(d.sex == "M") & (d.event == 1)].age_start

ax.hist([f_stable, m_stable], bins=bins, stacked=True,
        color=[F_COLOR, M_COLOR], alpha=0.85, edgecolor="white",
        linewidth=0.5, zorder=3)
ax.hist([f_conv, m_conv], bins=bins, stacked=True,
        color=[F_COLOR, M_COLOR], alpha=0.5, edgecolor="black",
        linewidth=0.7, hatch="///", zorder=4)

ax.set_xlabel("Age at first scan (years)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Cross-sectional sample", fontsize=13, pad=10)
ax.legend(handles=[mpatches.Patch(color=F_COLOR, label="F"),
                   mpatches.Patch(color=M_COLOR, label="M"),
                   mpatches.Patch(facecolor="white", edgecolor="black",
                                  hatch="///", label="Converter", alpha=0.7)],
          title="Sex / Group", frameon=True, facecolor="white",
          edgecolor="none", fontsize=9, title_fontsize=9)
ax.set_xlim(lo - 1, hi + 1)
ax.set_ylim(bottom=0)

# ---- right: one line per subject, sorted by age at first scan
ax = axes[1]
srt = d.sort_values("age_start").reset_index(drop=True)
for idx, row in srt.iterrows():
    colour = F_COLOR if row.sex == "F" else M_COLOR
    conv = row.event == 1
    ax.plot([row.age_start, row.age_end], [idx, idx], color=colour,
            linewidth=0.9 if conv else 0.7, linestyle="--" if conv else "-",
            alpha=0.75)
    ax.plot(row.age_start, idx, "o", color=colour, markersize=2.0, alpha=0.6)
    ax.plot(row.age_end, idx, "x" if conv else "o", color=colour,
            markersize=4.5 if conv else 2.5,
            markeredgewidth=1.2 if conv else 0.5, alpha=0.9)

ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("Subject", fontsize=11)
ax.set_title("Longitudinal sample", fontsize=13, pad=10)
ax.legend(handles=[
    Line2D([0], [0], color=F_COLOR, lw=1.2, marker="o", markersize=4,
           label="F, stable"),
    Line2D([0], [0], color=M_COLOR, lw=1.2, marker="o", markersize=4,
           label="M, stable"),
    Line2D([0], [0], color=F_COLOR, lw=1.2, linestyle="--", marker="x",
           markersize=5, markeredgewidth=1.2, label="F, converter"),
    Line2D([0], [0], color=M_COLOR, lw=1.2, linestyle="--", marker="x",
           markersize=5, markeredgewidth=1.2, label="M, converter")],
    title="Sex / Group", frameon=True, facecolor="white", edgecolor="none",
    fontsize=9, title_fontsize=9)

fig.tight_layout(pad=2.0)
_out = target("fig_splot_cohort")
_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_out, dpi=300, bbox_inches="tight")
print(f"wrote {_out}  "
      f"({len(d)} subjects, {int(d.event.sum())} converters)")
