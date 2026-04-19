"""
Sample overview figures: cross-sectional histogram + longitudinal S-curve.
Uses real ADNI demographics (age, sex) from IDA export.
Differentiates Converters (MCI→AD, event=1) vs Stable (event=0) with line style + marker.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

IDA_PATH = Path("/Users/ishu/Downloads/idaSearch_4_17_2026.csv")

# ── Load & merge data ──────────────────────────────────────────────────────────
ida = pd.read_csv(IDA_PATH)
demo = (
    ida.groupby("Subject ID")
    .agg(sex=("Sex", "first"), age_min=("Age", "min"), age_max=("Age", "max"))
    .reset_index()
    .rename(columns={"Subject ID": "subject"})
)

surv = pd.read_csv(DATA / "ml/survival/time_to_conversion.csv")[["subject", "event"]]

subj = demo.merge(surv, on="subject", how="left")
subj["event"] = subj["event"].fillna(0).astype(int)
subj["group"] = subj["event"].map({0: "Stable", 1: "Converter"})

# ── Style constants ────────────────────────────────────────────────────────────
F_COLOR    = "#F8766D"   # salmon  – Female
M_COLOR    = "#00BFC4"   # teal    – Male
BG_COLOR   = "#EBEBEB"
GRID_COLOR = "white"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": BG_COLOR,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "xtick.bottom": False,
    "ytick.left": False,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── LEFT: Cross-sectional age histogram ───────────────────────────────────────
ax = axes[0]

age_min, age_max = int(subj["age_min"].min()), int(subj["age_min"].max()) + 2
bins = np.arange(age_min, age_max + 1, 2)  # 2-year bins

# Stable females and males (solid bars)
f_stable = subj[(subj["sex"] == "F") & (subj["event"] == 0)]["age_min"]
m_stable = subj[(subj["sex"] == "M") & (subj["event"] == 0)]["age_min"]
# Converter females and males (hatched overlay)
f_conv   = subj[(subj["sex"] == "F") & (subj["event"] == 1)]["age_min"]
m_conv   = subj[(subj["sex"] == "M") & (subj["event"] == 1)]["age_min"]

ax.hist([f_stable, m_stable], bins=bins, stacked=True,
        color=[F_COLOR, M_COLOR], alpha=0.85,
        edgecolor="white", linewidth=0.5, zorder=3, label=["F", "M"])

ax.hist([f_conv, m_conv], bins=bins, stacked=True,
        color=[F_COLOR, M_COLOR], alpha=0.5,
        edgecolor="black", linewidth=0.7, hatch="///",
        zorder=4, label=["F (converter)", "M (converter)"])

ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Cross-sectional sample", fontsize=13, fontweight="normal", pad=10)

legend_handles = [
    mpatches.Patch(color=F_COLOR, label="F"),
    mpatches.Patch(color=M_COLOR, label="M"),
    mpatches.Patch(facecolor="white", edgecolor="black", hatch="///",
                   label="Converter", alpha=0.7),
]
ax.legend(handles=legend_handles, title="Sex / Group",
          frameon=True, facecolor="white", edgecolor="none",
          fontsize=9, title_fontsize=9)
ax.set_xlim(age_min - 1, age_max + 1)
ax.set_ylim(bottom=0)

# ── RIGHT: Longitudinal S-curve ───────────────────────────────────────────────
ax = axes[1]

# Sort by baseline age (age_min) → S-curve
subj_sorted = subj.sort_values("age_min").reset_index(drop=True)

for idx, row in subj_sorted.iterrows():
    color  = F_COLOR if row["sex"] == "F" else M_COLOR
    ls     = "--"    if row["event"] == 1  else "-"
    lw     = 0.9     if row["event"] == 1  else 0.7
    end_mk = "x"     if row["event"] == 1  else "o"
    mk_sz  = 4.5     if row["event"] == 1  else 2.5

    ax.plot([row["age_min"], row["age_max"]], [idx, idx],
            color=color, linewidth=lw, linestyle=ls, alpha=0.75)
    ax.plot(row["age_min"], idx, "o", color=color, markersize=2.0, alpha=0.6)
    ax.plot(row["age_max"], idx, end_mk, color=color, markersize=mk_sz,
            markeredgewidth=1.2 if end_mk == "x" else 0.5, alpha=0.9)

ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("Subject", fontsize=11)
ax.set_title("Longitudinal sample", fontsize=13, fontweight="normal", pad=10)

legend_elements = [
    Line2D([0], [0], color=F_COLOR, lw=1.2, linestyle="-",
           marker="o", markersize=4, label="F – Stable"),
    Line2D([0], [0], color=M_COLOR, lw=1.2, linestyle="-",
           marker="o", markersize=4, label="M – Stable"),
    Line2D([0], [0], color=F_COLOR, lw=1.2, linestyle="--",
           marker="x", markersize=5, markeredgewidth=1.2, label="F – Converter"),
    Line2D([0], [0], color=M_COLOR, lw=1.2, linestyle="--",
           marker="x", markersize=5, markeredgewidth=1.2, label="M – Converter"),
]
ax.legend(handles=legend_elements, title="Sex / Group",
          frameon=True, facecolor="white", edgecolor="none",
          fontsize=9, title_fontsize=9)

# ── Save ───────────────────────────────────────────────────────────────────────
fig.tight_layout(pad=2.0)
out = FIGS / "sample_overview.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
