"""Figure 2: analysis pipeline, drawn for a null-analysis paper.

The earlier schematic read like a prediction-tool diagram: the imaging branch
dominated the canvas and the flow ended on risk scores. That framing works
against the paper, which argues that BSC adds nothing once cheaper information is
in the model. This version makes two structural changes.

  * The three feature sources are siblings at one visual level. Clinical
    covariates and standard MRI are the control condition, not an afterthought,
    so they get the same size and weight as the BSC branch.
  * The flow ends on the incremental-value test rather than on a C-index, since
    the question the study asks is whether BSC adds anything, not how well a
    model predicts.

Feature counts are the three that appear in the reported comparison: regional BSC
(124), clinical covariates (8) and standard MRI (69). No global-slope count is
drawn, because the global set is not part of the final comparison. The delta on
the output chip is the regional-BSC-over-covariates-plus-standard-MRI result read
from results/spec_v3_harmonized/incr_xgb/increment_tests.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nireports_fignames import target

RES = Path(__file__).resolve().parent / "results/spec_v3_harmonized/incr_xgb"

BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#c9c9c9"
GROUP = "#f2f3f5"

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.dpi": 300,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})

fig, ax = plt.subplots(figsize=(13.5, 6.0))
ax.set_xlim(0, 13.5)
ax.set_ylim(0.15, 6.0)
ax.axis("off")


def group(x, y, w, h, title=None, tx=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.16",
                                facecolor=GROUP, edgecolor=MUTED, linewidth=0.9,
                                linestyle=(0, (4, 3)), zorder=1))
    if title:
        ax.text(x + w / 2 if tx is None else tx, y + h + 0.14, title,
                ha="center" if tx is None else "left", va="bottom",
                fontsize=9.2, color=INK, zorder=6)


def box(x, y, w, h, text, colour, fs=7.4, alpha=0.30, lw=1.0, weight="normal"):
    for fc, ec in ((colour, colour), ("none", colour)):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.09",
                                    facecolor=fc, edgecolor=ec,
                                    alpha=alpha if fc != "none" else 1.0,
                                    linewidth=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=5, fontweight=weight)


def arrow(x1, y1, x2, y2, colour=INK, lw=1.1, rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color=colour, linewidth=lw,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# ===================================================== 1. input
group(0.20, 2.30, 1.95, 1.45, "1. Input", tx=0.20)
box(0.34, 2.92, 1.67, 0.66, "Longitudinal\nT1-weighted MRI", BLUE, fs=7.6)
ax.text(1.18, 2.62, "pre-conversion window only", ha="center", fontsize=6.5,
        color=VERM, style="italic")
ax.text(1.18, 2.42, "baseline to last MCI scan", ha="center", fontsize=6.5,
        color=VERM, style="italic")

# ===================================================== 2. BSC computation
group(2.35, 0.95, 2.95, 4.25, "2. BSC computation", tx=2.35)
steps = ["N4 bias correction, skull strip,\nresample 1 mm$^3$",
         "Atropos 3-class segmentation\n(GM / WM / CSF)",
         "Boundary band\n$0.4 \\leq P_{GM} \\leq 0.6$",
         "Directional gradient projection\n"
         r"$\mathrm{BSC}_{\mathrm{dir}}=\nabla I\cdot\frac{\nabla P_{GM}}{\|\nabla P_{GM}\|}$",
         "LongComBat harmonization\nbatch = site $\\times$ vendor $\\times$ field strength"]
for i, t in enumerate(steps):
    box(2.50, 4.32 - i * 0.79, 2.65, 0.66, t, VERM, fs=6.9)
for i in range(len(steps) - 1):
    arrow(3.82, 4.32 - i * 0.79, 3.82, 4.32 - i * 0.79 - 0.13, lw=0.9)

# ===================================================== 3. feature sources
group(5.75, 0.95, 2.55, 4.25, "3. Feature sources", tx=5.75)
ax.text(7.02, 5.02, "compared on equal terms", ha="center", fontsize=6.5,
        color=MUTED, style="italic")
sources = [(3.95, "Regional BSC slopes\n$n=124$", VERM),
           (2.55, "Clinical covariates\n$n=8$", BLUE),
           (1.15, "Standard MRI\n$n=69$", GREEN)]
for y, t, c in sources:
    box(5.90, y, 2.25, 0.78, t, c, fs=7.4)
ax.text(7.02, 3.66, "age, sex, education, APOE4, MMSE,", ha="center",
        fontsize=6.1, color=MUTED)
ax.text(7.02, 3.48, "ADAS-Cog13, CDR-SB, field strength", ha="center",
        fontsize=6.1, color=MUTED)
ax.text(7.02, 2.26, "cortical thickness (68) + hippocampal", ha="center",
        fontsize=6.1, color=MUTED)
ax.text(7.02, 2.08, "volume, longitudinal slopes", ha="center",
        fontsize=6.1, color=MUTED)
ax.text(7.02, 0.72, "covariates and standard MRI enter from ADNI tables,",
        ha="center", fontsize=6.3, color=MUTED, style="italic")
ax.text(7.02, 0.54, "never through the imaging pipeline", ha="center",
        fontsize=6.3, color=MUTED, style="italic")

# ===================================================== 4. model
group(8.75, 2.05, 2.20, 2.05, "4. Model", tx=8.75)
box(8.89, 3.28, 1.92, 0.62, "XGBoost AFT\n(flagship)", GREEN, fs=7.4)
ax.text(9.85, 3.06, "plus 6 comparator families:", ha="center", fontsize=6.2,
        color=MUTED)
ax.text(9.85, 2.88, "RSF, Cox-L2, Cox-Lasso, Weibull,", ha="center",
        fontsize=6.2, color=MUTED)
ax.text(9.85, 2.70, "LogNormal, LogLogistic", ha="center", fontsize=6.2,
        color=MUTED)
ax.text(9.85, 2.48, "conclusion invariant to model family", ha="center",
        fontsize=6.2, color=INK, style="italic")
ax.text(9.85, 2.22, "stratified 5-fold CV,", ha="center", fontsize=6.2,
        color=MUTED)
ax.text(9.85, 2.08, "transforms fit inside folds", ha="center", fontsize=6.2,
        color=MUTED)

# ===================================================== 5. incremental-value test
d = pd.read_csv(RES / "increment_tests.csv")
row = d[(d.baseline == "F8_cov_stdmri")
        & (d.augmented == "F9b_cov_regional_stdmri")].iloc[0]
def sgn(v):
    return f"{v:+.3f}".replace("-", "\u2212")

delta = (f"$\\Delta$C-index = {sgn(row.fold_matched_delta)}\n"
         f"95% CI ({sgn(row.delta_ci_lo)}, {sgn(row.delta_ci_hi)})")

group(11.35, 2.05, 2.05, 2.05, "5. Incremental-value test", tx=10.95)
ax.text(12.37, 3.86, "Does BSC add C-index over", ha="center", fontsize=6.6,
        color=INK)
ax.text(12.37, 3.68, "covariates + standard MRI?", ha="center", fontsize=6.6,
        color=INK)
box(11.49, 2.62, 1.77, 0.80, delta, MUTED, fs=7.0, alpha=0.18)
ax.text(12.37, 2.36, "null", ha="center", fontsize=8.6, color=INK,
        fontweight="bold")
ax.text(12.37, 2.16, "interval includes zero", ha="center", fontsize=6.3,
        color=MUTED, style="italic")

# ===================================================== flow
arrow(2.17, 3.02, 2.35, 3.02)
# The BSC chip is the output of the branch, so the connector leaves the last
# step and runs up the channel between the two groups rather than over the boxes.
ax.plot([5.17, 5.52], [1.49, 1.49], color=INK, linewidth=1.1, zorder=2)
ax.plot([5.52, 5.52], [1.49, 4.34], color=INK, linewidth=1.1, zorder=2)
arrow(5.52, 4.34, 5.87, 4.34)
for y in (4.34, 2.94, 1.54):
    arrow(8.32, y, 8.73, 3.10, rad=-0.10 if y > 3.1 else 0.10)
arrow(10.97, 3.06, 11.33, 3.06)

_out = target("nireports_arch")
_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_out)
plt.close(fig)
print(f"wrote {_out}")
