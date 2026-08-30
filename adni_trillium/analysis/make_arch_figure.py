"""Analysis pipeline schematic for the NeuroImage: Reports manuscript.

Reads left to right: scan timeline with the pre-conversion cut, preprocessing,
segmentation, BSC extraction, harmonization and parcellation, feature families,
survival models, evaluation.

Stages 10 to 12 branch off that spine and are numbered by where they branch.
Stage 10 leaves the BSC map, because construct validity asks what the measure
tracks rather than what it predicts. Stages 11 and 12 leave evaluation, because
both re-fit the stage 8 models on the frozen folds under a different outcome
coding or in a restricted sample. Every branch carries a drawn connector back to
the stage it leaves.

Output is vector PDF, so every box, circle and arrow stays an editable object in
a drawing program. Shapes are deliberately plain (rounded rectangles, circles,
ellipses) so the layout can be restyled without being redrawn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

from nireports_fignames import target

# Same hues as the data figures, used at low alpha so text stays legible.
BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE = "#1a1a1a", "#6b6b6b", "#c9c9c9"
GROUP = "#f2f3f5"

mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.dpi": 300})

fig, ax = plt.subplots(figsize=(20.9, 7.4))
ax.set_xlim(0, 20.9); ax.set_ylim(0.2, 7.4); ax.axis("off")


def group(x, y, w, h, title=None, dashed=True):
    """Container for a processing stage."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.18",
                                facecolor=GROUP, edgecolor=MUTED, linewidth=0.9,
                                linestyle=(0, (4, 3)) if dashed else "solid", zorder=1))
    if title:
        ax.text(x + w / 2, y + h + 0.16, title, ha="center", va="bottom",
                fontsize=9.5, color=INK, zorder=6)


def box(x, y, w, h, text, colour, fs=8.2, alpha=0.30, strike=False):
    """A labelled rounded rectangle. One processing step or feature block."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                facecolor=colour, edgecolor=colour, alpha=alpha,
                                linewidth=1.0, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                facecolor="none", edgecolor=colour, linewidth=1.0, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=5,
            path_effects=None)
    if strike:
        ax.plot([x + 0.06, x + w - 0.06], [y + h / 2, y + h / 2], color=VERM,
                linewidth=1.3, zorder=6)


def ellipse(cx, cy, w, h, text, colour, fs=8.0):
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=colour, edgecolor=colour,
                         alpha=0.30, linewidth=1.0, zorder=3))
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor="none", edgecolor=colour,
                         linewidth=1.0, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, zorder=5)


def circle(cx, cy, r, colour, alpha=0.35):
    ax.add_patch(Circle((cx, cy), r, facecolor=colour, edgecolor=colour,
                        alpha=alpha, linewidth=1.0, zorder=3))
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=colour,
                        linewidth=1.0, zorder=4))


def arrow(x1, y1, x2, y2, colour=INK, style="-|>", lw=1.0, rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=colour, linewidth=lw,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# ===================================================================== column 1
# Scan timeline. Spacing is intentionally uneven: acquisition dates are now real
# rather than inferred from nominal visit labels.
ax.text(1.15, 7.05, "1. Scan series", ha="center", fontsize=9.5, color=INK)
ax.text(1.15, 6.74, "real acquisition dates", ha="center", fontsize=7.4,
        color=MUTED, style="italic")

scans = [(6.15, "Baseline", False), (5.35, "+0.51 yr", False),
         (4.70, "+1.04 yr", False), (3.90, "+2.13 yr", False),
         (2.40, "+3.02 yr", True), (1.75, "+4.10 yr", True)]
for cy, lab, post in scans:
    circle(0.60, cy, 0.22, VERM if post else BLUE, alpha=0.20 if post else 0.35)
    ax.text(0.95, cy, lab, ha="left", va="center", fontsize=7.4,
            color=MUTED if post else INK)
    if post:
        ax.plot([0.35, 0.85], [cy, cy], color=VERM, linewidth=1.4, zorder=6)

# The cut. Everything at or after the dementia diagnosis leaves the feature set,
# and the survival clock starts at the last remaining scan.
ax.plot([0.25, 2.30], [3.35, 3.35], color=VERM, linewidth=1.6,
        linestyle=(0, (5, 2)), zorder=6)
ax.text(0.25, 3.16, "dementia diagnosis", ha="left", fontsize=7.2, color=VERM)
ax.text(0.25, 2.94, "later scans excluded", ha="left", fontsize=7.0, color=VERM)
arrow(1.90, 3.90, 1.90, 3.45, colour=GREEN, lw=1.4, style="-|>")
ax.text(2.00, 3.68, "survival clock starts here", ha="left", va="center",
        fontsize=7.0, color=GREEN)

# ===================================================================== column 2
group(2.60, 3.90, 2.05, 2.00, "2. Preprocessing")
for i, t in enumerate(["N4 bias correction", "Skull stripping",
                       "Resample 1 mm$^3$"]):
    box(2.75, 5.24 - i * 0.58, 1.75, 0.44, t, BLUE, fs=7.6)

# ===================================================================== column 3
group(4.95, 3.90, 1.90, 2.00, "3. Segmentation")
for i, t in enumerate(["GM probability", "WM probability", "CSF probability"]):
    ellipse(5.90, 5.52 - i * 0.58, 1.62, 0.44, t, VIOLET, fs=7.4)
ax.text(5.90, 4.02, "Atropos 3-class $k$-means", ha="center", fontsize=7.0,
        color=MUTED, style="italic")

# ===================================================================== column 4
group(7.15, 3.90, 2.05, 2.00, "4. BSC computation")
for i, t in enumerate(["Boundary identification", "Gradient $\\nabla I$",
                       "Project to normal", "BSC map"]):
    box(7.30, 5.38 - i * 0.45, 1.75, 0.37, t, VERM, fs=7.4)
ax.text(8.18, 3.50, r"$\mathrm{BSC}_{\mathrm{dir}}=\nabla I\cdot"
                    r"\frac{\nabla P_{GM}}{\|\nabla P_{GM}\|}$",
        ha="center", fontsize=8.6, color=INK)

# ===================================================================== column 5
# New in the revision: harmonization, and the regional parcellation branch.
group(9.50, 4.90, 2.55, 1.25, "5. Harmonization")
box(9.68, 5.56, 2.20, 0.44, "LongComBat", GREEN, fs=8.0)
ax.text(10.78, 5.31, "batch = site $\\times$ vendor", ha="center", fontsize=6.8,
        color=MUTED)
ax.text(10.78, 5.09, "$\\times$ field strength", ha="center", fontsize=6.8,
        color=MUTED)

group(9.50, 2.55, 2.55, 1.75, "6. Parcellation")
for i, t in enumerate(["DKT labels, once per subject", "Rigid-propagate to visits",
                       "Extend 5 mm into boundary"]):
    box(9.68, 3.66 - i * 0.47, 2.20, 0.39, t, VIOLET, fs=7.0)
ax.text(10.78, 2.72, "62 cortical regions", ha="center", fontsize=7.0,
        color=MUTED, style="italic")

# ===================================================================== column 6
group(12.35, 1.55, 2.55, 5.05, "7. Feature families")
ax.text(13.63, 7.02, "per subject, over the pre-conversion window",
        ha="center", fontsize=6.8, color=MUTED, style="italic")
feat = [(5.95, "Global BSC slopes", "33", VERM),
        (5.15, "Regional BSC slopes", "124", VIOLET),
        (4.35, "AD-signature composite", "2", VIOLET),
        (3.55, "Clinical covariates", "8", BLUE),
        (2.75, "Thickness + hippocampus", "69", GREEN)]
for cy, name, n, c in feat:
    box(12.50, cy - 0.28, 2.25, 0.57, f"{name}\n$n={n}$", c, fs=7.2)
ax.text(13.63, 2.16, "slope $f(t)=\\beta_0+\\beta_1 t$ on real dates",
        ha="center", fontsize=7.0, color=MUTED)
ax.text(13.63, 1.78, "13 feature sets in total", ha="center", fontsize=7.2,
        color=INK)

# Comparator families enter from outside the imaging pipeline.
ax.text(11.95, 1.15, "from ADNI clinical and\nFreeSurfer tables", ha="center",
        fontsize=6.8, color=MUTED, style="italic")
arrow(11.95, 1.62, 12.42, 3.05, colour=MUTED, dashed=True, rad=-0.20)

# ===================================================================== column 7
group(15.20, 4.05, 2.30, 2.30, "8. Survival models")
box(15.35, 5.72, 2.00, 0.46, "XGBoost AFT (primary)", GREEN, fs=7.6)
for i, t in enumerate(["Random survival forest", "Cox L2  ·  Lasso-Cox",
                       "Weibull · LogNormal ·\nLogLogistic AFT"]):
    box(15.35, 5.12 - i * 0.50, 2.00, 0.42 if i < 2 else 0.50, t, MUTED, fs=7.0,
        alpha=0.16)

group(15.20, 0.70, 2.30, 2.70, "9. Evaluation")
for i, t in enumerate(["5-fold CV + 20% hold-out",
                       "Fold-matched increment\nwith bootstrap CI",
                       "Calibration: integrated\nBrier score, time-dep. AUC",
                       "Design comparison:\ncorrected vs original clock"]):
    box(15.35, 2.78 - i * 0.62, 2.00, 0.54, t, BLUE, fs=7.0)

# ============================================================ construct validity
# Branches from the BSC map rather than from the models: these ask what BSC
# tracks, which is prior to whether it predicts.
group(2.60, 0.42, 6.60, 2.42)
ax.text(2.66, 3.00, "10. Construct validity of BSC", ha="left", va="bottom",
        fontsize=9.5, color=INK, zorder=6)
cv_boxes = [(2.78, 1.78, "Within-subject mixed model:\n$\\Delta$BSC vs $\\Delta$image quality"),
            (5.98, 1.78, "Scanner-only survival model:\nsite · vendor · field strength"),
            (2.78, 0.86, "BSC slopes vs amyloid\nand tau biomarkers"),
            (5.98, 0.86, "Regional BSC residualized\non scanner identity")]
for bx, by, t in cv_boxes:
    box(bx, by, 3.02, 0.86, t, VERM, fs=7.0)
ax.text(5.90, 0.62, "2,386 scans with complete quality metrics; also raw versus "
        "quality-adjusted outcome association", ha="center", fontsize=6.6,
        color=MUTED, style="italic")

# ============================================================ column 8
group(18.05, 4.30, 2.30, 2.05, "11. Competing risks")
for i, t in enumerate(["28 deaths before\nconversion",
                       "Cause-specific C-index",
                       "Fine-Gray subdistribution"]):
    box(18.20, 5.60 - i * 0.62, 2.00, 0.54, t, GREEN, fs=7.0)

group(18.05, 0.85, 2.30, 2.50, "12. Biomarker subgroup")
for i, t in enumerate(["Amyloid PET / CSF A$\\beta$42\nat landmark",
                       "A+ subgroup:\n$n=188$, 81 events",
                       "Amyloid as covariate,\nall 417 subjects"]):
    box(18.20, 2.68 - i * 0.82, 2.00, 0.62, t, VIOLET, fs=7.0)
ax.text(19.20, 0.44, "from ADNI biomarker, registry and\nadverse-event tables",
        ha="center", fontsize=6.6, color=MUTED, style="italic")

# ===================================================================== flow
arrow(2.32, 4.90, 2.60, 4.90)
arrow(4.67, 4.90, 4.95, 4.90)
arrow(6.87, 4.90, 7.15, 4.90)
arrow(9.22, 5.15, 9.50, 5.50, rad=0.12)          # BSC to harmonization
arrow(9.22, 4.45, 9.50, 3.85, rad=-0.12)         # BSC to parcellation
arrow(12.07, 5.40, 12.35, 5.90, rad=0.12)        # harmonized global features
arrow(12.07, 3.45, 12.35, 4.55, rad=0.10)        # regional features
arrow(14.92, 4.60, 15.20, 5.10, rad=0.10)        # features to models
arrow(16.35, 4.05, 16.35, 3.46)                  # models to evaluation
# Stage 4 branches down into construct validity.
arrow(7.35, 3.88, 6.95, 2.92, rad=-0.10, lw=1.1)
ax.text(6.55, 3.42, "per-scan BSC with image-quality metrics", ha="right",
        va="center", fontsize=6.6, color=MUTED, style="italic")

# Stage 9 branches into 11 and 12: one riser, two arrowheads.
ax.plot([17.52, 17.80], [2.10, 2.10], color=INK, linewidth=1.1, zorder=2)
ax.plot([17.80, 17.80], [2.10, 5.30], color=INK, linewidth=1.1, zorder=2)
arrow(17.80, 5.30, 18.06, 5.30, lw=1.1)
arrow(17.80, 2.10, 18.06, 2.10, lw=1.1)
ax.text(17.67, 3.85, "re-fit on the frozen folds", rotation=90, ha="center",
        va="center", fontsize=6.6, color=MUTED, style="italic")

_out = target("fig0_architecture")
if _out is None:
    # Superseded by nireports_arch.py, which draws Figure 2 for the manuscript.
    print("skipped: this schematic is no longer in the manuscript, "
          "see nireports_arch.py")
else:
    _out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out)
    print(f"wrote {_out}")
