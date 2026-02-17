#!/usr/bin/env python3
"""
Generate comprehensive BSC pipeline diagram for paper.

Creates a publication-quality flowchart showing the complete pipeline from
raw MRI to survival prediction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import ConnectionPatch
import numpy as np


def create_rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    color,
    text_color="white",
    fontsize=11,
    fontweight="bold",
):
    """Create a rounded rectangle box with text."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.01",
        facecolor=color,
        edgecolor="black",
        linewidth=2,
        transform=ax.transData,
    )
    ax.add_patch(box)

    # Add text in center
    center_x = xy[0] + width / 2
    center_y = xy[1] + height / 2
    ax.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        wrap=True,
    )

    return box


def create_arrow(ax, start, end, color="black", linewidth=3, style="->"):
    """Create an arrow between two points."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        color=color,
        linewidth=linewidth,
        mutation_scale=25,
        zorder=1,
    )
    ax.add_patch(arrow)
    return arrow


def create_brain_placeholder(ax, xy, size, label="", color="gray"):
    """Create a circle placeholder for brain image."""
    circle = Circle(xy, size, facecolor=color, edgecolor="black", linewidth=2)
    ax.add_patch(circle)
    if label:
        ax.text(
            xy[0],
            xy[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )
    return circle


def generate_pipeline_diagram():
    """Create the complete BSC pipeline diagram."""

    # Create large figure
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Define color scheme
    colors = {
        "data": "#3A86FF",  # Blue - raw data
        "preproc": "#06A77D",  # Green - preprocessing
        "segment": "#FB5607",  # Orange - segmentation
        "bsc": "#E63946",  # Red - BSC computation
        "features": "#8338EC",  # Purple - feature extraction
        "slopes": "#FF6B35",  # Orange-red - slopes (KEY CONTRIBUTION)
        "ml": "#FFB627",  # Yellow - machine learning
        "output": "#2A9D8F",  # Teal - outputs
    }

    # ============================================================================
    # SECTION 1: DATA INPUT (Longitudinal Emphasis)
    # ============================================================================

    # Title
    ax.text(
        10,
        13.5,
        "BSC Slope-Based MCI-to-AD Conversion Prediction Pipeline",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )

    # Subtitle
    ax.text(
        10,
        13,
        "450 ADNI Subjects • ≥4 Longitudinal Scans • 1,824 Total MRI Scans",
        ha="center",
        fontsize=11,
        style="italic",
        color="gray",
    )

    # Timeline header
    timeline_y = 12
    create_rounded_box(
        ax,
        (1, timeline_y),
        18,
        0.8,
        "Longitudinal Data Acquisition (4+ Timepoints per Subject)",
        colors["data"],
        fontsize=12,
    )

    # Four timepoint scans
    timepoints = [
        "Baseline\n(bl)",
        "Month 12\n(m12)",
        "Month 24\n(m24)",
        "Month 36\n(m36)",
    ]
    scan_y = 10.5
    scan_spacing = 4.5

    for i, tp in enumerate(timepoints):
        x = 2 + i * scan_spacing
        # Brain placeholder
        create_brain_placeholder(ax, (x, scan_y), 0.4, "", colors["data"])
        # Label
        ax.text(x, scan_y - 0.8, tp, ha="center", fontsize=9, fontweight="bold")
        # Time annotation
        ax.text(x, scan_y + 0.8, f"t={i} years", ha="center", fontsize=8, color="gray")

    # Draw timeline arrow
    create_arrow(
        ax, (1.5, scan_y), (18.5, scan_y), color="gray", linewidth=2, style="->"
    )

    # ============================================================================
    # SECTION 2: PREPROCESSING
    # ============================================================================

    preproc_y = 8.5

    # Main preprocessing box
    create_rounded_box(
        ax,
        (2, preproc_y),
        16,
        1.2,
        "Image Preprocessing Pipeline",
        colors["preproc"],
        fontsize=12,
    )

    # Three preprocessing steps
    preproc_steps = [
        ("N4 Bias\nCorrection", 3),
        ("Skull\nStripping", 8),
        ("Resample\n1mm³ Isotropic", 13),
    ]

    for step, x in preproc_steps:
        create_rounded_box(
            ax, (x, preproc_y - 1.2), 3, 0.8, step, colors["preproc"], fontsize=9
        )

    # Arrow from scans to preprocessing
    create_arrow(ax, (10, scan_y - 0.8), (10, preproc_y + 1.2), colors["preproc"])

    # ============================================================================
    # SECTION 3: SEGMENTATION
    # ============================================================================

    seg_x = 7.5
    seg_y = 7.5

    create_rounded_box(
        ax,
        (seg_x, seg_y - 0.5),
        2.2,
        2.2,
        "Atropos\nSegment",
        colors["segment"],
        fontsize=9,
    )

    # Output maps (stacked vertically)
    seg_outputs = ["GM", "WM", "CSF"]
    for i, label in enumerate(seg_outputs):
        ax.text(
            seg_x + 1.1,
            seg_y + 0.6 - i * 0.4,
            label,
            ha="center",
            fontsize=7,
            color="white",
        )

    create_arrow(
        ax, (preproc_x + 2.1, preproc_y), (seg_x - 0.1, seg_y), colors["segment"]
    )

    # ============================================================================
    # SECTION 4: BSC COMPUTATION
    # ============================================================================

    bsc_x = 10.5
    bsc_y = 7.5

    create_rounded_box(
        ax,
        (bsc_x, bsc_y - 0.5),
        2.5,
        2.2,
        "BSC\nCompute",
        colors["bsc"],
        fontsize=10,
    )

    # BSC steps (stacked)
    bsc_steps = ["Boundary", "Gradient", "Project", "BSC Map"]
    for i, step in enumerate(bsc_steps):
        ax.text(
            bsc_x + 1.25,
            bsc_y + 0.6 - i * 0.35,
            step,
            ha="center",
            fontsize=6,
            color="white",
        )

    create_arrow(ax, (seg_x + 2.3, seg_y), (bsc_x - 0.1, bsc_y), colors["bsc"])

    # BSC equation (above box)
    ax.text(
        bsc_x + 1.25,
        bsc_y + 1.5,
        r"BSC = ∇I · n̂",
        ha="center",
        fontsize=7,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ============================================================================
    # SECTION 5: FEATURE EXTRACTION (Per Timepoint)
    # ============================================================================

    feat_x = 14
    feat_y = 7.5

    create_rounded_box(
        ax, (feat_x, feat_y - 0.5), 2.5, 2.2, "", colors["features"], fontsize=10
    )

    ax.text(
        feat_x + 1.25,
        feat_y + 1.1,
        "Features",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax.text(
        feat_x + 1.25,
        feat_y + 0.8,
        "182 per scan",
        ha="center",
        fontsize=8,
        color="white",
    )

    # Feature categories
    feat_categories = [
        "Count",
        "Mean/Std",
        "Percentiles",
        "Spatial",
    ]

    for i, cat in enumerate(feat_categories):
        ax.text(
            feat_x + 1.25,
            feat_y + 0.4 - i * 0.3,
            f"• {cat}",
            ha="center",
            fontsize=6,
            color="white",
        )

    create_arrow(ax, (bsc_x + 2.6, bsc_y), (feat_x - 0.1, feat_y), colors["features"])

    # ============================================================================
    # SECTION 6: SLOPE COMPUTATION (KEY CONTRIBUTION!)
    # ============================================================================

    slope_x = 17.5
    slope_y = 7.5

    # Highlighted box for slopes
    slope_box = create_rounded_box(
        ax, (slope_x, slope_y - 0.5), 3, 2.2, "", colors["slopes"], fontsize=10
    )

    ax.text(
        slope_x + 1.5,
        slope_y + 1.1,
        "Slope Extraction",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax.text(
        slope_x + 1.5,
        slope_y + 0.8,
        "⭐ KEY ⭐",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="yellow",
    )

    # Show linear regression formula
    ax.text(
        slope_x + 1.5,
        slope_y + 0.4,
        "f(t) = β₀ + β₁·t",
        ha="center",
        fontsize=7,
        style="italic",
        color="white",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.3),
    )

    ax.text(
        slope_x + 1.5,
        slope_y + 0.0,
        "Extract β₁",
        ha="center",
        fontsize=7,
        color="white",
    )
    ax.text(
        slope_x + 1.5,
        slope_y - 0.3,
        "182 slopes",
        ha="center",
        fontsize=7,
        color="white",
    )

    # Mini trajectory plot
    mini_plot_x = slope_x + 2.2
    mini_plot_y = slope_y - 0.6
    # Draw small scatter with line
    t_vals = np.array([0, 1, 2, 3])
    y_vals = np.array([1.0, 0.85, 0.7, 0.55])  # Declining
    scale = 0.1
    for i in range(len(t_vals)):
        ax.plot(
            mini_plot_x + t_vals[i] * scale,
            mini_plot_y - y_vals[i] * scale,
            "wo",
            markersize=2,
        )
    ax.plot(
        mini_plot_x + t_vals * scale, mini_plot_y - y_vals * scale, "y-", linewidth=1.5
    )

    # Arrow from features to slopes
    create_arrow(ax, (feat_x + 2.6, feat_y), (slope_x - 0.1, slope_y), colors["slopes"])

    # ============================================================================
    # SECTION 7: FEATURE SELECTION
    # ============================================================================

    select_x = 14
    select_y = 4

    create_rounded_box(
        ax,
        (select_x, select_y - 0.4),
        2.5,
        1.2,
        "",
        colors["ml"],
        fontsize=10,
    )

    ax.text(
        select_x + 1.25,
        select_y + 0.5,
        "Select Top 20",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )

    ax.text(
        select_x + 1.25,
        select_y + 0.1,
        "by Variance",
        ha="center",
        fontsize=7,
        color="white",
    )

    ax.text(
        select_x + 1.25,
        select_y - 0.2,
        "bsc_mag_p90_slope",
        ha="center",
        fontsize=5,
        style="italic",
        color="lightyellow",
    )

    # Arrow down from slopes to selection
    create_arrow(
        ax,
        (slope_x + 1.5, slope_y - 0.6),
        (select_x + 1.25, select_y + 0.8),
        colors["ml"],
    )

    # ============================================================================
    # SECTION 8: MACHINE LEARNING MODEL
    # ============================================================================

    ml_x = 17.5
    ml_y = 4

    create_rounded_box(ax, (ml_x, ml_y - 0.4), 3, 1.2, "", colors["ml"], fontsize=10)

    ax.text(
        ml_x + 1.5,
        ml_y + 0.5,
        "Random Survival Forest",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    ax.text(
        ml_x + 1.5,
        ml_y + 0.2,
        "1,000 trees",
        ha="center",
        fontsize=7,
        color="white",
    )
    ax.text(
        ml_x + 1.5,
        ml_y - 0.1,
        "Input: 20 slopes",
        ha="center",
        fontsize=6,
        color="white",
    )

    create_arrow(ax, (select_x + 2.6, select_y), (ml_x - 0.1, ml_y), colors["ml"])

    # ============================================================================
    # SECTION 9: OUTPUTS
    # ============================================================================

    output_x = 21.5
    output_y = 6

    # Risk stratification box
    create_rounded_box(
        ax, (output_x, output_y + 1), 2, 1.5, "", colors["output"], fontsize=10
    )
    ax.text(
        output_x + 1,
        output_y + 2.1,
        "Risk Groups",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    ax.text(
        output_x + 1,
        output_y + 1.7,
        "High/Med/Low",
        ha="center",
        fontsize=7,
        color="white",
    )
    ax.text(
        output_x + 1,
        output_y + 1.3,
        "Kaplan-Meier",
        ha="center",
        fontsize=6,
        color="white",
    )

    # Performance box
    create_rounded_box(
        ax, (output_x, output_y - 0.5), 2, 1.2, "", colors["output"], fontsize=10
    )
    ax.text(
        output_x + 1,
        output_y + 0.4,
        "Performance",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="white",
    )
    ax.text(
        output_x + 1,
        output_y + 0.05,
        "C-index: 0.63",
        ha="center",
        fontsize=8,
        color="white",
        fontweight="bold",
    )
    ax.text(
        output_x + 1,
        output_y - 0.25,
        "163% gain",
        ha="center",
        fontsize=7,
        color="white",
    )

    create_arrow(
        ax, (ml_x + 3.1, ml_y + 0.4), (output_x - 0.1, output_y + 1.5), colors["output"]
    )
    create_arrow(
        ax, (ml_x + 3.1, ml_y - 0.2), (output_x - 0.1, output_y + 0.2), colors["output"]
    )

    # ============================================================================
    # KEY INSIGHTS BOX (Bottom Left)
    # ============================================================================

    # Add legend/key insights
    legend_x = 0.5
    legend_y = 2.5

    ax.text(
        legend_x + 1.2,
        legend_y,
        "Key Insights:",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

    insights = [
        "• 450 subjects, 95 converters",
        "• Follow-up: 4.84 years",
        "• Slopes >> Baseline",
        "• Top: high percentiles",
        "• RSF > Parametric",
    ]

    for i, insight in enumerate(insights):
        ax.text(
            legend_x + 1.2,
            legend_y - 0.35 - i * 0.25,
            insight,
            ha="left",
            fontsize=6,
            color="black",
        )

    # Add a border around key insights
    insight_border = Rectangle(
        (legend_x, legend_y - 1.8),
        2.2,
        2.1,
        facecolor="lightyellow",
        edgecolor="black",
        linewidth=2,
        alpha=0.3,
    )
    ax.add_patch(insight_border)

    # ============================================================================
    # FINAL TOUCHES
    # ============================================================================

    plt.tight_layout()
    return fig, ax


def main():
    """Generate and save the pipeline diagram."""
    print("Generating BSC pipeline diagram...")

    fig, ax = generate_pipeline_diagram()

    # Save as high-res PNG
    output_png = "figs/bsc_pipeline_diagram.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved PNG: {output_png}")

    # Save as PDF (vector format for paper)
    output_pdf = "figs/bsc_pipeline_diagram.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved PDF: {output_pdf}")

    plt.show()

    print("\n" + "=" * 60)
    print("Pipeline diagram generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
