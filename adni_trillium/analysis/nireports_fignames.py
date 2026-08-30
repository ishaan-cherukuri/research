"""Where the NeuroImage: Reports figures go, and what they are called there.

The figure generators grew descriptive filenames (fig6_construct_validity.png)
while the manuscript refers to figures by position (fig6.png). Keeping the two
in step by hand is how a caption ends up describing a different plot, so the
mapping lives here and every generator writes through it.

Figure 2 is the pipeline schematic, which is drawn by make_arch_figure.py and is
not regenerated from results.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "mri-bsc/paper/neuroimage_clinical"

# descriptive name used by the generator -> name the manuscript includes
MANUSCRIPT_NAME = {
    "fig_splot_cohort": "fig1",
    "fig0_architecture": "fig2",
    "fig1_model_comparison": "fig3",
    "fig2_kaplan_meier": "fig4",
    "fig4_design_effect": "fig5",
    "fig6_construct_validity": "fig6",
    # fig7 is written directly by nireports_figures.py.
}

# Generated but not included in the manuscript, which reports the increment as
# Table 4 instead. Kept so the generator can still be run.
NOT_IN_MANUSCRIPT = {"fig5_increment"}


def target(name: str) -> Path | None:
    """Path to write `name` to, or None if the manuscript does not include it."""
    if name in NOT_IN_MANUSCRIPT:
        return None
    return OUT / f"{MANUSCRIPT_NAME[name]}.png"
