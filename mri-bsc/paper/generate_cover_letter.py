"""Generate cover_letter.pdf for JCN Springer Nature submission."""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from pathlib import Path

OUT = Path(__file__).parent / "cover_letter.pdf"

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=LETTER,
    leftMargin=1.1 * inch,
    rightMargin=1.1 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
)

# ── Styles ─────────────────────────────────────────────────────────────────────
base = ParagraphStyle(
    "base",
    fontName="Times-Roman",
    fontSize=10.5,
    leading=15,
    textColor=colors.black,
)
bold_style = ParagraphStyle("bold", parent=base, fontName="Times-Bold")
justify = ParagraphStyle("justify", parent=base, alignment=TA_JUSTIFY)
left = ParagraphStyle("left", parent=base, alignment=TA_LEFT)
small_gap = 2
para_gap = 6

# ── Content ────────────────────────────────────────────────────────────────────
story = []

def p(text, style=left, space_after=para_gap):
    story.append(Paragraph(text, style))
    story.append(Spacer(1, space_after))

# Sender block
p("Ishaan Cherukuri", bold_style, small_gap)
p("Independent Researcher", left, small_gap)
p("Princeton, NJ, USA", left, small_gap)
p("ishaan.cherukuri@gmail.com", left, para_gap)

# Date
p("April 19, 2026", left, para_gap)

# Recipient block
p("The Editorial Board", left, small_gap)
p("Journal of Computational Neuroscience", left, small_gap)
p("Springer Nature", left, para_gap)

# Salutation
p("Dear Editors,", left, para_gap)

# Body paragraph 1: submission statement
p(
    "I am pleased to submit an original research manuscript entitled "
    "<b>Longitudinal Boundary Sharpness Coefficient Slopes Predict Time to "
    "Alzheimer Disease Conversion in Mild Cognitive Impairment: A Survival "
    "Analysis Using the ADNI Cohort</b> for consideration in the "
    "<i>Journal of Computational Neuroscience</i>.",
    justify,
)

# Body paragraph 2: scientific significance
p(
    "Predicting which individuals with mild cognitive impairment will progress "
    "to Alzheimer disease remains one of the most clinically consequential "
    "unsolved problems in computational neuroscience. Most existing approaches "
    "rely on a single baseline scan, yet the structural brain changes that "
    "signal impending conversion unfold over months and years. This manuscript "
    "addresses that gap by introducing longitudinal slope features derived from "
    "the Boundary Sharpness Coefficient, a voxel-level measure of gray-white "
    "matter interface integrity computed from standard T1-weighted MRI. Across "
    "1,824 scans from 450 ADNI participants, annualized rates of boundary "
    "degradation modeled with a Random Survival Forest achieved a "
    "concordance index of 0.63, representing a 163 percent improvement over "
    "a baseline parametric model. The method requires no PET imaging, no "
    "cerebrospinal fluid collection, and no sequence beyond the structural MRI "
    "already obtained in routine clinical evaluations.",
    justify,
)

# Body paragraph 3: journal fit
p(
    "This work sits at the intersection of computational neuroimaging, survival "
    "analysis, and machine learning applied to neurodegeneration, areas that "
    "fall squarely within the scope of the <i>Journal of Computational "
    "Neuroscience</i>. The manuscript will be of direct interest to readers "
    "working on quantitative brain imaging biomarkers, longitudinal modeling of "
    "neural tissue, and data-driven approaches to clinical risk stratification. "
    "The proposed framework is reproducible, interpretable, and designed with "
    "real-world deployment constraints in mind.",
    justify,
)

# Body paragraph 4: declarations
p(
    "This manuscript is original, has not been published previously, and is not "
    "currently under consideration by any other journal. I have no competing "
    "interests to disclose. As the sole author, I have approved the final "
    "version of the manuscript and take full responsibility for its content.",
    justify,
)

# Closing
p("Thank you for your time and consideration. I look forward to your response.", justify, para_gap)

p("Sincerely,", left, para_gap * 1.5)

p(
    "<b>Ishaan Cherukuri</b> | Independent Researcher | Princeton, NJ, USA | ishaan.cherukuri@gmail.com",
    left, 0
)

doc.build(story)
print(f"Saved -> {OUT}")
