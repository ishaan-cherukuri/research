"""Generate cover_letter_neuroinformatics.pdf for Neuroinformatics (Springer Nature)."""

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors
from pathlib import Path

OUT = Path(__file__).parent / "cover_letter_neuroinformatics.pdf"

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=LETTER,
    leftMargin=1.1 * inch,
    rightMargin=1.1 * inch,
    topMargin=0.85 * inch,
    bottomMargin=0.85 * inch,
)

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
p("Neuroinformatics", left, small_gap)
p("Springer Nature", left, para_gap)

# Salutation
p("Dear Editors,", left, para_gap)

# Body paragraph 1: submission statement
p(
    "I am pleased to submit an original research manuscript entitled "
    "<b>Longitudinal Boundary Sharpness Coefficient Slopes Predict Time to "
    "Alzheimer Disease Conversion in Mild Cognitive Impairment: A Survival "
    "Analysis Using the ADNI Cohort</b> for consideration in "
    "<i>Neuroinformatics</i>.",
    justify,
)

# Body paragraph 2: scientific significance
p(
    "Predicting which individuals with mild cognitive impairment will convert to "
    "Alzheimer disease is a problem that sits at the intersection of neuroimaging "
    "and clinical informatics. Existing tools either rely on expensive molecular "
    "imaging or extract only cross-sectional features from a single scan, missing "
    "the dynamic trajectory of tissue change that unfolds across years. This "
    "manuscript introduces a fully automated pipeline that derives longitudinal "
    "slope features from the Boundary Sharpness Coefficient, a voxel-level "
    "index of gray-white matter interface integrity computed from standard "
    "T1-weighted MRI. Annualized rates of boundary degradation, estimated by "
    "linear regression across four serial scans per subject, are modeled with a "
    "Random Survival Forest to produce individual conversion risk scores. Applied "
    "to 1,824 scans from 450 ADNI participants, the approach achieved a "
    "concordance index of 0.63, a 163 percent improvement over a parametric "
    "baseline, using only structural MRI data already collected in routine "
    "clinical practice.",
    justify,
)

# Body paragraph 3: journal fit
p(
    "Neuroinformatics is a natural home for this work. The manuscript contributes "
    "a reproducible, open neuroimaging pipeline, a longitudinal biomarker "
    "extraction framework, and a survival modeling approach validated on a "
    "large public dataset. These are precisely the kinds of computational tools "
    "and datasets that the neuroinformatics community develops and applies. "
    "Readers working on brain MRI analysis, longitudinal cohort studies, and "
    "data-driven clinical decision support will find the methods and findings "
    "directly relevant.",
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
