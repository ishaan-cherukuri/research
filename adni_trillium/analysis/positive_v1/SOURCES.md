# Sources for the boundary sharpness and gray/white contrast literature

Gathered 2026-11-10 while looking for a defined positive. Recorded here with
names and identifiers so the manuscripts can cite them without re-searching.
Each entry says what it supports, because the point of the search was to find
what this measure class has actually been shown to do.

Note on verification: a previous round of this project cited a reference the
spec had guessed, and it turned out to be a different paper by different
authors. Entries below carry the identifier that was actually returned by
search. Anything marked UNVERIFIED has not been opened and checked, and must
be confirmed against the journal page before it goes into a manuscript.

## The measure this project is named after

**Olafson E, Bedford SA, et al. "Examining the Boundary Sharpness Coefficient
as an Index of Cortical Microstructure in Autism Spectrum Disorder." Cerebral
Cortex 2021;31(7):3338-3352. doi:10.1093/cercor/bhab015. PMID 33693614.**
Preprint: bioRxiv 2020.07.09.196212.

The source of the BSC definition. BSC is the growth rate parameter of a
sigmoid fit to the T1w intensity profile crossing the gray/white boundary. In
1,136 individuals, ASD showed greater BSC in bilateral superior temporal gyrus
and left inferior frontal gyrus, with significant spatial overlap between the
diagnosis effect on BSC and on cortical thickness.

**No independent replication was found.** The search returned the paper, its
preprint, and indexing copies, but no replication in ABIDE or elsewhere, by
this group or others. The authors themselves framed it as a call for further
study. This matters for how our null should be positioned: we are not
contradicting a replicated finding.

## Gray/white contrast, which is where the positives in this literature are

This is the key result of the search. The measure with consistent positive
findings is the gray/white intensity contrast ratio (GWC), not the sigmoid
growth rate and not an intensity gradient.

**Salat DH, et al. "Age-Associated Alterations in Cortical Gray and White
Matter Signal Intensity and Gray to White Matter Contrast." Neurobiology of
Aging / PMC2750073.** GWC declines with age across most of the cortex,
independently of cortical thinning. The decrease is driven mostly by white
matter signal falling toward gray matter signal. Proposed mechanism: increased
white matter water content plus neuronal loss in gray matter. This is the
anchor our A3 analysis tests against.

**"Automated characterization of the gray matter white matter distribution
demonstrates age-related decline." Scientific Reports 2025;
s41598-025-34355-y.** Reproduces the age-related decline with an automated
pipeline.

**Jorgensen KN, et al. "Reduced Gray to White Matter Tissue Intensity Contrast
in Schizophrenia." PLoS ONE 2012;7(6):e37016.** GWC reduced in schizophrenia.
Also useful for its own caution that acquisition at different field strengths
may partly explain discrepancies between studies, which is the same problem
this project hit. UNVERIFIED author list.

**"Gray/White Matter Contrast in Parkinson's Disease." PMC5881246.** Computes
GWC by dividing white matter by gray matter values and projecting the ratio
onto a common surface, and describes GWC as a very sensitive measure for
detecting aging effects.

**Pieperhoff P, et al. "Cortical gray-white matter contrast alterations
precede amyloid-beta positivity and macrostructural changes in older adults
without dementia." Alzheimer's and Dementia 2026. doi:10.1002/alz.71609.
Also PMC13290647.** The most directly relevant paper found. GWC alterations
precede amyloid positivity and macrostructural change in non-demented older
adults, and higher amyloid burden was most strongly associated with reduced
GWC and cortical thinning. UNVERIFIED author list beyond the first name shown
in the search result.

**"Grey-White matter contrast as an early brain pathology marker in Familial
Alzheimer's disease." PMC12777538.** GWC as an early marker in familial AD.
UNVERIFIED.

## What the search did NOT find

**No study testing gray/white contrast against white matter hyperintensity
burden.** Our A2 analysis is therefore not a replication of anything. Both WMH
and contrast loss increase with age, so the expected direction is that higher
WMH goes with lower boundary sharpness, but no source establishes it. This was
checked before running A2 and is the reason A2 is framed as a single
pre-specified test rather than a confirmation.

**No study measuring test-retest reliability of gray/white contrast across
field strengths.** The closest evidence is indirect: one age study compared two
field strengths and attributed regional differences to reduced gray/white
contrast in T1w images at the lower strength, while finding global cortical
thickness unaffected. A multiple sclerosis study reported high scan-rescan
reliability for deep gray matter volumes at 3T, which concerns volumes rather
than contrast. This gap is what makes our A1 result worth reporting: a
same-scanner ICC and a cross-field-strength ICC for this measure class does
not appear to exist in the literature.

## How these sources bear on our results

The literature's positives are for a ratio of tissue intensities. Our
published measure is the magnitude of a gradient of an Atropos tissue
probability field, which scales with image intensity and is therefore
sensitive to acquisition gain. A3 found that the scale-invariant cosine
reformulation recovers the age decline these papers report, at an effect size
comparable to brain parenchymal fraction, while the gradient measure does not
recover it at all. That is consistent with the mechanism rather than a
coincidence, and it points at the tissue intensity ratio `bscsigratio_*` on
Trillium as the measure most likely to reproduce the published findings.

## Search provenance

Queries run 2026-11-10, extended mode:
- grey-white matter contrast GWC MRI positive association white matter
  hyperintensities aging
- Olafson boundary sharpness coefficient BSC cortical microstructure autism
  replication findings
- gray white matter contrast GWC test-retest reliability scanner field
  strength effects reproducibility
