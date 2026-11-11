# Pre-specification: what is gBSC a defined positive for?

Written 2026-11-10, before any model in this folder was run. Committed before
the first result. Anything not named here is exploratory and is labelled as
such in its own output.

## Why this exists

Four previous rounds of this project reported nulls for MCI-to-dementia
prognosis, across global, regional, AD-signature, vertex-wise and
scale-invariant formulations, two follow-up designs, four landmarks, and
harmonized and unharmonized features. Three failures in those rounds were
caused by analysis freedom rather than by the measure:

- a Benjamini-Hochberg step-up run in the wrong direction once reported 158 of
  159 slopes surviving FDR when the correct answer was 2
- a vertex-wise cluster at cwp 0.006 shrank to d -0.16 with a 95 percent
  interval crossing zero under split-half validation
- a cosBSC increment whose bootstrap interval excluded zero at one seed went
  negative at three other seeds

A search for "anything positive" across four measures, 68 parcels and several
cohorts will return something at p < 0.05 whether or not anything is there.
This document fixes the targets, the measures, the covariates and the stopping
rules in advance so that it cannot.

## Scope decided with the user

In scope: (a) measurement positives, meaning defined and reproducible
properties of the measure itself, and (c) prognostic positives. Group
difference contrasts in OASIS are out of scope by user decision.

Primary measure is gBSC as published (`bsc_dir_*`, `bscdir_roi*`,
`bscdir_adsig`), because that is the measure the manuscripts describe. cosBSC
(`bsc_cos_*`, `cos_roi*`, `cos_adsig`, and the regional `bscratio_roi*`
columns, which are cosine means despite the prefix) is carried alongside as a
pre-specified second measure, not as a substitute.

Blocked, not abandoned: the tissue intensity ratio `bscsigratio_*`, the direct
analogue of the published gray/white contrast measure, lives only on Trillium
at `/scratch/ishaan/adni_trillium/sigmoid/features_nn/per_scan_sigmoid.csv`.
Trillium rejected the SSH key on 2026-11-10. That arm runs when access
returns and is pre-specified here so it cannot be rewritten afterwards.

## Naming trap recorded

`bscratio_*` is used by two different pipelines for two different quantities.
In `per_scan_cos_regional.csv` and `per_scan_cosratio.csv` it is the cosine
measure, bounded in [0,1]; observed range 0.012 to 0.9999, never above 1. In
the CIVET sigmoid pipeline (`extract_sigmoid_features.py`) the same prefix
means `model_ratio`, the WM_25/GM_25 tissue intensity ratio, which is above 1
in a T1w image. `fs_sigmoid_features.py` avoids the clash by writing
`bscsigratio_*`. Any analysis that mixes these two is wrong. Verify the value
range before using a column with this prefix.

## A1. Gating analysis: scan-rescan reliability

Run first. Every other result is interpreted against it.

Sample: ADNI subjects with both a screening (`sc`) and a baseline (`bl`) T1,
using real IDA acquisition dates from `manifest_realdates.csv`. Expected
around 67 pairs at a median interval near 23 days. Pairs are retained only
where both real dates are present and the interval is under 180 days.

Estimand: ICC(1,1), absolute agreement, single measurement, on the
cortex-wide mean of each measure, with a 95 percent interval. Reported
separately for gBSC, cosBSC, and the AD-signature composite.

Pre-specified reading:
- ICC below 0.5 means the measure cannot support any of the association tests
  below at the available sample sizes. In that case A2 to A4 are reported as
  underpowered by construction, and no positive claimed from them.
- ICC at or above 0.75 means the previous nulls are strong rather than
  uninformative, which is itself a reportable measurement positive.
- Between 0.5 and 0.75, associations are reported with the attenuation bound
  stated alongside each one.

A rescan interval of 23 days carries real biological change of essentially
zero for these measures, so disagreement across a pair is measurement
variance.

## A2. Primary biological target: white matter hyperintensity burden

Rationale, with sources. The gray/white contrast literature reports
consistent positives where this project reports nulls: contrast declines with
age across most of the cortex independently of cortical thinning (Salat et al.
2009, Neurobiology of Aging, PMC2750073; Nature Scientific Reports 2025,
s41598-025-34355-y); it is reduced in schizophrenia (Jorgensen et al. 2012,
PLoS ONE 7:e37016); it is sensitive in Parkinson's disease, where it is
described as a very sensitive measure for detecting aging effects
(PMC5881246); it is altered early in familial Alzheimer's disease
(PMC12777538); and alterations precede amyloid-beta positivity and
macrostructural change in older adults without dementia (Pieperhoff et al.
2026, Alzheimer's and Dementia, doi 10.1002/alz.71609). The mechanism those
papers propose for age-related contrast loss is increased white matter water
content with white matter signal approaching gray matter. White matter
hyperintensity burden is the direct measure of that process, so it is the
association most likely to be real and the one with the clearest mechanism.

Note on direction and on the literature gap: a targeted search found no study
directly testing gray/white contrast against WMH burden, so this test is not a
replication. Both WMH and contrast loss increase with age, so the expected
direction is that higher WMH goes with lower boundary sharpness.

Data: `ADNIMERGE2/data/UCD_WMH.rda`, 7,437 rows, 2,296 subjects, of whom 1,050
have gBSC. Outcome variable `TOTAL_WMH`, log transformed, as its distribution
is strongly right skewed. Independent of our pipeline: measured by a different
lab on a different modality, so there is no shared-processing circularity.

Design: cross-sectional, one scan per subject, the BSC scan closest in time to
the WMH measurement, requiring the gap to be under 365 days. Cross-sectional
rather than longitudinal because the slope framing is what failed before and
because the scan-rescan variance above bounds slope reliability.

Primary test: linear model, cortex-wide mean gBSC on log TOTAL_WMH, adjusted
for age, sex, intracranial volume, field strength and scanner site. One test,
one measure, one coefficient. Significance at two-sided p < 0.05.

Secondary, in this order and only if the primary is significant: the same test
for cosBSC; the same test for the AD-signature composite; then the 68 regional
parcels with Benjamini-Hochberg correction, step-up, verified by the
`raw[::-1].cummin()[::-1]` idiom given the earlier bug.

Confound control, declared before looking: site, vendor and field strength are
confounded with case mix in ADNI, and the batch sweep in this project showed
that only site-aware harmonization moves the scanner effect, at plus 24.7
percent, while field strength alone does essentially nothing at minus 1.8
percent. The primary test therefore adjusts for site and is repeated within
the 3T stratum alone. A positive that does not survive the 3T-only repeat is
reported as scanner, not biology.

Split-half validation is mandatory, not optional: 200 stratified splits,
coefficient estimated in one half and evaluated in the other, reporting the
held-out estimate and its shrinkage. The full-sample coefficient is not the
headline if it does not hold out.

## A3. Literature anchor: age

Not a discovery test. Contrast loss with age is the single most reproducible
finding in this literature, so a measure that does not show it is not
measuring tissue contrast. An earlier run of this project found
cross-sectional gBSC against age at r = -0.007 with p = 0.74, which is the
anchor failing.

Test: cortex-wide mean of each measure on age, one scan per subject, adjusted
for sex, field strength and site. Pre-specified reading: failure to recover an
age effect is evidence for the measurement-positive claim (a), namely that
gBSC is dominated by acquisition variance rather than tissue contrast. It is
reported as such rather than buried.

## A4. Autopsy, as a pre-specified secondary

Sample: `ADNIMERGE2/data/NEUROPATH.rda`, 121 subjects, all with complete
NPBRAAK, NPTHAL, NPADNC and NPNEUR. 95 of them have gBSC. This is the only
outcome available that is not a clinical proxy.

Primary test: one ordinal regression of Braak stage on cortex-wide mean gBSC,
adjusted for age at death, sex and field strength, using the BSC scan closest
to death.

Hard constraint, fixed now because the temptation is obvious: at n = 95 this
is powered only for a large effect. If the global test is null, no regional
follow-up is permitted and no other neuropathology variable is substituted.
If it is null, that is the reported result.

## A5. Prognostic arm (c), on untested endpoints

Conversion to dementia is exhausted in this cohort and is not retested here.
Two endpoints this project has never used:

- all-cause mortality as a competing risk, from the already built
  `analysis/results/nireports/competing_risks_subject_status.csv`, 417
  subjects, using `cr_time` and `cr_status`. Death dates are a median 0.95
  years after the last clinical visit and ADNI3/4 records give year only, so
  the time axis is coarse and this is stated with the result.
- rate of cognitive decline as a continuous outcome rather than a diagnostic
  event, using the ADNI composite z-scores `ADNI_MEM` and `ADNI_EF` from
  `UWNPSYCHSUM.rda` and `mPACCdigit` from `PACC.rda`.

Test for each: does the measure add over the covariate block of age, sex,
education and baseline cognition. Fold-matched increment with a bootstrap
interval, plus the pre-registered equivalence margin of 0.03 C-index already
fixed in this project for the survival endpoint. The penalized Cox likelihood
ratio test is not used; the L2 penalty invalidates its null, which is why it
was dropped from the manuscript.

Seed stability is mandatory: any interval that excludes zero at the default
seed is rerun at seeds 101, 202 and 303 and reported as an artifact if the
sign changes. This rule exists because a cosBSC result in this project did
exactly that.

## Stopping rules

- A1 runs first and conditions the interpretation of everything else.
- A2 has exactly one primary test. Secondaries run only in the stated order
  and only after the primary is significant.
- A4 gets no regional follow-up on a null.
- Nothing in A5 is reported as positive on a single seed.
- Every result in this folder is written out with a flag saying whether it was
  pre-specified here or exploratory. Exploratory results may be reported but
  not claimed.
- No new cluster compute is spent on any of this unless a pre-specified
  primary test is positive and survives its split-half and stratum checks.

## What would count as a positive

Measurement positive (a): a defined ICC for the published measure, plus the
age-anchor result in whichever direction it falls, plus the already
established estimator floor at c near 0.059 and the 13-fold within-subject CV
reduction from the cosine reformulation.

Biological positive (b, by the single route left in scope): WMH burden
associates with boundary sharpness, survives adjustment for site, survives the
3T-only repeat, and holds out under split-half.

Prognostic positive (c): a measure adds over covariates for mortality or
cognitive decline rate, with a seed-stable interval excluding zero.
