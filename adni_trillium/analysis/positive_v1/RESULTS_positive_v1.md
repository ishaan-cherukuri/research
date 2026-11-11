# Results: what gBSC is a defined positive for

Run 2026-11-10 against `PRESPEC_positive_v1.md`, which was committed before
the first model (commit 4d7bccb). Every number below comes from a script in
this folder and its CSV output. Literature in `SOURCES.md`.

## Summary

Two positives, three nulls, two artifacts caught.

| Analysis | Result |
|---|---|
| A1 same-scanner reliability | **POSITIVE.** gBSC ICC(1,1) 0.843 [0.819, 0.863] |
| A1 cross-field reproducibility | **POSITIVE finding.** collapses to 0.189 |
| A2 WMH burden | Null. beta -0.006 SD/SD, p=0.83 |
| A3 age anchor, cosBSC | **POSITIVE.** -0.092 SD/SD, p=1.2e-04 |
| A3 age anchor, gBSC | Null. -0.014 SD/SD, p=0.56. Anchor fails |
| A4 autopsy Braak | Null. OR 0.88 per SD, p=0.51 |
| A5 mortality, cognitive decline | Null. Two seed artifacts caught |

## A1. The measurement positive

`a1_test_retest_icc.py`, output `a1_icc.csv`.

On 680 within-subject scan pairs at the same field strength and the same
scanner model, one shortest-gap pair per subject, median gap 112 days:

| measure | ICC(1,1) | 95% CI | within-subject CV |
|---|---|---|---|
| gBSC directional | **0.843** | 0.819, 0.863 | 10.2% |
| gBSC magnitude | 0.836 | 0.812, 0.857 | 10.5% |
| gBSC AD-signature | 0.819 | 0.790, 0.845 | 16.2% |
| cosBSC | 0.679 | 0.637, 0.718 | 1.4% |
| cosBSC AD-signature | 0.658 | 0.614, 0.699 | 3.9% |
| CONTROL ICV | 0.914 | 0.900, 0.925 | 5.5% |
| CONTROL brain volume | 0.921 | 0.909, 0.932 | 9.2% |
| CONTROL BPF | 0.856 | 0.835, 0.875 | 7.9% |
| CONTROL SNR | 0.955 | 0.948, 0.961 | 2.4% |

gBSC is about as reliable as brain parenchymal fraction from the same
pipeline. Because the median gap is 112 days and carries some genuine
biological change, 0.843 is a lower bound.

**This is the result that matters most for the existing manuscripts.** The
prognostic nulls are strong rather than noise-limited: the measure agrees with
itself, it just does not track the outcome. Every previous version of this
paper lacked that number, and a reviewer could have used its absence to
dismiss the whole thing.

### Two things found on the way

**The screening-versus-baseline pairs are not a test-retest sample.** 81
subjects have both, 67 with real dates, median 23 days apart. But ADNI1
deliberately scanned subjects at 1.5T and 3T, so 65 of 67 straddle a field
strength change. There everything collapses, including the controls:

| measure | cross-field ICC |
|---|---|
| gBSC directional | 0.189 [-0.054, 0.412] |
| cosBSC | 0.150 [-0.095, 0.378] |
| gBSC AD-signature | -0.306 [-0.565, 0.010] |
| CONTROL ICV | 0.518 |
| CONTROL brain volume | 0.307 |
| CONTROL SNR | 0.054 |

Reported separately rather than pooled. Since even ICV drops to 0.52, this is
a statement about the whole Atropos pipeline across field strengths, not about
gBSC specifically. It does say that no cross-field-strength comparison in this
project's data is safe without harmonization, which is consistent with the
earlier finding that only site-aware ComBat batches help.

**cosBSC is less reliable than gBSC** (0.679 vs 0.843) despite a 13-fold
smaller within-subject CV. The reason is visible in the table: its
within-subject CV is 1.4%, but its between-subject spread shrinks faster
still. That is a mechanistic explanation for why the scale-invariant
reformulation added no prognostic value. It buys stability by compressing
between-subject differences, and ICC is a ratio of the two.

## A2. White matter hyperintensity burden: null

`a2_wmh.py`, output `a2_wmh_results.csv`.

1,038 subjects with a UCD FLAIR WMH measurement, median join gap 0 days
(same session). All 3T, since WMH needs FLAIR, so the planned 3T-only guard
is vacuous here.

Primary, pre-specified: cortex-wide gBSC on log WMH volume, adjusted for age,
sex, ICV, field strength and site. **beta = -0.006 SD per SD, p = 0.83.**
Split-half over 200 splits: only 1% of held-out halves significant, sign
agreement 0.20, shrinkage incoherent.

Secondaries and the regional screen were **not run**, because the pre-spec
forbids them after a null primary.

The null is trustworthy rather than a setup failure: brain parenchymal
fraction, from the same pipeline, the same scans and the same covariate block,
does track WMH at **-0.132 SD per SD, p = 2.1e-06**.

## A3. The age anchor: a positive, and not for the published measure

`a3_age.py`, outputs `a3_age_results.csv` and `a3_age_guards.csv`.

1,601 subjects, one scan each, adjusted for sex, field strength and site.

| measure | beta per SD | p |
|---|---|---|
| **cosBSC** | **-0.092** | **1.2e-04** |
| **cosBSC AD-signature** | **-0.092** | **2.0e-04** |
| gBSC directional | -0.014 | 0.56 |
| gBSC magnitude | -0.003 | 0.89 |
| gBSC AD-signature | -0.027 | 0.40 |
| REFERENCE BPF | -0.105 | 2.0e-06 |
| REFERENCE ICV | -0.134 | 2.3e-14 |
| REFERENCE SNR | -0.095 | 1.1e-06 |

Age-related loss of gray/white contrast is the most reproducible finding in
this literature (Salat PMC2750073; Sci Rep s41598-025-34355-y; PMC5881246).
The scale-invariant cosine reformulation recovers it at an effect size
comparable to brain parenchymal fraction. The gradient measure the
manuscripts describe does not recover it at all.

That is coherent rather than lucky. The published GWC measure is a ratio of
tissue intensities; our gBSC is the magnitude of a gradient, which scales with
image intensity and is therefore sensitive to acquisition gain. Dividing that
dependence out restores the effect.

### Guards applied to the positive, not only to the null

| guard | beta per SD | p |
|---|---|---|
| 3T only (n=1120) | -0.093 | 4.3e-04 |
| + SNR adjusted | -0.068 | 0.0039 |
| + ICV adjusted | -0.061 | 0.011 |
| + diagnosis adjusted | -0.083 | 5.2e-04 |
| + SNR, ICV and diagnosis jointly | -0.045 | **0.059, does not survive** |
| split-half, 200 splits | held-out CI excludes zero; 99% sign agreement; 81% of held-out halves significant | |

Honest reading: a real, out-of-sample-replicable age association, about 40% of
which is shared with image quality and head size and is not separable from
them at this n. The split-half behaviour is a different world from the
vertex-wise cluster that held out in 4% of splits.

## A4. Autopsy: null

`a4_autopsy.py`, output `a4_autopsy_results.csv`.

95 subjects with complete neuropathology and gBSC, median age at death 82.
Ordinal logit of Braak stage, adjusted for age at death, sex and field
strength.

| measure | OR per SD | 95% CI | p |
|---|---|---|---|
| gBSC directional | 0.882 | 0.609, 1.277 | 0.51 |
| gBSC magnitude | 0.888 | 0.612, 1.289 | 0.53 |
| cosBSC | 0.995 | 0.662, 1.494 | 0.98 |
| gBSC AD-signature (n=52) | 0.834 | 0.496, 1.400 | 0.49 |
| cosBSC AD-signature | 1.310 | 0.887, 1.934 | 0.18 |

No secondary pathology outcome and no regional follow-up was run, per the
pre-spec constraint written before the result was seen.

Power limitation, stated in advance: the sample is enriched for advanced
disease. 71 of 95 are Braak 5 or 6, so the usable outcome range is compressed
and this is weaker than n = 95 suggests.

## A5. Prognostic arm on untested endpoints: null, two artifacts caught

`a5_prognostic.py`, output `a5_prognostic_results.csv`. Measures taken at or
before the landmark, so nothing here repeats the guarantee-time bias.

**Mortality before dementia**, 28 events in 417 subjects. Covariates alone
C = 0.587.

| measure | delta C at seed 42 | 95% CI | across seeds 42/101/202/303 |
|---|---|---|---|
| gBSC baseline | -0.0224 | -0.030, -0.010 | -0.022, -0.018, -0.006, **+0.005** |
| cosBSC baseline | +0.0304 | -0.357, +0.452 | +0.030, +0.086, +0.061, **-0.031** |

The gBSC interval excluded zero at the default seed. It flips sign across
seeds, so the pre-specified rule marks it an artifact. This is the third time
that rule has caught something in this project, which is the argument for
keeping it.

**Rate of cognitive decline**, continuous, post-landmark slopes:

| endpoint | n | measure | beta per SD | p |
|---|---|---|---|---|
| ADNI_MEM slope | 126 | gBSC | -0.119 | 0.24 |
| ADNI_MEM slope | 126 | cosBSC | +0.007 | 0.94 |
| ADNI_EF slope | 122 | gBSC | -0.145 | 0.15 |
| ADNI_EF slope | 122 | cosBSC | -0.049 | 0.58 |

Underpowered and labelled so: only 126 of 417 subjects have three or more
post-landmark cognitive visits spanning at least a year.

## Blocked

The tissue intensity ratio `bscsigratio_*`, the direct analogue of the
published GWC measure that carries all the positive findings in the
literature, lives only on Trillium at
`/scratch/ishaan/adni_trillium/sigmoid/features_nn/per_scan_sigmoid.csv`.
Trillium refused the SSH key on 2026-11-10 with
`Permission denied (keyboard-interactive,hostbased)`, which is authentication
rather than the cluster being down. Given the A3 result, this is now the
highest-value remaining analysis and it is pre-specified in A2 and A3 so it
cannot be rewritten after the fact.

## What this adds up to

gBSC is a defined positive for **same-scanner measurement reliability**
(ICC 0.843, comparable to conventional morphometry) and a defined negative
for everything biological tested here. It does not track WMH, age, Braak
stage, mortality or cognitive decline rate.

cosBSC is a defined positive for **age**, the one anchor this literature
agrees on, at an effect size matching brain parenchymal fraction, with about
40% of it shared with image quality and head size.

The most useful framing for a manuscript is that the measure class splits:
the intensity ratio the literature uses has real tissue-contrast content, and
the intensity gradient this project built does not, because it inherits
acquisition gain. The A1 reliability number and the A3 cosine-versus-gradient
contrast are the two results worth writing up.
