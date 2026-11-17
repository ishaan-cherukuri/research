# Vertex-wise gBSC: variant results

Run 2026-10-27 to 2026-10-29 on Trillium. 416 subjects, 133 converters, gBSC slopes on
fsaverage6 at 10 mm FWHM, cluster-forming threshold p < 0.01, 10,000 permutations under
Freedman-Lane. Model A is group plus covariates; Model B adds thickness and curvature slopes
as per-vertex regressors.

## 1. The whole-cortex null stands

Unweighted, two-sided, whole cortex, group and age and sex: nothing survives in either model.
The largest observed cluster reaches 0.32 of the critical area in A and 0.38 in B, so it is not
close. This was the original result and none of the variants below overturns it.

## 2. A pre-specified test does give a cluster

Restricting the search to the seven-parcel AD signature, weighting subjects by slope precision,
and testing one-sided negative (the only direction the hypothesis allows):

| spec | n / converters | cluster | area | peak t | cwp |
|---|---|---|---|---|---|
| Model A | 416 / 133 | rh inferiortemporal | 710 mm2 | -2.96 | 0.017 |
| Model B | 416 / 133 | rh inferiortemporal | 1153 mm2 | -3.12 | **0.0062** |

Model B survives adjustment for thickness and curvature change at the same vertex, which is
the whole point of the exercise: the signal is not a restatement of atrophy.

### It is not a permutation artifact

Four permutation seeds, 10,000 permutations each, Model B:

| seed | cwp |
|---|---|
| 0 | 0.0062 |
| 101 | 0.0047 |
| 202 | 0.0049 |
| 303 | 0.0065 |

Identical cluster, identical extent, identical peak vertex. The spread is Monte Carlo noise on
the null distribution and nothing more.

### The field-strength confound does not explain it

3 T covers 75 per cent of non-converters and 50 per cent of converters, so this was the leading
alternative explanation for two sessions. It does not hold for this cluster. Taking the Model B
cluster mean per subject:

| sample | n / conv | beta | t | p | d |
|---|---|---|---|---|---|
| all | 416 / 133 | -8.52e-03 | -3.05 | 0.0024 | -0.362 |
| field-stable | 366 / 114 | -9.57e-03 | -3.15 | 0.0018 | -0.427 |
| 3 T only | 281 / 67 | -1.04e-02 | -2.50 | 0.0130 | -0.425 |
| 1.5 T only | 135 / 66 | -4.52e-03 | -1.37 | 0.1727 | -0.214 |

In one pooled interaction model (df = 410) the converter term is t = -2.97, p = 0.0032, while
field strength is t = 0.51, p = 0.61 and the converter by field interaction is t = -0.95,
p = 0.34. The 3 T stratum alone reproduces the pooled effect size. The 1.5 T stratum is simply
small.

A note on reading the strata by cluster extent instead: do not. Cluster extent scales with
sample size and with spatial coherence, so a stratum showing a smaller cluster has not failed
to replicate. Only the effect size answers that question.

## 3. Split-half validation is where it comes apart

Every number above chose the cluster and measured it on the same 416 subjects. 200 stratified
splits, selecting the Model B cluster in one half and measuring it in the held-out half:

| | training half | held-out half |
|---|---|---|
| median d | -0.436 | -0.164 |
| IQR | -0.490 to -0.384 | -0.218 to -0.076 |
| 95 per cent range | -0.564 to -0.231 | -0.319 to +0.218 |

Shrinkage 0.37. A cluster formed in the training half on all 200 splits. In the held-out half
84 per cent of splits point the hypothesised way but only 4 per cent reach p < 0.05.

So the full-sample d of -0.36 is inflated by roughly a factor of two and a half by selecting
and testing on the same subjects. The honest estimate is around -0.16.

### What survives this, and it is not nothing

Two things in the split-half file argue the effect is real but small rather than absent.

The localisation is consistent. 170 of 200 splits put their training cluster in the right
hemisphere and 30 in the left. The right-hemisphere splits replicate: held-out d -0.185,
negative in 92 per cent of splits, shrinkage 0.42. The left-hemisphere splits do not: held-out
d +0.056, negative in 37 per cent, which is what noise looks like. Left-hemisphere clusters are
therefore dragging the pooled held-out estimate down.

The significance is one-sided. Across 200 splits, 7 reached p < 0.05 in the hypothesised
direction and 1 in the opposite direction. Under a true null those counts should be about equal.
The splits overlap heavily so this is not a valid test and no p-value should be quoted for it,
but the asymmetry is the clearest signal in the file.

### What it would take

At the held-out effect size, with 32 per cent converter prevalence and 80 per cent power:

| assumed d | subjects needed | converters |
|---|---|---|
| 0.164 (all splits) | about 1,340 | about 430 |
| 0.185 (right hemisphere) | about 1,050 | about 340 |

This cohort has 416 and 133. It is underpowered for its own effect by a factor of about two and
a half, which is exactly what the shrinkage predicted.

## 4. The cluster is not fragile, and the first QC run said otherwise for a bad reason

Dropping subjects above the 97.5th percentile of cortex-mean absolute slope, taken over the
pooled sample, destroys the cluster: 1153 mm2 becomes fragments of 344, 241 and 127 mm2, best
cwp 0.059, and the peak moves from inferiortemporal to middletemporal. Read on its own that
looks like severe fragility to 2.6 per cent of the sample.

It is not. That filter is outcome-blind in construction but not outcome-neutral in effect. It
dropped 6 converters out of 11 subjects against a 32 per cent base rate. Converters are
enriched in the upper tail of absolute slope precisely because they change faster, which is the
hypothesis, so a pooled cut removes the signal along with the noise.

Taking the same percentile inside each group fixes that. The cut then removed 12 subjects of
whom 4 were converters, 33 per cent against the 32 per cent base rate, and the cluster holds:

| spec | n / conv | cluster | area | peak t | cwp |
|---|---|---|---|---|---|
| full sample, Model B | 416 / 133 | rh inferiortemporal | 1153 mm2 | -3.12 | 0.0062 |
| stratified QC, Model B | 404 / 129 | rh inferiortemporal | 941 mm2 | -3.18 | 0.0093 |
| stratified QC, Model A | 404 / 129 | rh inferiortemporal | 479 mm2 | -2.86 | 0.032 |

Same parcel, same sign, a slightly stronger peak, and an extent reduced about in proportion to
the lost sample. Model A survives too, which it did not in the pooled version. So the cluster
is robust to outlier removal once the removal is not aimed at the effect.

The general lesson is worth keeping: an exclusion rule that never looks at the outcome can
still be correlated with it through the very mechanism under test, and then a loss of
significance after exclusion is uninterpretable rather than informative.

## 5. The sigmoid features were not usable at all, and are now rebuilt

The native per-vertex sigmoid fits are 92.7 per cent finite, because the fit does not converge
everywhere and the failures are scattered. `mri_surf2surf` propagates NaN through its kernel,
so resampling those maps to fsaverage6 at 20 mm FWHM turned 7 per cent missing into 66 per cent
missing. `model_ratio` is resampled unsmoothed and was never affected.

The first reading of that was too generous. It is not that each ROI mean was an average over a
biased third of its parcel. For most parcels there was no mean at all. Coverage of the seven
AD-signature parcels in the shipped `sigmoid/features/per_scan_sigmoid.csv`, as a percentage of
the 2,251 scans holding any value:

| parcel | old lh | old rh | rebuilt lh | rebuilt rh |
|---|---|---|---|---|
| entorhinal | 0.0% | 0.0% | 100% | 100% |
| parahippocampal | 0.0% | 0.0% | 100% | 100% |
| inferiortemporal | 61.7% | 0.0% | 100% | 100% |
| middletemporal | 97.6% | 97.6% | 100% | 100% |
| precuneus | 0.0% | 0.0% | 100% | 100% |
| posteriorcingulate | 0.0% | 0.0% | 100% | 100% |
| fusiform | 0.0% | 0.0% | 100% | 100% |

Five of the seven signature parcels were empty in both hemispheres for every scan. Across all
68 parcels, 58.0 per cent of the `bscsig_*` and `bscsigfree_*` ROI columns were entirely NaN.
So the `bscsig_adsig` composite, which is documented as a seven-parcel AD signature, was in
practice middletemporal plus a partial left inferiortemporal and nothing else. That is why it
still correlates 0.934 with the rebuilt composite: both track middletemporal. The old one was a
single-parcel measure wearing a seven-parcel label.

**Anything fitted on `bscsig_*` or `bscsigfree_*` ROI features needs rerunning.** The affected
columns are whatever `build_spec_cohort.py` drew from that CSV. `bscsigratio_*` is unaffected
and matches the rebuild at r = 1.000 across every column, which is a useful control: it is the
one measure resampled at FWHM 0.

The fix is normalized convolution in `fs_sigmoid_features.py`: the valid values and a validity
mask get the same kernel and are divided, keeping vertices with at least half the kernel on
real data. Vertex coverage goes from 31.5 to 93.2 per cent. On the vertices both methods keep,
the values are identical at r = 1.000 with a median absolute difference of 0, which is the
expected result and therefore a good correctness check: a vertex that survives plain smoothing
had no NaN anywhere in its kernel, so its denominator is exactly 1. The old values were right
where they existed. What was wrong was which vertices existed, and for five of seven signature
parcels the answer was none.

The rebuild (job 2700648, 4m44s on 120 workers) covers 2,251 scans and 417 subjects, matching
the old file's row count exactly, and writes `sigmoid/features_nn`. The old outputs are kept
rather than overwritten, since the comparison is the evidence above. No infinities in either
file.

Note that the sigmoid pipeline smooths at 20 mm against the gradient pipeline's 10 mm, so
cluster extents between the two are not comparable.

Operational trap worth keeping: `resample()` returns an existing target map untouched, so a
cache written by the old method is silently reused and the rebuild would have been a no-op.
The cache directory is now positional 6 on `slurm/run_sigmoid_features.sh` for that reason.

## 6. Method checks

- Permutation inference here is conservative, not liberal. 100 synthetic null trials with
  spatially correlated noise reject at 2 per cent for a nominal 5 per cent, 3 for 10, and 8 for
  20. An earlier calibration without spatial correlation was worthless, and said so by rejecting
  at the same rate for every alpha.
- On a synthetic world with a planted effect the pipeline recovers all 144 planted vertices in
  the right parcel, peak t -19.9, cwp 0.0005.
- Weighted least squares matches the closed form to 12 decimals, and the unweighted path is
  bitwise unchanged from `HEAD` for both slopes and GLM.
- Weights must be bounded. Raw slope precision spans 128-fold; unbounded inverse-variance
  weights took effective n from 40 to 17 on synthetic data. Hard clipping fails differently,
  saturating every subject so the weight degenerates into a two-level factor. The shipped
  compression raises precision to the power that brings the 5th-to-95th ratio to 4, keeping 86
  per cent of effective n at Spearman 0.9998 with raw precision.

## What this means for the paper

The defensible claim is narrower than cwp 0.0062 suggests and wider than the original null.
Longitudinal gBSC change in right inferior temporal and fusiform cortex is associated with
MCI-to-dementia conversion independently of thickness and curvature change, with a small effect
of about d = 0.18 that this cohort cannot establish on its own. The whole-cortex search is
negative. Both the in-sample cluster statistic and the split-half shrinkage belong in the
paper; reporting the first without the second would be the error.
