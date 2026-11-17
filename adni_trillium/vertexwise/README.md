# Vertex-wise complementary-information analysis

Does gBSC carry spatial information on the cortical surface that cortical
thickness and curvature do not? This pipeline answers that for the frozen
417-subject pre-outcome cohort (133 converters, 2,252 scans) used in the
paper's Section 3, and feeds Section 3.11 and Figure 8 of
`mri-bsc/paper/aperture_neuro/main_aperture.tex`.

## What it does

1. `build_vertex_cohort.py` (local) rebuilds the 2,252-scan window from the
   frozen cohort and the real-date manifest, asserts the count matches, and
   writes `cohort/vertex_scans.csv` and `cohort/vertex_subjects.csv`.
2. `slurm/01_recon_all.sh` runs FreeSurfer 7.4.1 `recon-all -all`
   cross-sectionally on every scan (15 whole nodes, 160 concurrent per node).
3. `slurm/02_sample_surface.sh` recomputes the dense gBSC field per scan
   (`compute_gbsc_field.py`: the stored `bsc_dir_map` is masked to the
   atropos boundary band, a sparse set of voxels that meets the white surface
   at a few percent of vertices and at none in a quarter of scans), samples it
   at the FreeSurfer white surface averaged 0 to 1.5 mm outward along the
   normal (where the field peaks), carries thickness and curvature from the
   same reconstruction, and resamples all three to fsaverage6 with 10 mm FWHM
   smoothing on the target surface.
4. `compute_vertex_slopes.py` divides each scan's gBSC map by its own
   cortex-wide mean (the level moves with scanner by 10% median, up to 2x,
   and dominates raw slopes; `--normalize none` for the raw sensitivity run),
   drops scans whose field is zero over >5% of cortex (empty upstream
   segmentation, 13 scans) or lacks surfaces (1 scan), and fits per-subject
   OLS slopes against years on the real acquisition-date axis.
5. `make_design.py` stacks the slopes and writes the design (converter,
   non-converter, centred age, sex, APOE4 count, 3 T indicator) and contrasts.
6. `slurm/03_glm.sh` runs `vertex_glm.py` for Model A (unadjusted) and
   Model B (thickness and curvature slopes as per-vertex regressors), both
   hemispheres, 10,000 Freedman-Lane permutations, CFT p<0.01 two-sided,
   cluster area on the white surface; `combine_clusters.py` takes the
   per-iteration maximum across hemispheres for the corrected null. Model A
   is also run through `mri_glmfit` / `mri_glmfit-sim --perm` as a cross-check
   (`mri_glmfit` refuses per-vertex regressors under permutation, which is
   why the primary analysis is in Python; on synthetic data the two agree to
   machine precision on betas and p-values).
7. `make_surface_figure.py` (local) renders Figure 8 from fetched results.

## Running it (Trillium login node)

```
cd /project/rrg-mchakrav-ab/ishaan/adni_trillium/vertexwise
bash run_vertexwise.sh debug          # optional 1 h check on 4 scans
bash run_vertexwise.sh recon          # ~1 day wall, 15 nodes
bash run_vertexwise.sh recon-status   # repeat until pending == 0; rerun recon for stragglers
bash run_vertexwise.sh sample
bash run_vertexwise.sh slopes         # login node, minutes
bash run_vertexwise.sh design
bash run_vertexwise.sh glm            # < 1 h
bash run_vertexwise.sh fetch          # copies results/ to /project
```

Everything the compute nodes write goes under
`$SCRATCH/adni_trillium/derivatives/vertexwise/` (`/project` is read-only on
compute nodes): `freesurfer/`, `surf/`, `slopes/`, `glm/`, `logs/`.
The FreeSurfer license must be at `~/.licenses/freesurfer.lic`.

Then locally:

```
rsync -az trillium:/project/rrg-mchakrav-ab/ishaan/adni_trillium/vertexwise/results/ results/
/Users/ishu/research/mri-bsc/.venv/bin/python make_surface_figure.py \
    --results results --fs_subjects_dir fs_subjects \
    --out ../../mri-bsc/paper/aperture_neuro/fig8.png
```

`fs_subjects/fsaverage6/{surf,label}` is a copy of FreeSurfer's fsaverage6
(46 MB, not committed); rsync it from
`$EBROOTFREESURFER/subjects/fsaverage6` on Trillium.

`slurm/synthetic_glm_smoke.sh` runs the GLM stage on random data at cohort
size in a few minutes and is the place to start if anything in stage 6 is
changed.

## Result (2026-09-16)

416 subjects (133 converters). No cluster survives cluster-wise correction in
Model A (largest 274 mm2, cwp 0.33) or Model B (largest 329 mm2, cwp 0.25);
critical areas 850 and 869 mm2. mri_glmfit-sim agrees. With the raw
(unnormalized) field no vertex reaches the cluster-forming threshold because
scan-level scale jumps inflate the residual variance (`glm_raw` on scratch).
An earlier run that sampled the sparse band at projfrac 0.5 produced two
clusters and is an artefact of smoothing sparse hits (`*_v1_band_projfrac05`
on scratch); do not quote it.

## Outputs that go into the paper

- `results/modelA_cluster_summary.csv`, `results/modelB_cluster_summary.csv`:
  every observed cluster with area, peak t, peak aparc label, corrected p.
- `results/{lh,rh}.modelB.sig_clusters.mgh`: t inside surviving clusters.
- `results/fsA_{lh,rh}/perm.th20.abs.sig.cluster.summary`: mri_glmfit-sim
  cross-check for Model A.
- Section 3.11, the Methods subsection, the abstract sentence and the
  Discussion paragraph carry `[TBD: ...]` markers where the numbers go.

## Variants (2026-10-27)

Mallar asked for the simplest possible model, group plus age plus sex. That
model put one cluster through correction: right middle temporal, 1,168 mm2,
peak t -3.48, cwp 0.024 against a critical area of 827 mm2. The same region
with APOE4 and field strength in the design was 329 mm2 at cwp 0.25.

Dropping covariates cannot create signal, so one of two things is true: the
covariates were absorbing a real effect, or the simple model is reading a
confound as a group difference. Field strength is the candidate, because 3 T
covers 75 per cent of non-converters and 50 per cent of converters
(p < 0.001) and gBSC level tracks scanner strongly.

Covarying for field strength is not a clean test of that, for the same reason:
a covariate that correlates with the group partly adjusts the group away.
Fitting the two field strengths separately does test it, and the split is
unusually favourable here, 280 subjects with 67 events at 3 T against 136 with
66 events at 1.5 T. Two samples of near-equal power, so neither is the
footnote. If middle temporal appears in both, it is real; if it appears only
in the pooled sample, it is the scanner.

### What is fixed in advance

Every variant is weighted by slope precision, restricted to the seven
pre-specified AD-signature parcels, and tested one-sided in the hypothesised
direction (converters blurring faster, negative t). Those three choices apply
to all of them, so this is one configuration applied to several samples rather
than a search over configurations. The parcels are the same seven
`compute_regional_bsc.py` and `fs_sigmoid_features.py` use.

Together they are worth roughly a factor of six in critical area: the mask cuts
the search space, the one-sided tail halves the threshold, and the weights
recover the variance an unweighted two-stage fit throws away. None of it is
free of assumption, which is why the calibration below exists.

It passes, in the safe direction. On 100 null trials with the converter labels
reshuffled, 1,000 permutations each, the configuration rejected 2 per cent at a
nominal 5 per cent, 3 at 10 and 8 at 20. Conservative, not inflated, which
costs power rather than credibility. So if the real data gives a surviving
cluster under this configuration, the cluster-wise p is not the thing to
distrust; the sample it came from is.

### New pieces

- `compute_vertex_slopes.py --measure/--measure_root/--normalize demean/--qc`
  takes any fsaverage6 measure, not just gBSC, and writes
  `slope_precision.csv`: per subject, `sxx` from the scan times and `resid_sd`
  pooled over every cortex vertex, so the slope's precision carries tens of
  thousands of residuals rather than the n-2 a per-vertex estimate would.
- `make_design.py --field_subset {all,stable,3t,15t} --weights
  --max_weight_ratio` and `--y_measure`.
- `vertex_glm.py --mask_labels --tail {two,neg,pos} --weights --y_measure`.
- `slurm/04_recon_long.sh` runs FreeSurfer's longitudinal stream (`-base` then
  `-long`) on top of the cross-sectional runs; `02_sample_surface.sh` takes a
  fifth argument `long` to sample those surfaces into `surf_long/`.
- `slurm/05_glm_variant.sh` runs one named variant end to end,
  `slurm/06_slopes.sh` builds slopes for one measure, `run_variants.sh`
  submits the stages.
- `verify_slopes.py` checks the regenerated gradient slopes against the
  published ones, because the loop that writes them was touched.

### Why longitudinal recon is the one worth 10,000 core-hours

The analysis is a slope, a slope is a difference, and cross-sectional recon
gives every timepoint its own independent segmentation and surface placement.
The disagreement between those independent solutions goes into the slope
twice. The longitudinal stream initialises each timepoint from one
within-subject template, which removes most of that term. It is also the first
thing a reviewer will ask about. If the null survives it, the null is no
longer attributable to slope noise.

### Weights: bounded on purpose

Raw slope precision spans a factor of sixty in this cohort, and strict
inverse-variance weighting hands the fit to a handful of subjects. On
synthetic data with that spread it took the effective sample size from 40 to
17. With 133 events, losing half the effective sample to reclaim some
measurement variance is a bad trade, so the spread is compressed by raising
precision to the power that brings the 5th-to-95th-percentile ratio down to
`--max_weight_ratio` (default 4). That holds the effective sample size near 86
per cent of n, keeps the weights monotone in precision (Spearman 0.9998) and
saturates almost nobody. Clipping alone was tried first and was wrong: with a
spread this wide every subject lands on a bound and the weight degenerates
into a two-level factor.

### Local test harness

`tests/make_synthetic_world.py` builds a synthetic fsaverage6-shaped world: a
2,500-vertex triangulated grid per hemisphere with a 16-parcel annotation,
spatially smooth noise, unequal scan counts and spans, and a known slope
planted in right middle temporal. The full chain recovers it exactly, 144
vertices of the 144 planted, peak t -19.9, cwp 0.0005.
`tests/calibrate_variant.py` reshuffles the converter labels on a zero-effect
world and counts how often a cluster survives.

```
python3 tests/make_synthetic_world.py /tmp/w -0.03     # recovery
python3 tests/make_synthetic_world.py /tmp/n 0.0       # calibration
python3 tests/calibrate_variant.py /tmp/n . python3 100 1000
```

Three bugs were caught this way before anything reached the cluster. The weight
collapse above; a permutation pool that depends on `fork`, where the start
method was left to the default, which is spawn on macOS and becomes forkserver
on Linux in Python 3.14, and under either the workers start with an empty state
dict (it works on Trillium today and would have failed silently on the next
Python bump, so the context is now named explicitly); and a first calibration
run that was itself invalid because the synthetic noise had no spatial
correlation, which it announced by rejecting at the same rate for every alpha.

