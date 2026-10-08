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
