# Sigmoid BSC (Olafson et al. 2021) on the Trillium ADNI cohort

Reimplementation of the boundary sharpness coefficient as actually published:
the growth-rate parameter of a sigmoid fit to a ten-point T1w intensity profile
crossing the gray/white boundary, per cortical vertex, in stereotaxic space.

This is a different quantity from the one in `mri-bsc/code/seg/atropos_bsc.py`,
which computes a gradient magnitude of the Atropos GM/WM probability field in
native voxel space. Both pipelines are kept so the two can be compared on the
same scans. Column prefixes keep them apart: `bsc_*`/`bscdir_*`/`bscmag_*` are
Atropos, `bscsig_*`/`bscsigfree_*`/`bscratio_*` are the sigmoid.

Reference implementation: <https://github.com/CoBrALab/BSC> (vendored under
`methods/`). Paper: Olafson et al., *Cerebral Cortex* 31:3338-3352, 2021.

## The measure

Ten samples per vertex. Six gray surfaces at 0, 6.25, 12.5, 18.75, 25 and 50%
of cortical thickness from the boundary toward the pial surface, and four white
surfaces mirrored through the boundary at the matching percentages. The 50%
white equivalent is dropped: it crosses into neighbouring cortex at thin gyral
crowns.

    profile(x) = a + exp(k) - exp(k) / (1 + exp(-c * (x - d)))

`c` is the BSC. `x` is in **percent of cortical thickness, not millimetres** —
the reference uses `c(-25, -18.75, -12.5, -6.25, 0, 6.25, 12.5, 18.75, 25, 50)`
and never converts to a distance, so `c` has units of inverse percent-thickness
and is not comparable across cohorts with different thickness distributions
without care.

Then: `log(c + 0.1)` → 20 mm FWHM smoothing → resample to the ICBM mesh →
residualise against mean curvature within site.

## Known limitation of the published estimator

The reference bounds the fit with `a >= min(y)`, which asserts the sigmoid
reaches its gray-side asymptote inside the sampled window. Sharp boundaries do;
blurred ones do not, and for those the bound binds and is wrong. On noiseless
simulated profiles:

| true `c` | reference bounds | `a` unbounded below |
|---------:|-----------------:|--------------------:|
| 0.01     | 0.0587           | 0.0121              |
| 0.02     | 0.0588           | 0.0200              |
| 0.05     | 0.0652           | 0.0500              |
| 0.10     | 0.1020           | 0.1000              |
| 0.50     | 0.5000           | 0.5000              |

BSC has a floor near 0.059. It cannot resolve degrees of blurring, which is the
direction pathology is hypothesised to move in both the ASD paper and this
project. `fit_sigmoid.py` therefore fits twice: `model_c` reproduces the
published estimator and is primary, `model_c_free` relaxes the bound and is a
sensitivity analysis. Reproduce with `--check` in `fit_sigmoid.py`.

## Pipeline

| Stage | Script | Where |
|---|---|---|
| 0 | verify CIVET 2.1.0 / MINC 1.9.16 / `$SCRATCH` quota | login node |
| 1 | `prepare_civet_inputs.py`, `slurm/run_civet.sh` | compute, ~26-52k core-hours |
| 2-4 | `run_surfaces.py`, `fit_sigmoid.py`, `postprocess_vertex.py` via `slurm/run_sigmoid_bsc.sh` | compute, ~100 core-hours |
| 5 | `gather_curvature.py`, `extract_sigmoid_features.py` | login node |
| 6 | `run_validity_checks.py` | login node |
| 7 | existing `analysis/` scripts, inputs swapped | login node |

Every stage skips work it has already done, so all of them are safe to
resubmit. CIVET will not finish in one job; resubmit `run_civet.sh` until it
reports nothing left.

    sbatch slurm/run_civet.sh       $PROJECT_DIR $SCRATCH/bsc_sigmoid $PROJECT_DIR/from_cluster/manifest_v3.csv
    sbatch slurm/run_sigmoid_bsc.sh $PROJECT_DIR $SCRATCH/bsc_sigmoid

    python3 gather_curvature.py --civet_root $SCRATCH/bsc_sigmoid/civet_out \
        --work_root $SCRATCH/bsc_sigmoid --manifest from_cluster/manifest_v3.csv \
        --out $SCRATCH/bsc_sigmoid/cache/curvature.npz

    python3 extract_sigmoid_features.py --work_root $SCRATCH/bsc_sigmoid \
        --manifest from_cluster/manifest_v3.csv \
        --curvature_npz $SCRATCH/bsc_sigmoid/cache/curvature.npz \
        --atlas_left ... --atlas_right ... --atlas_labels ... \
        --site_csv ... --out_dir from_cluster/

Then the existing survival machinery, with no code change:

    python3 analysis/build_spec_cohort.py \
        --bsc from_cluster/bsc_sigmoid_simple_features_sigmoid_v3.csv \
        --regional from_cluster/per_scan_regional_sigmoid_sigmoid_v3.csv \
        --out_dir analysis/results/sigmoid_v1
    python3 analysis/run_spec_models.py    --cohort analysis/results/sigmoid_v1/spec_cohort.csv --out_dir analysis/results/sigmoid_v1
    python3 analysis/run_spec_increment.py --cohort analysis/results/sigmoid_v1/spec_cohort.csv --out_dir analysis/results/sigmoid_v1

Compare `sigmoid_v1/incr_xgb_aft/increment_tests.json` against
`spec_v3/incr_xgb_aft/increment_tests.json` (baseline 0.790 → augmented 0.803,
Δ +0.013 [-0.018, +0.033], 0/33 slopes FDR-significant).

## Departures from the reference

- The reference R loop runs vertices `1:40963` over a table whose 40963rd
  column is the x vector, so its last "vertex" fits x against x. Only the 40962
  real vertices are fit here.
- The reference fits every vertex a second time on min-max rescaled intensities
  and discards the result. Not reproduced.
- The fit is `scipy.optimize.least_squares` with an analytic Jacobian rather
  than R `nls(algorithm="port")`, keeping the same box constraints, start
  values and iteration cap. Everything else in this project is Python and an
  R 3.4.0 / RMINC dependency is not worth carrying.
- `model_c_free` is added; see above.
- The ICBM model path is a flag rather than the reference's hardcoded
  CIC-local `/opt/quarantine` path.

## Things to watch

- **Entorhinal cortex is missing from the AD signature.** CIVET's surface atlas
  has no entorhinal parcel, so `bscsig_adsig` is six regions where the
  volumetric `bscdir_adsig` in `compute_regional_bsc.py` is seven. Compare
  within a measure, not across.
- **CIVET is cross-sectional.** Surface placement varies between a subject's
  timepoints and that variance lands directly in the longitudinal slopes the
  survival models consume — the problem `compute_regional_bsc.py` avoids by
  propagating labels rigidly. `run_validity_checks.py` estimates it as an
  ICC over short-interval rescan pairs. If that ICC is poor, the slope-based
  framing needs to give way to a baseline-BSC framing.
- **`--site_csv` matters.** Curvature residualisation is fit within site.
  Without it the manifest's coarse `dataset` column is used, which is a weaker
  control than the real scanner/site grouping in `spec_cohort.csv`.
