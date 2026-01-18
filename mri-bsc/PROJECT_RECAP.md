# MRI-BSC (ADNI) — Project Recap (Jan 2026)

This repo is an end-to-end pipeline to compute **Boundary Sharpness Coefficient (BSC)** from longitudinal T1 MRI (ADNI-style data), store voxel-wise maps + summary metrics in **S3**, and then use compact derived features for downstream ML.

---

## 1) What BSC is (in this repo)

**Boundary Sharpness Coefficient (BSC)** measures how “sharp” the gray/white matter boundary is.

This project computes **probability-based voxel-wise BSC** using GM/WM probability maps:

- Build GM/WM probability maps (mainly via **Atropos** segmentation).
- Define a **boundary region** (either a band where GM probability is near 0.5, or a tighter “interface” mask derived from GM/WM adjacency).
- Compute intensity-gradient-based BSC, often reported as:
  - **Directional map**: `bsc_dir_map.nii.gz`
  - **Magnitude map**: `bsc_mag_map.nii.gz`

Expected behavior: **BSC maps should be zero outside the boundary mask**.

---

## 2) Data + storage layout (S3-first)

This repo is designed around S3.

### Raw ADNI export root

- `s3://ishaan-research/data/raw/adni_5/<SUBJECT>/<SESSION_FOLDER>/<FILE>.nii.gz`

### Canonical manifest schema

Manifest CSV used throughout the pipeline has exactly these columns:

- `subject` (e.g., `002_S_0729`)
- `visit_code` (e.g., `bl`, `m12`, `m24`, `m36`)
- `acq_date` (YYYY-MM-DD)
- `path` (S3 path to the chosen T1 NIfTI)
- `diagnosis` (label from clinical CSVs)

### Stable per-scan image_id

- `image_id := <subject>_<visit_code>_<acq_date>`

This `image_id` is the folder name used for derivatives.

### Derivatives layout

- Preprocess derivatives (per scan):
  - `s3://ishaan-research/data/derivatives/preprocess/adni_5/<image_id>/...`

- BSC derivatives (per scan):
  - `s3://ishaan-research/data/derivatives/bsc/adni/atropos/<image_id>/...`

Each BSC folder typically contains:

- `t1w_preproc.nii.gz`
- `brain_mask.nii.gz`
- `gm_prob.nii.gz`
- `wm_prob.nii.gz`
- `bsc_dir_map.nii.gz`
- `bsc_mag_map.nii.gz`
- `boundary_band_mask.nii.gz`
- `bsc_metrics.json`
- `subject_metrics.csv`

---

## 3) Main pipeline stages

### Stage A — Manifest building

Goal: pick one “best” T1 per subject/visit and attach diagnosis/visit metadata.

Key module(s):
- `code/ingest/build_manifest_adni.py` (and related ingest helpers)

Output: a manifest CSV used by downstream stages.

### Stage B — Preprocessing

Goal: produce standardized inputs consumed by the BSC engine.

Typical outputs per scan:
- `t1w_preproc.nii.gz`
- `brain_mask.nii.gz`
- `gm_prob.nii.gz`, `wm_prob.nii.gz` (placeholders in simple version; real ones come from Atropos engine)

Key module(s):
- `code/preprocess/simple_preproc.py` (lightweight placeholder version)

### Stage C — BSC computation (Atropos engine)

Goal: compute voxel-wise BSC maps and metrics.

Key module(s):
- `code/pipeline/run_batch.py` (batch runner)
- `code/seg/atropos_bsc.py` (Atropos segmentation + BSC)
- `code/bsc/bsc_core.py` (BSC math)

Important behavior:
- The batch runner should **skip** scans already processed.
- “Done” is best defined by both:
  - `bsc_dir_map.nii.gz` and `bsc_metrics.json`

### Optional engine — FreeSurfer

There is optional FreeSurfer-related code (useful if FS outputs already exist), but the main production path in this repo is the Atropos engine.

---

## 4) Known issue: BSC maps non-zero across the brain

Symptom:
- `bsc_dir_map.nii.gz` / `bsc_mag_map.nii.gz` show non-zero values beyond the GM/WM boundary.

Expected:
- Maps should be **exactly zero outside a boundary mask**.

Fix implemented:
- Run an S3-wide postprocessing step that masks BSC maps and overwrites results back into S3.

Key module:
- `code/pipeline/postprocess_mask_bsc_s3.py`

What it does:
- Downloads per-scan outputs into local scratch under `data/splits/tmp/<image_id>/`
- Builds a mask (recommended: `--mask_mode interface`)
- Applies mask to `bsc_dir_map.nii.gz` and `bsc_mag_map.nii.gz`
- Optionally rewrites:
  - `boundary_band_mask.nii.gz` (`--write_mask`)
  - `bsc_metrics.json` and `subject_metrics.csv` (`--write_metrics`)
- Uploads the corrected files back to S3
- Cleans local temp per scan (unless `--keep_local`)

---

## 5) macOS temp / disk safety

On macOS, native libraries (ANTs/ITK/gzip) can dump large temporary files into:

- `/private/var/folders/.../T/`

Project convention:
- Force temp into `data/splits/tmp`.

Shell env variables used:
- `TMPDIR`, `TMP`, `TEMP`

---

## 6) Batch runner entrypoint used in practice

A practical shell entrypoint exists:

- `scripts/MAIN.sh`

It sets temp directories and runs pipeline steps (including postprocessing).

---

## 7) New index + labels for ML

You asked for a simple “index CSV” that lists the BSC file paths and prepares labels from the manifest.

What was created:

- Script: `code/index/build_bsc_file_index.py`
- Output CSV: `code/index/bsc_file_index.csv`

Index output is scan-level (one row per scan/visit) and includes:
- scan identifiers (`subject`, `visit_code`, `acq_date`, `image_id`)
- `diagnosis`
- subject label
- S3 paths to relevant BSC + aux files

Current label rule (as requested):
- `label = 1` if the subject’s `diagnosis` is `3` at **any** timepoint in the manifest, else `0`.

Note:
- This is “ever AD” labeling, not strictly “MCI at baseline → AD within 4 years.”
- If you want strict conversion labeling (baseline diagnosis==2 and AD within 48 months), that can be added as a second label column.

---

## 8) Recommended ML approach (4 timepoints per subject)

Given ~400 subjects and only 4 scans each:

Best baseline:
- Extract compact per-scan features from boundary-only BSC (global stats or coarse spatial bins)
- Convert 4 timepoints into longitudinal features (baseline, last, change, slope)
- Train XGBoost / LightGBM or elastic-net logistic regression

Why:
- Sequence models (RNN/Transformer) tend to overfit at this dataset size unless inputs are already compact and robust.

---

## 9) Key files to know

- Manifest ingest:
  - `code/ingest/build_manifest_adni.py`

- Batch runner:
  - `code/pipeline/run_batch.py`

- Atropos BSC:
  - `code/seg/atropos_bsc.py`
  - `code/bsc/bsc_core.py`

- S3 helpers:
  - `code/io/s3.py`
  - `code/utils/io_any.py`

- Postprocess mask (boundary-only enforcement):
  - `code/pipeline/postprocess_mask_bsc_s3.py`

- ML index:
  - `code/index/build_bsc_file_index.py`
  - `code/index/bsc_file_index.csv`

---

## 10) Typical commands

### Build the BSC file index from your manifest

```bash
python3 -m code.index.build_bsc_file_index \
  --manifest /Users/ishu/Downloads/adni_manifest.csv \
  --bsc_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --out_csv code/index/bsc_file_index.csv \
  --min_visits_per_subject 4
```

### Postprocess: force BSC maps to be boundary-only (S3 overwrite)

```bash
python3 -m code.pipeline.postprocess_mask_bsc_s3 \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_root  s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --temp_root data/splits \
  --mask_mode interface \
  --write_mask \
  --write_metrics
```

---

## 11) Next logical steps

1. Decide final labeling:
   - “ever AD” vs “MCI baseline → AD within 4 years”
2. Decide feature strategy:
   - global stats only, or coarse spatial bins, or atlas/FreeSurfer ROIs restricted to boundary mask
3. Train baseline model with clean subject-level CV split

