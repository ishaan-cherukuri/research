# mri-bsc (ADNI) — Copilot Context

This repository contains an end-to-end **MRI biomarker pipeline** for Alzheimer’s disease research.

Primary goal: compute **Boundary Sharpness Coefficient (BSC)** from **longitudinal structural MRI (T1w)**, store voxel-wise maps and summary metrics in **S3**, and later use compact derived features for classical ML.

---

## What BSC is (in this project)

BSC measures how sharp the **gray/white matter boundary** is. Lower sharpness can reflect tissue/contrast changes and is often studied as a biomarker.

This project computes **probability-based, directional BSC**:

- Build a GM probability map (Atropos segmentation)
- Define a boundary band where GM prob is near 0.5
- Compute a smoothed intensity gradient and project it onto the boundary normal (derived from ∇GMprob)

Outputs:

- **Voxel-wise maps** (3D NIfTI): `bsc_dir_map.nii.gz`, `bsc_mag_map.nii.gz`
- **Mask**: `boundary_band_mask.nii.gz`
- **Scalar summaries + QC**: `bsc_metrics.json`, `subject_metrics.csv`

---

## Data is stored on S3

This repo is designed around an S3-backed dataset layout.

### Raw ADNI export root

`s3://ishaan-research/data/raw/adni_5/`

Typical structure (conceptual):

```
raw/adni_5/
  <SUBJECT_ID>/
    <SESSION_DATE_FOLDER>/
      ... one or more *.nii or *.nii.gz
      ... one or more *.json sidecars
```

Session folders are commonly date/time strings such as:

- `YYYY-MM-DD` or `YYYY-MM-DD_HH_MM_SS(.fff)`

Within each session folder there should be:

- at least one NIfTI image (`.nii` or `.nii.gz`)
- at least one JSON (`.json`) for metadata

### Helper “clinical” table

`s3://ishaan-research/data/raw/final_df.csv`

This helper CSV provides visit and diagnosis information (and/or exam dates). It is used when building the scan manifest.

---

## Canonical manifest schema

The key manifest used throughout the pipeline must have **exactly** these columns:

| column | meaning |
|---|---|
| `subject` | ADNI subject ID (e.g., `002_S_0729`) |
| `visit_code` | visit label (e.g., `bl`, `sc`, `m12`, `m24`, `m36`) |
| `acq_date` | scan acquisition date (YYYY-MM-DD) |
| `path` | S3 path to the chosen T1 NIfTI for this visit |
| `diagnosis` | label derived from `final_df.csv` (CN/MCI/AD etc.) |

This manifest is stored in S3, e.g.:

`s3://ishaan-research/data/manifests/adni_manifest.csv`

### Stable per-scan image_id

A stable identifier is derived from the manifest row:

```
image_id := <subject>_<visit_code>_<acq_date>
```

This `image_id` is used as the folder name for derivatives:

- Preprocessing derivatives: `.../preprocess/.../<image_id>/...`
- BSC derivatives: `.../bsc/.../<image_id>/...`

---

## Manifest building

The ingest script builds a clean manifest by:

1. Listing S3 sessions for each subject under `raw/adni_5/<subject>/...`
2. Choosing sessions that correspond to expected longitudinal timepoints
3. Picking a “best” T1 series `.nii(.gz)` file inside each chosen session
4. Merging diagnosis/visit metadata from the helper CSV `final_df.csv`

Key rules the project uses:

- Each subject must have **≥ 4 scans** with different timepoints (longitudinal).
- Session folder dates should **align by year** with helper EXAMDATE.
- Matching can be “relaxed”: as long as scans are reasonably separated in time, it’s acceptable.

Output:

- `adni_manifest.csv` (one row per scan)

---

## Preprocessing step (simple)

The preprocessing stage produces standardized inputs that the BSC pipeline consumes.

Preprocess outputs per scan (`<image_id>`):

- `t1w_preproc.nii.gz`
- `gm_prob.nii.gz` *(placeholder in simple version; true GM/WM produced later in Atropos engine)*
- `wm_prob.nii.gz`
- `brain_mask.nii.gz`
- `preprocess_metadata.json`

S3 output root example:

`s3://ishaan-research/data/derivatives/preprocess/adni_5/<image_id>/...`

The initial `simple_preproc.py` performs:

- basic intensity normalization
- dummy masks/probabilities (placeholders)

Later improvements may include:

- N4 bias correction
- real skull stripping (Otsu / SynthStrip / HD-BET)

---

## BSC computation stage

There are two engines:

### 1 Atropos engine (main)

`code.seg.atropos_bsc` performs:

- Load preprocessed T1
- Bias correction (N4)
- Skull strip / mask generation
- Resample to 1mm isotropic
- Atropos segmentation (3-class KMeans)
- Identify GM/WM probability volumes
- Compute BSC

Outputs uploaded to S3 per scan:

```
<out_root>/<image_id>/
  t1w_preproc.nii.gz
  brain_mask.nii.gz
  gm_prob.nii.gz
  wm_prob.nii.gz
  bsc_dir_map.nii.gz
  bsc_mag_map.nii.gz
  boundary_band_mask.nii.gz
  bsc_metrics.json
  subject_metrics.csv
```

### 2) FreeSurfer engine (optional)

`code.seg.freesurfer_bsc` computes BSC using FreeSurfer outputs if available.

---

## Batch pipeline runner

Batch runner module:

- `code/pipeline/run_batch.py`

Inputs:

- `--manifest` points to the manifest CSV (local path or S3)
- `--engine` one of `atropos` or `freesurfer`
- `--out_root` S3 prefix for outputs
- `--preproc_root` S3 prefix for preprocessing derivatives
- optional `--skip`, `--limit`

Core behavior:

- Iterates over rows in the manifest (each row is one scan/visit)
- Creates `image_id`
- Downloads required inputs from S3 to local temp
- Runs the chosen BSC engine
- Uploads outputs back to S3

### IMPORTANT: skip already processed scans

The batch runner should skip a scan if output already exists in S3.

A scan is considered “done” if the output folder contains:

- `bsc_dir_map.nii.gz` (primary completion flag)

Safer completion check (recommended):

- `bsc_dir_map.nii.gz` AND `bsc_metrics.json`

This prevents skipping partially written outputs.

---

## Temporary storage requirements (macOS + ANTs)

This project must avoid uncontrolled temp growth in macOS:

`/private/var/folders/.../T`

ANTs/ITK and gzip operations can create temp files there.

### Project convention

Use a project-local temp workspace:

- `data/splits/`
- `data/splits/tmp/`

### Enforce temp directory

Before importing/using ANTs, set environment variables:

- `TMPDIR`, `TMP`, `TEMP`

and set Python’s `tempfile.tempdir`.

This ensures all temp files go into `data/splits/tmp` instead of the OS default.

---

## S3 helper utilities

Main S3 utilities live in:

- `code/io/s3.py`

Common helpers:

- `parse_s3_uri(uri)` → `(bucket, key)`
- `download_to_temp(s3_uri)` → downloads to local NamedTemporaryFile
- `upload_file(local_path, s3_uri)`
- `ensure_s3_prefix(prefix)`
- `clear_s3_prefix(prefix)` *(dangerous)*

Note: `download_to_temp()` currently uses the OS temp directory unless `TMPDIR` is set.

---

## Skull stripping / masking options

Masking is critical because BSC should be computed on brain tissue only.

This repo supports multiple mask approaches:

- `code/skullstrip/ants_mask.py` (simple mask, Otsu-based, etc.)
- `code/skullstrip/run_synthstrip.py` (higher quality)
- `code/skullstrip/run_hdbet.py` (high quality, heavier)

Project preference for fast + stable runs:

- Otsu + morphological cleanup (GetLargestComponent, FillHoles)

---

## Recommended run commands

### Build manifest

```
python3 -m code.ingest.build_manifest_adni \
  --final_df s3://ishaan-research/data/raw/final_df.csv \
  --s3_raw_root s3://ishaan-research/data/raw/adni_5 \
  --out_csv s3://ishaan-research/data/manifests/adni_manifest.csv
```

### Preprocess (simple placeholder)

```
python3 -m code.preprocess.simple_preproc \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_root s3://ishaan-research/data/derivatives/preprocess/adni_5 \
  --temp_root data/splits
```

### Run Atropos BSC batch

```
python3 -m code.pipeline.run_batch \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --engine atropos \
  --out_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --preproc_root s3://ishaan-research/data/derivatives/preprocess/adni_5 \
  --skip 0
```

---

## How voxel-wise maps are used for ML (don’t explode input size)

Voxel maps are high-dimensional; this project does not feed full 3D maps directly into classical ML.

Instead, derive compact features per scan or subject:

- boundary-only distribution stats (median/mean/std/quantiles)
- ROI pooled means using an atlas (50–200 features)
- longitudinal slopes over ≥4 timepoints (very compact + powerful)

This keeps computation simple while still using voxel-wise maps as high-quality intermediate representations.

---

## Common failure modes + fixes

### “wrote only 0 bytes” / corrupted `.nii.gz`

Cause: local disk/temp pressure or interrupted writes.

Fix:

- use controlled temp folder (`data/splits/tmp`)
- ensure enough local disk space
- skip already-done scans to resume safely

### Partial outputs in S3

If a run crashes mid-upload, a folder may exist but be incomplete.

Fix:

- treat scan as complete only if both `bsc_dir_map.nii.gz` and `bsc_metrics.json` exist

### BSC maps non-zero across the brain

Symptom: `bsc_dir_map.nii.gz` / `bsc_mag_map.nii.gz` look non-zero over large brain regions.

Expected: voxel-wise BSC maps should be **exactly zero outside the GM/WM boundary** (boundary band / interface).

Fix: run the S3 post-processing masker to hard-zero values outside a boundary mask:

```
python -m code.pipeline.postprocess_mask_bsc_s3 \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_root  s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --temp_root data/splits \
  --mask_mode interface
```

Notes:

- This step does **not** use an atlas; it uses segmentation-derived boundary masks.
- `--mask_mode band` uses `boundary_band_mask.nii.gz` (Atropos output).
- `--mask_mode interface` builds a tighter 1-voxel-ish interface mask from `gm_prob.nii.gz` + `wm_prob.nii.gz` (+ `brain_mask.nii.gz`).
- All temp I/O is forced into `data/splits/tmp` (macOS-safe).
- Optional FreeSurfer constraint is available if you have `ribbon.mgz` + `wm.mgz` in S3 (see script help).

---

## Design principles

- **S3-first storage**: raw + derivatives live in S3
- **deterministic IDs**: `image_id = subject_visit_code_acq_date`
- **resume-friendly**: skip scans already processed
- **local compute**: download inputs locally, compute locally, upload results
- **lightweight ML**: compress voxel maps into compact features


### Skip already-processed scans (required behavior)

The batch runner should **skip** any scan that is already present in S3 to avoid duplicates and wasted compute.

Definition of “done” (recommended):

- Consider a scan complete if this file exists:
  - `<out_root>/<image_id>/bsc_dir_map.nii.gz`

Safer definition:

- Require BOTH:
  - `<out_root>/<image_id>/bsc_dir_map.nii.gz`
  - `<out_root>/<image_id>/bsc_metrics.json`

This prevents skipping partially-uploaded outputs after a crash.

---

## Local temp folder requirements (macOS + ANTs)

This project uses temporary local disk storage to avoid slow / fragile S3-native operations.

### Desired behavior

- All scratch space should live under:

```
./data/splits/
```

Example:

- `data/splits/tmp/` for global temp files
- `data/splits/tmp/<image_id>/` for per-scan work folders

### Why

On macOS, Python + gzip + native libraries may otherwise create large hidden temp files in:

- `/private/var/folders/.../T/`

This causes apparent “disk explosion”.

### How to enforce project temp usage

Set environment variables so temp files go to `data/splits/tmp`:

```bash
export TMPDIR="$(pwd)/data/splits/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
```

In Python entrypoints, set `tempfile.tempdir` early (before importing ANTs):

```python
import os, tempfile
from pathlib import Path

tmp = Path("data/splits/tmp").resolve()
tmp.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(tmp)
os.environ["TMP"] = str(tmp)
os.environ["TEMP"] = str(tmp)
tempfile.tempdir = str(tmp)
```

---

## S3 IO helpers

`code/io/s3.py` provides small helpers:

- `parse_s3_uri(uri)` → (bucket, key)
- `download_to_temp(s3_uri)` → downloads an object to a local temp file
- `upload_file(local_path, s3_uri)` → uploads a local file to S3
- `ensure_s3_prefix(prefix)` → creates prefix marker
- `clear_s3_prefix(prefix)` → deletes everything under a prefix (dangerous)

Notes:

- `download_to_temp()` uses the OS temp directory by default; the project prefers a **controlled temp root** under `data/splits/`.

---

## Running the pipeline (example commands)

### 1) Build a manifest

```bash
python3 -m code.ingest.build_manifest_adni \
  --final_df s3://ishaan-research/data/raw/final_df.csv \
  --s3_root s3://ishaan-research/data/raw/adni_5 \
  --out s3://ishaan-research/data/manifests/adni_manifest.csv
```

### 2) Run preprocessing

```bash
python3 -m code.preprocess.simple_preproc \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_root s3://ishaan-research/data/derivatives/preprocess/adni_5 \
  --temp_root data/splits
```

### 3) Compute voxel-wise BSC maps (Atropos)

```bash
python3 -m code.pipeline.run_batch \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --engine atropos \
  --out_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --preproc_root s3://ishaan-research/data/derivatives/preprocess/adni_5 \
  --temp_root data/splits \
  --skip 0
```

---

## Longitudinal intent (4 scans per subject)

This project intentionally builds longitudinal sequences.

Typical target timepoints:

- `bl` (baseline)
- `m12`
- `m24`
- `m36`

Some subjects may have additional timepoints; the manifest builder chooses 4 well-spaced visits.

The downstream ML strategy is expected to use **compact features per scan**, then model:

- cross-sectional diagnosis
- or longitudinal slopes (change in BSC over time)

---

## Machine learning plan (compute-safe)

Voxel-wise BSC maps are too large to feed directly to standard ML.

Recommended approach:

1) For each scan, extract boundary-band values and compute summary stats:
   - median/mean/std
   - percentiles (p10/p25/p50/p75/p90)
   - Nboundary
2) Optionally compute ROI means via atlas/parcellation
3) Train small models:
   - logistic regression
   - XGBoost / LightGBM

Later (optional): downsample to 16³ or 32³ grid features.

---

## Key files & modules

- `code/ingest/build_manifest_adni.py`
  - manifest building from raw S3 + helper clinical table
- `code/preprocess/simple_preproc.py`
  - lightweight preprocessing + uploads to S3
- `code/pipeline/run_batch.py`
  - main batch runner for BSC
- `code/seg/atropos_bsc.py`
  - Atropos segmentation + voxel-wise BSC computation
- `code/bsc/bsc_core.py`
  - image loading + BSC math core
- `code/io/s3.py`
  - S3 utilities
- `code/pipeline/postprocess_mask_bsc_s3.py`
  - post-process voxel-wise BSC maps in S3 to be zero outside the boundary

---

## Known failure modes & expectations

- **Disk/temp blowups on macOS** due to `/private/var/folders/.../T/`
  - Fix: force TMPDIR to `data/splits/tmp`

- **Truncated .nii.gz** producing nibabel `EOFError`
  - Usually means a partial write (disk full or interrupted write)
  - Fix: ensure enough disk space, use controlled temp, prefer `.nii` intermediates

- **Pipeline restarts**
  - Batch runner must skip scans already written to S3
  - Should be restart-safe (idempotent)

- **S3 latency**
  - Minimize downloading/uploading inside inner loops
  - Prefer: download once → compute locally → upload final outputs

---

## Conventions

- All S3 paths are `s3://bucket/key`.
- Outputs are organized by `image_id` folder.
- Prefer “one scan per folder” for reproducible outputs.
- Avoid destructive clears of output prefixes unless explicitly requested.

