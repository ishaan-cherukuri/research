# MRI-BSC (ADNI) — Project Recap (Jan 2026) RUN WITH CMD + SHIFT + V

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

## 8) Feature extraction and ML progress (Jan 2026)

### Feature extraction completed
- **Script:** `code/features/extract_simple_features.py`
- **Output:** `data/index/bsc_simple_features.csv` (1,718 scans × 187 features)
- **Features include:**
  - Global statistics (mean, std, median, percentiles p5-p95, skew, kurtosis, IQR, range)
  - GM/WM-weighted means (6 features)
  - Spatial binning (4×4×4 = 64 bins × 2 maps = 128 features)
  - Texture metrics (CV, entropy)
  - Sign-specific stats (pos/neg fractions and means for directional BSC)
- **Filtering:** Non-zero BSC values only to handle sparse data (96.5% of voxels are zero outside boundary)
- **Error handling:** Try-except per scan to skip corrupted NIfTI files

### ML experiments conducted

#### 1. Survival analysis approach (FAILED)
**Goal:** Predict time-to-AD-conversion from baseline BSC features

**Datasets created:**
- `data/ml/survival/time_to_conversion.csv` (424 subjects, 91 events, 333 censored)
- `data/ml/survival/landmark_1yr.csv` (409 subjects, 79 events)
- `data/ml/survival/landmark_2yr.csv` (353 subjects, 40 events)
- `data/ml/survival/landmark_3yr.csv` (277 subjects, 15 events)
- `data/ml/survival/joint_longitudinal_survival.csv` (all timepoints)

**Models trained:**
- Parametric AFT (Weibull) via lifelines
- XGBoost survival (AFT objective) - created but not tested due to library issues

**Results:** **All failed spectacularly**
- Baseline (top 20): C-index = 0.2386 (worse than random 0.5)
- Baseline (all 187): C-index = 0.1362 (even worse!)
- Landmark 1yr: C-index = 0.2227
- Landmark 2yr: C-index = 0.2074

**Diagnosis:**
- Predictions NOT inverted (converters correctly predicted shorter times than non-converters)
- BUT weak correlation (0.132) between actual and predicted times
- Model over-optimistic (predicts converters at 11 years, actual 1.9 years)
- Some predictions unrealistic (100+ years for non-converters)

**Key insight:** BSC features measure **current tissue state**, not **future conversion risk**
- Baseline BSC doesn't predict WHEN someone will convert
- Conversion timing depends on many factors BSC can't capture (genetics, cognitive reserve, other pathologies)

#### 2. Classification approach (PROPOSED - NOT YET RUN)
**Goal:** Predict diagnosis category (CN/MCI/AD) from BSC features

**Rationale:**
- BSC should show categorical differences: CN > MCI > AD (sharper → blurrier boundaries)
- Measuring **what is** (current disease state) not **what will be** (future conversion)
- Simpler problem aligned with BSC's biological meaning

**Script created:** `code/ml/train_diagnosis_classifier.py`
- Supports 3-class (CN/MCI/AD), 2-class (CN+MCI vs AD), binary (CN vs AD only)
- Logistic regression + Random Forest
- 5-fold cross-validation
- Balanced accuracy for class imbalance
- Feature importance from RF

**Status:** Code ready, not yet executed

### Critical finding: What BSC should be used for

**❌ WEAK for:**
- Baseline-only survival prediction (failed)
- Predicting conversion TIMING from cross-section
- Standalone predictor without clinical context

**✅ STRONG for (proposed):**

1. **Cross-sectional staging (CN/MCI/AD classification)** ← Next step
   - BSC should show: CN (sharp) > MCI (moderate blur) > AD (severe blur)
   
2. **Longitudinal BSC slopes** ← **KILLER FEATURE (NOT YET TRIED!)**
   - Compute **rate of BSC decline** per subject (4 timepoints available)
   - Rate of boundary degradation might predict conversion
   - Literature shows: change rate >> absolute baseline values
   - Example: `-0.067/year` (rapid decline) vs `-0.017/year` (stable)
   - **This aligns with successful volumetric predictors (hippocampal atrophy rates)**
   
3. **Regional BSC analysis** (hippocampal/entorhinal boundaries)
   - Focus on anatomically-relevant regions
   - Atlas-based regional BSC
   - Align with known AD pathology
   
4. **Microstructural complement to volumetrics**
   - Volumes measure size, BSC measures tissue quality
   - Combine: "small hippocampus + blurry boundaries" = high risk
   - BSC might catch early changes before atrophy
   
5. **Subtyping/stratification marker**
   - AD heterogeneity matters (multiple papers show this)
   - BSC spatial patterns might identify subtypes
   - Stratify first, then predict within strata

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

