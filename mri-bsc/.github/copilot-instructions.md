# MRI-BSC: Copilot Instructions

## Project Overview
**MRI-BSC** is a modular pipeline to compute **Boundary Sharpness Coefficient (BSC)** from T1-weighted MRI scans. BSC quantifies gray matter to white matter intensity transition sharpness (units: SD/mm). The pipeline is engine-agnostic with pluggable implementations for preprocessing, skull-stripping, and segmentation.

### Core Workflow
1. **Ingest** (`code/ingest/`): Build manifests from ADNI/OASIS/NACC source data
2. **Preprocess** (`code/preprocess/`): Bias correction, orientation, resampling (MONAI or ANTs)
3. **Skull-strip** (`code/skullstrip/`): Remove non-brain tissue (SynthStrip/HD-BET/ANTs)
4. **Segment** (`code/seg/`): Produce GM/WM probability maps (SynthSeg/FreeSurfer/Atropos)
5. **Compute BSC** (`code/bsc/`): Calculate boundary sharpness via gradient analysis
6. **Evaluate** (`code/eval/`): Site robustness, clinical associations

## Architecture & Key Patterns

### Engine Abstraction Pattern
Multiple interchangeable implementations exist for preprocessing, skull-stripping, and segmentation. Each has its own module:
- **Preprocessing**: `code/preprocess/monai_preproc.py`, `code/preprocess/simple_preproc.py` (strategy pattern)
- **Skull-strip**: `code/skullstrip/{ants_mask.py, run_hdbet.py, run_synthstrip.py}`
- **Segmentation**: `code/seg/{synthseg_bsc.py, freesurfer_bsc.py, atropos_bsc.py}`
- **Selection**: Configure active engine in `configs/pipeline.yaml` (e.g., `seg_engine: synthseg`)
- **Runtime loading**: `code/pipeline/run_batch.py` uses `importlib.import_module()` to dynamically load the selected engine module

**When modifying engines**: Ensure all implementations accept the same input/output signatures. Document required parameters in the module docstring.

### Data Flow & File Conventions
1. **Manifests** (CSV): Source lists with paths. Required columns vary by source (ADNI has `subject_id`, `image_id`, `diagnosis_group`; see `code/ingest/build_manifest_*.py` for specifics).
2. **Preprocessed outputs**: Standard intermediate filenames (enforced by batch runner):
   - `t1w_preproc.nii.gz` (reoriented, resampled T1)
   - `gm_prob.nii.gz`, `wm_prob.nii.gz` (probability maps)
   - `brain_mask.nii.gz` (binary mask)
3. **S3 Integration** (`code/io/s3.py`): All paths use `s3://bucket/key` URIs; local tools download via `download_to_temp()`, upload results via `upload_file()`. Temp files are cleaned up by OS.

### BSC Computation Core
Located in `code/bsc/bsc_core.py`. Key functions:
- **`wm_zscore(t1_arr, wm_prob, brain_mask, thr=0.5)`**: Z-score normalize T1 intensities using WM (or brain) statistics.
- **`directional_bsc(t1_norm, gm_prob, brain_mask, spacing, band_eps=0.05, sigma_mm=1.0)`**: Main calculation:
  - Defines GM/WM boundary band as voxels where `P_gm ≈ 0.5 ± band_eps`
  - Computes intensity gradient `∇I` (after Gaussian smoothing in physical units)
  - Computes probability gradient `∇P_gm` and normalizes to unit normal `n̂`
  - Calculates directional derivative `∂I/∂n` along boundary normal
  - Returns median `|∂I/∂n|` (directional BSC) and magnitude as dict
- **Spacing**: Always in mm (voxel sizes extracted from NIfTI affine), converted to voxel sigma via `sigma_mm / voxel_size[axis]`

**When modifying BSC logic**: Boundary detection is probability-based (not atlas-based). Changes to `band_eps` or smoothing `sigma_mm` significantly affect results; document in config.

### Manifest Building Patterns
Each cohort (ADNI, OASIS, NACC) has its own builder (`build_manifest_adni.py`, etc.):
- Reads source CSV with cohort-specific column names
- Validates required columns (subject_id, image_id, diagnosis_group, etc.)
- Lists S3 raw data paths
- Outputs standardized manifest with normalized columns
- Date parsing is permissive (`try/except` wrapper)

**When adding new cohorts**: Create `code/ingest/build_manifest_<cohort>.py`, follow the same structure, ensure S3 path consistency.

## Configuration & External Tools

### Config Files
- **`configs/pipeline.yaml`**: Active engine selection and BSC parameters (band_eps, sigma_mm)
- **`configs/paths.yaml`**: Local/S3 folder paths (data root, derivatives, outputs)
- **`configs/synthseg_label_map.json`**: Maps SynthSeg channel IDs to tissue labels (e.g., `{"0": "WM", "1": "GM"}`)

### External Dependencies
- **FreeSurfer** (`recon-all`, `mri_vol2surf`): Set `$SUBJECTS_DIR` env var if using FreeSurfer engine
- **SynthSeg**: Can be called via CLI or Docker; output is 4D posterior probability stack
- **dcm2niix**: For DICOM→NIfTI conversion (optional, for NACC ingestion)
- **ANTsPy**: Platform-dependent (Linux/macOS only; excluded on Windows in `pyproject.toml`)

### Installation
- **UV** (recommended): `uv venv && uv pip install -e .`
- **Pip**: `pip install -e .` (sets up editable install with `code/` as importable package)

## Development & Testing Patterns

### Batch Processing
`code/pipeline/run_batch.py::run_batch()` is the main entry point:
- Takes manifest CSV, engine name, output root, and optional kwargs (eps, sigma_mm, subjects_dir, etc.)
- Streams data via `pandas.iterrows()`; large datasets use `skip=` and `limit=` for checkpointing
- Clears output prefix before starting (idempotent design)
- Errors in individual subjects are logged but don't halt processing

**When debugging**: Pass `limit=5` to test on small subset; check S3 paths in error logs.

### Notebooks
Use `/notebooks/` for exploratory analysis and prototyping. Examples:
- `ANTsPy/`: ANTs mask generation and transformation experiments
- `Misc_Test/`: Ad-hoc algorithm validation
- CSV data is in `csv_misc/` for quick iteration

### QC & Evaluation
- **`code/eval/compute_bsc_summary.py`**: Aggregates BSC results across cohorts
- **`code/eval/site_robustness.py`**: Compares BSC distributions (OASIS-3 vs NACC)
- **`code/eval/assoc_status.py`**: Clinical correlations (if diagnosis labels provided)
- **`code/utils/qc.py`**: Visualization helpers (generate PNG quicklooks)

## Dependencies & Integration Points

### Key Libraries
- **NiBabel**: NIfTI I/O, affine transforms, voxel size extraction
- **NumPy/SciPy**: Core math (gradients, filtering, normalization)
- **MONAI**: Preprocessing transforms (N4 bias correction, resampling)
- **Pandas**: Manifest/result CSV handling
- **Boto3/S3fs**: Cloud storage (all paths prefixed with `s3://`)
- **Scikit-image, Scikit-learn**: Image metrics, optional ML (calibration)

### Platform Notes
- **Linux/macOS**: Full feature set (ANTsPy, HD-BET available)
- **Windows**: Missing ANTsPy; FreeSurfer path handling may differ
- **S3 credentials**: Assumed via AWS CLI or environment (`AWS_ACCESS_KEY_ID`, etc.)

## Conventions & Gotchas

1. **Z-score normalization**: Always normalize T1 intensities before gradient computation. WM is preferred reference tissue.
2. **Affine awareness**: All gradients computed in physical (mm) space; voxel spacing extracted from NIfTI headers.
3. **Probability thresholds**: GM/WM boundary at `P_gm = 0.5` is a soft criterion; `band_eps` controls tolerance.
4. **Module imports**: Use relative imports within `code/` subpackages (e.g., `from code.bsc.bsc_core import ...`); notebooks and scripts use absolute.
5. **Temp file cleanup**: S3 download creates tempfiles; OS cleans up on exit. Don't rely on explicit deletion.
6. **CSV column names**: Always strip whitespace when reading user-provided CSVs (`df.columns = [c.strip() for c in df.columns]`).

---

**Last Updated**: December 31, 2025
