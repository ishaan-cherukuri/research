# ADNI Feature Extraction with Segmentation

Enhanced feature extraction pipeline for ADNI survival analysis with brain segmentation support.

## New Features

### 1. Diagnosis Labels ✅
- **Baseline diagnosis** tracking (CN/MCI/AD)
- **MCI→AD conversion** detection from metadata
- **Event time** calculation in years
- **Proper censoring** for survival analysis

### 2. Segmentation Features ✅
Two methods available:

#### Simple Segmentation (Fast, ~5 seconds/scan)
- Intensity-based tissue classification
- GM, WM, CSF volumes
- Ventricle estimates
- Brain parenchymal fraction (BPF)
- Total intracranial volume (TIV)

#### SynthSeg (Accurate, ~1-2 minutes/scan)
Install with: `uv add freesurfer`

Provides detailed volumes for:
- Hippocampus (L/R)
- Amygdala (L/R)
- Ventricles (lateral, 3rd, 4th)
- Cortical regions
- Normalized volumes (critical for comparison)

### 3. Longitudinal Tracking ✅
Computes slopes and changes over time for:
- Brain volume
- Brain intensity metrics
- SNR (Signal-to-Noise Ratio)
- Brain-background ratio

Features per biomarker:
- `_last`: Most recent value
- `_delta`: Absolute change from baseline
- `_pctchg`: Percent change from baseline
- `_slope_yr`: Rate of change per year (linear)
- `_mean`: Average across all timepoints
- `_std`: Standard deviation

## Quick Start

### Process 2 local subjects (test):
```bash
python process_all_subjects.py --max-subjects 2 --no-s3 --output test.csv
```

### Process all 456 subjects from S3:
```bash
python process_all_subjects.py --output features_all.csv
```

### With SynthSeg (if installed):
```bash
python process_all_subjects.py --method synthseg --output features_synthseg.csv
```

### Without segmentation (faster):
```bash
python process_all_subjects.py --no-segmentation --output features_noseg.csv
```

## Command Line Options

```
--metadata PATH          Path to metadata TSV (default: data/subject_metadata.tsv)
--output FILE           Output CSV file (default: features_with_segmentation.csv)
--no-s3                 Use local files only, no S3 access
--no-segmentation       Skip segmentation feature extraction
--method {simple,synthseg}  Segmentation method (default: simple)
--max-subjects N        Limit to first N subjects (for testing)
```

## Output Features

The output CSV contains ~80-100 features per subject:

### Survival Labels
- `event_observed`: 1 if MCI→AD conversion, 0 if censored
- `event_time_years`: Time to conversion or censoring
- `baseline_diagnosis`: Diagnosis at baseline (1=CN, 2=MCI, 3=AD)

### Baseline Segmentation (`_bl` suffix)
- `seg_hippocampus_total_mm3_bl`: Total hippocampus volume
- `seg_hippocampus_norm_bl`: Normalized by TIV
- `seg_ventricles_total_mm3_bl`: Total ventricle volume
- `seg_ventricles_norm_bl`: Normalized ventricles
- `seg_bpf_bl`: Brain parenchymal fraction
- `seg_gm_total_mm3_bl`: Gray matter volume
- `seg_wm_total_mm3_bl`: White matter volume
- `seg_tiv_mm3_bl`: Total intracranial volume

### Longitudinal Features
- `long_brain_vol_slope_yr`: Brain volume change rate
- `long_brain_vol_pctchg`: % change in brain volume
- `long_snr_slope_yr`: SNR decline rate
- `long_n_scans`: Number of scans used

### QC Features (`qc_` prefix)
- Brain intensity statistics
- SNR estimates
- Brain mask volumes

### Metadata (`meta_` and `hdr_` prefixes)
- Scanner parameters (TR, TE, flip angle, field strength)
- Image dimensions and voxel sizes
- Manufacturer, model

## Performance

- **Simple segmentation**: ~15-20 seconds per subject
- **SynthSeg**: ~2-3 minutes per subject
- **Full cohort (456 subjects)**:
  - Simple method: ~2-3 hours
  - SynthSeg: ~20-30 hours

## Notes

- Results saved every 50 subjects as backup
- S3 access requires AWS credentials configured
- Segmentation features have `_bl` suffix for baseline
- Missing data handled gracefully (None values)
