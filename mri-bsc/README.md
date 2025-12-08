# MRI-BSC: Boundary Sharpness Coefficient Pipeline (OASIS-3 + NACC)

**BSC (Boundary Sharpness Coefficient)** quantifies how sharply gray matter transitions into white matter on T1-weighted MRI.
Units are **SD/mm** (standard-deviation change in normalized intensity per millimeter).

This repository provides a **modular, engine-agnostic** pipeline to compute BSC using:
- **Segmentation engines:** SynthSeg (DL, contrast-agnostic), FreeSurfer/FastSurfer (surface-based), or ANTs Atropos (classical baseline).
- **Preprocessing:** MONAI (N4, spacing, orientation) or ANTsPyX (optional).
- **I/O & QC:** NiBabel/Nilearn for NIfTI handling and quick-look figures.
- **Evaluation:** Site robustness (OASIS-3 vs NACC), optional clinical association/calibration (if labels available).

> You add the data locally. This repo contains **all code & folder structure**.

## 1) Environment (UV or pip)

**Using `uv` (recommended):**
```bash
cd mri-bsc
uv venv
uv pip install -e .
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

> External tools you may optionally install and point to:
> - **FreeSurfer** (`recon-all`, `mri_vol2surf`, `mri_synthstrip`) with `SUBJECTS_DIR` set.
> - **SynthSeg** CLI (or Docker) for segmentation posteriors.
> - **dcm2niix** for DICOM→NIfTI conversion (for NACC).

## 2) Folder layout

```
mri-bsc/
├─ code/
│  ├─ bsc/                 # core math
│  ├─ ingest/              # manifests for OASIS/NACC
│  ├─ preprocess/          # MONAI or ANTs preproc
│  ├─ skullstrip/          # SynthStrip / HD-BET / ANTs mask
│  ├─ seg/                 # Atropos, SynthSeg, FreeSurfer engines
│  ├─ eval/                # site robustness, calibration, association
│  ├─ pipeline/            # batch runner
│  └─ utils/               # IO, QC visuals, atlas warp
├─ configs/
│  ├─ paths.yaml           # data folders
│  ├─ pipeline.yaml        # choose engines & params
│  └─ atlas/               # put your atlas NIfTIs here (README included)
├─ data/
│  ├─ raw/                 # you add OASIS-3 (NIfTI) & NACC (DICOM/BIDS)
│  ├─ derivatives/         # outputs (preproc, seg, bsc, qc)
│  ├─ manifests/           # CSV manifests
│  └─ splits/              # split definitions (e.g., leave-one-site-out)
├─ scripts/                # example scripts
├─ env/                    # environment files (pyproject.toml for uv/pip)
├─ .gitignore
└─ README.md
```

> Never commit PHI or raw imaging data. `.gitignore` includes `data/raw/**`, NIfTI, DICOM, and heavy outputs.

## 3) Minimal quick start

### 3.1 Create manifests
- **OASIS-3 (NIfTI you copied locally):**
```bash
python code/ingest/build_manifest_oasis.py   --root data/raw/oasis3/nifti   --out_csv data/manifests/oasis3_manifest.csv   --qc_dir data/derivatives/qc/oasis3_manifest --n_qc 24
```

- **NACC (after you run `dcm2niix` to BIDS-like NIfTI):**
```bash
python code/ingest/build_manifest_nacc.py   --root data/raw/nacc/bids   --out_csv data/manifests/nacc_manifest.csv   --qc_dir data/derivatives/qc/nacc_manifest --n_qc 24
```

### 3.2 Preprocess (MONAI)
```bash
python code/preprocess/monai_preproc.py   --manifest data/manifests/oasis3_manifest.csv   --out_dir data/derivatives/preprocess/oasis3 --n 20
```

### 3.3 Seg + BSC (pick one engine first)

- **Atropos (classical, end-to-end):**
```bash
python code/seg/atropos_bsc.py   --t1 data/derivatives/preprocess/oasis3/sub-XXX_ses-YYY_T1w_preproc.nii.gz   --out_dir data/derivatives/bsc/oasis3/atropos/sub-XXX_ses-YYY
```

- **SynthSeg (posteriors):**
```bash
python code/seg/synthseg_bsc.py   --t1 data/derivatives/preprocess/oasis3/sub-XXX_ses-YYY_T1w_preproc.nii.gz   --posteriors data/derivatives/seg/oasis3/synthseg/sub-XXX_ses-YYY_posteriors.nii.gz   --label_map_json configs/synthseg_label_map.json   --out_dir data/derivatives/bsc/oasis3/synthseg/sub-XXX_ses-YYY
```

- **FreeSurfer surfaces (after recon-all or FastSurfer):**
```bash
python code/seg/freesurfer_bsc.py   --subjects_dir /path/to/SUBJECTS_DIR   --subject_id sub-XXX_ses-YYY   --t1_mgz mri/brain.mgz   --out_dir data/derivatives/bsc/oasis3/freesurfer/sub-XXX_ses-YYY
```

### 3.4 Site robustness & plots
```bash
python code/eval/site_robustness.py   --bsc_csv_glob "data/derivatives/bsc/*/*/*/subject_metrics.csv"   --out_dir data/derivatives/qc/site_plots
```

## 4) Configuration

Edit `configs/paths.yaml` and `configs/pipeline.yaml` to point to your data and to select engines (preproc, skullstrip, seg).

## 5) Notes

- Units: **BSC** is in **SD/mm** (standard deviation per millimeter).
- Boundary band: default GM probability ~0.5±0.05.
- For FreeSurfer sampling, we use `mri_vol2surf` to sample intensities along white-surface normals.
- For SynthSeg, supply a **label map** to combine posteriors into GM/WM (see `configs/synthseg_label_map.json` template).
