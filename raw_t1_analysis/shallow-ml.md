Below is a **practical, “solid” feature menu** you can extract into a **tabular dataset** from (a) the **per-scan JSON metadata** and (b) the **.nii image itself**, specifically with a **survival analysis** goal (time-to-conversion to AD, with censoring).

Structure this as:

1. **Per-scan features** (one row per scan/visit)
2. **Per-subject longitudinal summary features** (one row per subject)
3. A **recommended minimal baseline set** to start with (small N = 405)

---

## 1) Per-scan features from JSON metadata (scanner/protocol covariates)

These are valuable because ADNI has multi-site / multi-scanner variation and you don’t want imaging differences masquerading as disease signal.

### A. Scanner identity and site (categorical)

* `Manufacturer` (e.g., Siemens/GE/Philips)
* `ManufacturerModelName` or scanner model
* `DeviceSerialNumber` (often too specific; may overfit—use with caution)
* `Site` / `Center` (if present)
* `StationName`
* `SoftwareVersions`
* `BodyPartExamined` (should be head/brain)

**Engineering tip:** One-hot encode manufacturer/model/site; consider grouping rare categories.

### B. Sequence/protocol descriptors (categorical)

* `Modality` (MR)
* `SeriesDescription`, `ProtocolName`
* `ScanningSequence` (e.g., GR, SE)
* `SequenceVariant`, `ScanOptions`
* `MRAcquisitionType` (2D vs 3D)
* `ImageType` (e.g., ORIGINAL/DERIVED)
* `PulseSequenceDetails` (if present)

### C. Core acquisition parameters (numeric)

(Names vary by dataset; pull whatever exists)

* `MagneticFieldStrength` (1.5T vs 3T)
* `RepetitionTime` (TR)
* `EchoTime` (TE)
* `InversionTime` (TI) (often for MPRAGE)
* `FlipAngle`
* `PixelBandwidth`
* `EchoTrainLength`
* `NumberOfAverages` / `NEX`
* `ParallelReductionFactorInPlane` (acceleration factor)
* `PhaseEncodingDirection`
* `SAR` (specific absorption rate) if present

### D. Timing (use carefully)

* `AcquisitionTime` / `AcquisitionDateTime`
* `StudyDate`, `SeriesDate`

**Important:** Don’t use absolute dates as predictive features (they can proxy cohort/scanner upgrades). Convert them into:

* `months_since_subject_baseline` (safe and useful)
* optionally `age_at_scan` (if you have DOB elsewhere)

---

## 2) Per-scan features from the NIfTI itself (header + image-derived)

### A. NIfTI header geometry (robust and easy)

From `.nii` header (`pixdim`, shape, affine):

* `dim_x, dim_y, dim_z` (matrix size)
* `voxel_size_x, voxel_size_y, voxel_size_z` (mm)
* `voxel_volume_mm3 = vx*vy*vz`
* `FOV_x = dim_x*vx`, `FOV_y`, `FOV_z`
* `orientation` (RAS/LPS-like; from affine)
* `qform_code`, `sform_code` (sometimes correlates with processing pipelines)
* `datatype` (int16/float32)
* `slice_thickness` (if encoded separately)

These help catch protocol differences and can be strong confounders to control for.

---

## 3) Image quality / artifact features (high value baseline)

These features often explain performance differences and help you avoid “model learned scanner noise.”

### A. MRIQC-style QC metrics (recommended)

If you can run **MRIQC** (or implement similar):

* **SNR** (overall / tissue-specific if segmentation available)
* **CNR** (e.g., GM vs WM)
* **EFC** (entropy focus criterion; blur/ghosting proxy)
* **FBER** (foreground-background energy ratio)
* **CJV** (coefficient of joint variation; noise/bias proxy)
* **INU** / bias field strength estimates
* **WM2MAX** (white-matter to max intensity ratio)
* **Artifact indices** (ghosting, spikes) if available

Even a small set like `SNR`, `CNR`, `EFC`, `CJV` can be useful.

### B. Simple QC if you don’t want MRIQC yet

Compute after skull stripping (or approximate with thresholding):

* `brain_volume_vox` (brain mask voxel count)
* `mean_intensity_brain`, `std_intensity_brain`
* `p01/p50/p99` brain intensity percentiles
* `background_mean`, `background_std`
* `brain_to_background_ratio`

---

## 4) Core morphometric / neurodegeneration features (most predictive, most interpretable)

This is usually the strongest “tabular baseline” for AD progression.

### A. Global brain volumes (per scan)

Requires tissue segmentation (GM/WM/CSF), e.g., FAST/ANTs/FreeSurfer/FastSurfer:

* `TIV` / `ICV` (total intracranial volume)
* `total_brain_volume` (GM+WM)
* `GM_volume`, `WM_volume`, `CSF_volume`
* `brain_parenchymal_fraction = (GM+WM)/TIV`
* `ventricle_volume_total` (or lateral + third)

### B. AD-relevant ROI volumes (per scan)

From atlas/segmentation (FreeSurfer/FastSurfer/ANTs-atlas):

* Left/right **hippocampus volume**
* Left/right **amygdala volume**
* Left/right **entorhinal cortex volume/thickness**
* **parahippocampal gyrus** volume/thickness
* **inferior temporal**, **middle temporal** volume/thickness
* **fusiform** volume/thickness
* **temporal pole**
* **posterior cingulate**, **precuneus**
* **inferior parietal**
* **lateral ventricles** (L/R), **3rd ventricle**

### C. Cortical thickness summary features (per scan)

If you have surface-based measures:

* `mean_cortical_thickness`
* `AD_signature_thickness_mean` (average thickness across an “AD signature” region set)
* regional thickness values for the ROIs above

### D. Normalized and asymmetry features (cheap + strong)

Normalization:

* `hippocampus_L_norm = hippo_L / TIV`
* `hippocampus_R_norm = hippo_R / TIV`
* `ventricle_norm = ventricle / TIV`

Asymmetry (often informative):

* `hippocampus_asym = (L - R) / (L + R + eps)`
* similar for amygdala, temporal lobe ROIs

---

## 5) Radiomics / texture features (useful, but control dimensionality)

These can add predictive signal beyond volumes, but they can explode into thousands of columns.

### A. First-order intensity stats (within ROIs)

Compute within hippocampus / temporal GM / whole GM:

* mean, median, std, IQR
* skewness, kurtosis
* entropy, energy
* min/max, p10/p90

### B. Texture features (within ROIs)

Using something like PyRadiomics (within segmented ROIs):

* **GLCM**: contrast, correlation, homogeneity, ASM, dissimilarity, entropy
* **GLRLM**: short-run emphasis, long-run emphasis, run-length nonuniformity
* **GLSZM**: small/large area emphasis, zone nonuniformity
* **NGTDM**: coarseness, busyness, complexity
* **GLDM**: dependence nonuniformity, etc.

**Recommendation:** Start with **a handful of ROIs (hippocampus L/R, entorhinal L/R, temporal GM)** and keep features to a manageable set (or do PCA).

---

## 6) Longitudinal feature engineering (key for survival)

You have repeated scans per subject. Even with tabular XGBoost, you can extract “trajectory” features that carry progression info.

You can do this in two ways:

### Option A: One row per subject (baseline + slopes)

For each ROI/measure (hippocampus volume, ventricle volume, AD signature thickness, etc.) compute:

* `value_at_baseline`
* `value_at_last_followup_before_event_or_censor`
* `absolute_change = last - baseline`
* `percent_change = (last - baseline) / baseline`
* `annualized_slope` (fit linear regression vs time in years)
* `slope_se` or residual std (trajectory noise)
* `min`, `max`, `mean` over follow-up

Also add:

* `n_scans_used`
* `followup_duration_years`

This is often the cleanest baseline for survival with 405 subjects.

### Option B: Time-dependent covariates (one row per scan interval)

Create “start-stop” rows per subject:

* interval: `[t_i, t_{i+1}]`
* features at `t_i` (or change from baseline to `t_i`)
* event flag = 1 if conversion happens at end of interval

This better matches longitudinal survival theory, but it’s more bookkeeping and not every XGBoost survival setup handles start-stop cleanly. Still, it’s a strong framework if you implement it carefully.

---

## 7) Change-map / deformation features (powerful if you register longitudinally)

If you do within-subject registration (follow-up → baseline), you can compute deformation-based summaries:

* **Jacobian determinant** stats in ROIs:

  * mean log-Jacobian in hippocampus, temporal lobe, ventricles
  * captures local tissue shrinkage/expansion
* Difference-image stats in ROIs (after intensity normalization)

These can be extremely predictive for progression because they encode atrophy patterns directly.

---

## 8) “Solid minimal baseline feature set” I’d start with (to avoid overfitting)

Given only ~405 subjects, I’d start with something like:

### Per-subject baseline + slopes (maybe 30–80 columns total)

**Time/coverage**

* `n_scans`, `followup_years`

**Scanner/protocol controls (baseline scan)**

* `Manufacturer`, `Model`, `FieldStrength`
* `voxel_size_x/y/z`, `TR`, `TE`, `TI`, `FlipAngle` (whatever exists)

**QC**

* `SNR`, `CNR`, `EFC` (or your simpler QC equivalents)

**Morphometry (baseline + annualized slope)**

* `TIV`
* hippocampus L/R (normalized) + slope
* entorhinal L/R thickness or volume + slope
* ventricles total (normalized) + slope
* posterior cingulate / precuneus / inferior parietal thickness (or volume) + slope
* mean cortical thickness + slope
* asymmetry indices (baseline and/or slope)

That baseline is usually already strong and interpretable, and you can later add radiomics or deformation features if needed.

---

## Small but important leakage note (for survival)

When you compute slopes/last values, make sure you only use scans **up to the event time** for converters, and up to **last follow-up** for censored subjects. Otherwise you accidentally let “post-conversion” anatomy influence the features.

---

# Column schema
Below is a practical data dictionary (column schema) plus a tiered extraction plan that’s designed for:

* **Longitudinal ADNI MRI** stored as `subject_id / datetime_folder / {scan.nii, meta.json}`
* A **survival** setup: time-to-AD conversion with censoring
* A **tabular baseline** (XGBoost) that’s leakage-safe and expandable

I’m going to propose **three tables** (recommended), because it makes the pipeline much cleaner:

1. `visits.parquet` — one row per scan/visit (raw + derived per-visit features)
2. `subjects_survival.parquet` — one row per subject (baseline + longitudinal summaries for survival models)
3. `intervals_survival.parquet` (optional) — start/stop “time-dependent covariate” format

You can start with (1) + (2) and ignore (3) until you want time-dependent Cox/hazard models.

---

## Naming conventions (so the schema stays sane)

**General**

* `snake_case`
* include **units** in names: `_mm`, `_mm3`, `_t` (Tesla), `_s` (seconds), `_deg`
* prefixes:

  * `meta_` = from JSON metadata
  * `hdr_` = from NIfTI header/affine
  * `qc_` = quality/intensity features
  * `seg_` = tissue/global segmentation volumes
  * `roi_` = atlas/ROI features
  * `long_` = longitudinal summary features (baseline, slope, delta, etc.)

**Longitudinal suffixes**

* `_bl` = baseline value (baseline = first MCI scan)
* `_last` = last available scan before event (converter) or before censoring
* `_delta` = `last - baseline`
* `_pctchg` = `(last - baseline) / (baseline + eps)`
* `_slope_yr` = annualized slope from regression vs time (years)

---

# 1) `visits.parquet` schema (one row per scan)

This table is your “source of truth” for modeling + aggregation.

### A. Identifiers & timeline

| Column                |       Type | Description                                                    |
| --------------------- | ---------: | -------------------------------------------------------------- |
| `subject_id`          |     string | e.g. `002_S_0729`                                              |
| `visit_id`            |     string | e.g. folder name `2006-08-02_07_02_00.0`                       |
| `acq_datetime`        |   datetime | parsed from visit folder name or JSON                          |
| `visit_index`         |        int | 0,1,2… sorted by `acq_datetime`                                |
| `t_days_from_first`   |      float | days since subject’s first scan                                |
| `t_years_from_first`  |      float | years since subject’s first scan                               |
| `t_days_from_mci_bl`  |      float | days since baseline MCI scan (NaN if no MCI baseline)          |
| `t_years_from_mci_bl` |      float | years since baseline MCI scan                                  |
| `label_raw`           | int/string | original label (you mentioned “2 MCI, 3 AD”)                   |
| `dx_bin`              |        int | 0=MCI, 1=AD (mapped from `label_raw`)                          |
| `is_mci_baseline`     |        int | 1 if this scan is the subject’s baseline MCI scan              |
| `is_first_ad`         |        int | 1 if this scan is the first AD-labeled scan after baseline MCI |

### B. File pointers (optional but useful for traceability)

| Column      |   Type | Description                           |
| ----------- | -----: | ------------------------------------- |
| `nii_path`  | string | absolute/relative path to NIfTI       |
| `json_path` | string | path to metadata JSON                 |
| `file_hash` | string | optional checksum for reproducibility |

### C. JSON metadata features (`meta_*`)

*(Exact field names vary—extract what exists and keep raw strings too.)*

**Categorical (store as strings; encode later)**

| Column                          |   Type | Description                 |
| ------------------------------- | -----: | --------------------------- |
| `meta_modality`                 | string | expected `MR`               |
| `meta_manufacturer`             | string | Siemens/GE/Philips…         |
| `meta_model`                    | string | scanner model               |
| `meta_software_versions`        | string | scanner software version    |
| `meta_scanning_sequence`        | string | e.g. GR/SE variants         |
| `meta_sequence_variant`         | string | if present                  |
| `meta_mr_acquisition_type`      | string | 2D/3D                       |
| `meta_series_description`       | string | protocol/series name        |
| `meta_protocol_name`            | string | protocol name               |
| `meta_image_type`               | string | ORIGINAL/DERIVED etc.       |
| `meta_phase_encoding_direction` | string | e.g. `i/j/k` variants       |
| `meta_site`                     | string | if present (center/site ID) |

**Numeric**

| Column                           |  Type | Description             |
| -------------------------------- | ----: | ----------------------- |
| `meta_field_strength_t`          | float | 1.5, 3.0, …             |
| `meta_tr_s`                      | float | repetition time         |
| `meta_te_s`                      | float | echo time               |
| `meta_ti_s`                      | float | inversion time (MPRAGE) |
| `meta_flip_angle_deg`            | float | flip angle              |
| `meta_pixel_bandwidth`           | float | bandwidth               |
| `meta_echo_train_length`         | float | if present              |
| `meta_num_averages`              | float | NEX/averages            |
| `meta_parallel_factor`           | float | acceleration factor     |
| `meta_slice_thickness_mm`        | float | if present              |
| `meta_spacing_between_slices_mm` | float | if present              |

### D. NIfTI header/geometry features (`hdr_*`)

| Column                                         |       Type | Description                         |
| ---------------------------------------------- | ---------: | ----------------------------------- |
| `hdr_dim_x`, `hdr_dim_y`, `hdr_dim_z`          |        int | voxel matrix size                   |
| `hdr_vox_x_mm`, `hdr_vox_y_mm`, `hdr_vox_z_mm` |      float | voxel spacing                       |
| `hdr_voxvol_mm3`                               |      float | voxel volume                        |
| `hdr_fov_x_mm`, `hdr_fov_y_mm`, `hdr_fov_z_mm` |      float | field of view                       |
| `hdr_orientation`                              |     string | derived orientation (e.g., RAS-ish) |
| `hdr_qform_code`, `hdr_sform_code`             |        int | qform/sform indicators              |
| `hdr_datatype`                                 | string/int | int16/float32…                      |

### E. Minimal intensity / QC features (`qc_*`)

These can be computed early and help control confounds.

| Column                                         |  Type | Description                                      |
| ---------------------------------------------- | ----: | ------------------------------------------------ |
| `qc_brain_mask_vol_mm3`                        | float | brain mask volume (if you skull-strip)           |
| `qc_brain_mean`, `qc_brain_std`                | float | brain intensity stats                            |
| `qc_brain_p01`, `qc_brain_p50`, `qc_brain_p99` | float | brain percentiles                                |
| `qc_bg_mean`, `qc_bg_std`                      | float | background intensity stats                       |
| `qc_brain_bg_ratio`                            | float | `brain_mean / (bg_mean+eps)`                     |
| `qc_efc`                                       | float | entropy focus criterion (if you implement/MRIQC) |
| `qc_cjv`                                       | float | CJV noise/bias proxy (if implement/MRIQC)        |
| `qc_snr`                                       | float | SNR estimate (simple or MRIQC-style)             |
| `qc_cnr`                                       | float | CNR (if tissue masks exist)                      |

### F. Segmentation/global morphometry (`seg_*`)  *(Tier 3+)*

| Column                                    |  Type | Description                              |
| ----------------------------------------- | ----: | ---------------------------------------- |
| `seg_tiv_mm3`                             | float | total intracranial volume (ICV/TIV)      |
| `seg_gm_mm3`, `seg_wm_mm3`, `seg_csf_mm3` | float | tissue volumes                           |
| `seg_brain_mm3`                           | float | GM+WM                                    |
| `seg_bpf`                                 | float | brain parenchymal fraction = (GM+WM)/TIV |
| `seg_ventricles_mm3`                      | float | total ventricle volume                   |
| `seg_ventricles_norm`                     | float | ventricles/TIV                           |

### G. ROI features (`roi_*`) *(Tier 4+)*

I strongly recommend a “small, AD-relevant ROI set” first (keeps feature count reasonable). Use either FreeSurfer/FastSurfer ROIs or an atlas-based parcellation.

**Recommended ROI columns (examples; left/right where possible):**

* Hippocampus L/R
* Amygdala L/R
* Entorhinal cortex L/R
* Parahippocampal L/R
* Inferior temporal L/R
* Middle temporal L/R
* Fusiform L/R
* Temporal pole L/R
* Precuneus L/R
* Posterior cingulate L/R
* Inferior parietal L/R
* Lateral ventricles L/R
* 3rd ventricle

Schema pattern:

| Column pattern            |  Type | Description                     |
| ------------------------- | ----: | ------------------------------- |
| `roi_<roi_name>_vol_mm3`  | float | ROI volume                      |
| `roi_<roi_name>_vol_norm` | float | ROI volume normalized by TIV    |
| `roi_<roi_name>_asym`     | float | (L−R)/(L+R+eps) for paired ROIs |

If you do cortical thickness (Tier 5), add:

| Column pattern            |  Type | Description           |
| ------------------------- | ----: | --------------------- |
| `roi_<roi_name>_thick_mm` | float | mean thickness in ROI |

---

# 2) `subjects_survival.parquet` schema (one row per subject)

This is the **modeling table** for survival XGBoost (or any survival learner). It’s built by aggregating `visits.parquet`.

### A. Core survival label columns (Cox / AFT compatible)

| Column             |     Type | Description                                    |
| ------------------ | -------: | ---------------------------------------------- |
| `subject_id`       |   string | subject identifier                             |
| `mci_bl_datetime`  | datetime | baseline MCI scan datetime                     |
| `event_datetime`   | datetime | first AD datetime (if converter)               |
| `censor_datetime`  | datetime | last available scan datetime                   |
| `event_observed`   |      int | 1 if converted to AD, else 0                   |
| `event_time_years` |    float | time from MCI baseline to event/censor (years) |

**If you plan to use AFT with explicit censor bounds:**

| Column        |  Type | Description                              |
| ------------- | ----: | ---------------------------------------- |
| `aft_y_lower` | float | = event_time_years for all               |
| `aft_y_upper` | float | = event_time_years if event, else `+inf` |

### B. Cohort bookkeeping / QC

| Column                  |   Type | Description                               |
| ----------------------- | -----: | ----------------------------------------- |
| `n_visits_total`        |    int | total scans                               |
| `n_visits_post_bl`      |    int | scans after MCI baseline                  |
| `followup_years`        |  float | censor_time_years                         |
| `has_ad_at_baseline`    |    int | should be 0 for this study (else exclude) |
| `site_mode`             | string | most frequent site (if available)         |
| `manufacturer_mode`     | string | most frequent manufacturer                |
| `field_strength_mode_t` |  float | most frequent field strength              |

### C. Baseline covariates (pulled from the baseline MCI scan)

For any per-visit feature `X` you consider important, store:

* `X_bl` (baseline value)

Examples (recommended):

| Column                        |  Type | Description                                         |
| ----------------------------- | ----: | --------------------------------------------------- |
| `meta_field_strength_t_bl`    | float | baseline field strength                             |
| `hdr_voxvol_mm3_bl`           | float | baseline voxel volume                               |
| `qc_snr_bl`                   | float | baseline SNR                                        |
| `seg_tiv_mm3_bl`              | float | baseline TIV                                        |
| `roi_hippocampus_vol_norm_bl` | float | baseline normalized hippo                           |
| `roi_ventricles_norm_bl`      | float | baseline ventricle/TIV                              |
| `roi_ad_signature_thick_bl`   | float | baseline AD-signature thickness (if you compute it) |

### D. Longitudinal summary features (computed using scans up to event/censor)

For each core biomarker feature `X` (e.g., hippocampus_norm, ventricles_norm, AD-signature thickness), compute:

| Column pattern        |  Type | Description                              |
| --------------------- | ----: | ---------------------------------------- |
| `<X>_last`            | float | last available before event/censor       |
| `<X>_delta`           | float | last - baseline                          |
| `<X>_pctchg`          | float | (last - baseline)/(baseline+eps)         |
| `<X>_slope_yr`        | float | annualized slope from regression on time |
| `<X>_mean`, `<X>_std` | float | mean/std across follow-up visits         |
| `<X>_n_obs`           |   int | number of observations used              |

**Examples of `X` you should include early (small + strong):**

* `roi_hippocampus_vol_norm`
* `roi_entorhinal_thick_mm` (or volume norm)
* `roi_ventricles_norm`
* `seg_bpf` (brain parenchymal fraction)
* `roi_precuneus_thick_mm` (if available)
* `roi_posterior_cingulate_thick_mm` (if available)

---

# 3) `intervals_survival.parquet` (optional but powerful)

Use this if you want **time-dependent covariates** (start/stop). It can also support **discrete-time hazard** modeling.

| Column                |    Type | Description                               |
| --------------------- | ------: | ----------------------------------------- |
| `subject_id`          |  string | subject identifier                        |
| `t_start_yr`          |   float | interval start time since MCI baseline    |
| `t_end_yr`            |   float | interval end time since MCI baseline      |
| `event_at_end`        |     int | 1 if first AD occurs at `t_end_yr`        |
| `dx_bin_start`        |     int | dx at start (should be MCI if pre-event)  |
| `features_at_start_*` | numeric | your covariates evaluated at `t_start_yr` |
| `delta_from_bl_*`     | numeric | optional: feature(t_start) - feature(bl)  |
| `dt_yr`               |   float | interval length                           |

This table is more work, but it avoids compressing trajectories into slopes/deltas.

---

# Tiered extraction plan

This is a “start simple, add complexity only if it buys you lift” plan.

## Tier 0 — Indexing, labels, timelines, and cohort definition (mandatory)

**Goal:** build clean `visits.parquet` with IDs/time/labels only.

**Steps**

1. Parse directory structure: `subject_id` → sorted `visit_id` (datetime folders)
2. Read label for each scan (your MCI/AD code) → `dx_bin`
3. Define **baseline**:

   * baseline = earliest scan where `dx_bin == 0` (MCI)
   * exclude subjects with no MCI scan, or treat separately (depends on study)
4. Define **event**:

   * event time = first visit after baseline with `dx_bin == 1` (AD)
   * if never AD → censored at last scan
5. Create `subjects_survival.parquet` with `event_observed`, `event_time_years`

**Outputs**

* `visits.parquet` (minimal)
* `subjects_survival.parquet` (labels + follow-up coverage)

**QC checks**

* Are there “AD → MCI → AD” flips? Decide rule: usually “first AD locks event”.
* Confirm no subject-level leakage in splitting.

---

## Tier 1 — JSON metadata + NIfTI header features (fast, low risk)

**Goal:** add scanner/protocol/geometry covariates (helps control confounds).

**Steps**

* From JSON: extract manufacturer/model/field strength/TR/TE/TI/flip angle/etc.
* From NIfTI header: dims/voxel size/FOV/orientation/qform/sform/datatype

**Outputs**

* Adds `meta_*`, `hdr_*` columns to `visits.parquet`
* Add baseline versions (suffix `_bl`) to `subjects_survival.parquet`

**Why now?**

* Almost free, and avoids your model learning “scanner upgrade” instead of disease.

---

## Tier 2 — Lightweight intensity + brain mask QC (still relatively fast)

**Goal:** simple, robust QC/intensity features; brain volume proxy.

**Steps**

1. Reorient to standard (e.g., RAS) and optionally resample to consistent spacing
2. (Optional but helpful) N4 bias correction
3. Skull strip / brain mask (SynthStrip / HD-BET / another tool)
4. Compute `qc_*` features from brain + background

**Outputs**

* `qc_*` columns per visit
* baseline and longitudinal summaries in subject table

**Stop and evaluate**

* Train survival baseline with Tier 0–2 features to set your “floor”.

---

## Tier 3 — Tissue segmentation + global morphometry (high value)

**Goal:** get interpretable neurodegeneration proxies.

**Steps**

* Segment GM/WM/CSF (FAST/Atropos/etc.)
* Estimate TIV/ICV
* Compute global volumes + ventricle volume if feasible

**Outputs**

* `seg_tiv_mm3`, `seg_gm_mm3`, `seg_wm_mm3`, `seg_csf_mm3`, `seg_bpf`, `seg_ventricles_mm3`

**Why it’s worth it**

* Global + ventricle + TIV-normalized features are strong and stable.

---

## Tier 4 — ROI parcellation volumes (AD-relevant ROI set)

**Goal:** targeted ROIs that carry most of the predictive signal for AD progression.

**Steps**

* Register scan to template OR use a segmentation tool that outputs ROIs
* Apply atlas/parcellation and compute ROI volumes
* Normalize by TIV + compute asymmetry

**Outputs**

* `roi_*_vol_mm3`, `roi_*_vol_norm`, `roi_*_asym`

**Keep it small at first**

* Start with ~10–20 ROI pairs + ventricles; expand only if justified.

---

## Tier 5 — Cortical thickness (optional, more compute, often strong)

**Goal:** thickness measures are very informative for AD signature regions.

**Steps**

* Run FreeSurfer or FastSurfer per scan
* Extract regional thickness + AD-signature mean thickness

**Outputs**

* `roi_*_thick_mm`
* `roi_ad_signature_thick_mm` (your curated average across key regions)

---

## Tier 6 — Longitudinal change mechanics (big lift potential, careful with leakage)

**Goal:** extract explicit progression dynamics.

**Two levels**

1. **Trajectory summaries** (cheap, already in schema):

   * slopes, deltas, percent changes for key ROIs
2. **Deformation-based** (more complex):

   * within-subject registration (follow-up → baseline)
   * Jacobian determinant stats in ROIs (atrophy/expansion)

**Outputs**

* `long_*` features (subject table)
* optional `jac_*` ROI summaries if doing deformation fields

**Leakage rule**

* For converters: only use scans strictly **before or at** first AD event (decide and stick to it)
* Ideally: “predict conversion using information available up to time t”, which suggests building an interval table later.

---

## Tier 7 — Radiomics (optional, easy to overdo)

**Goal:** add texture signal without blowing up dimensionality.

**Steps**

* Compute radiomics only inside a few ROIs (hippocampus/entorhinal/temporal GM)
* Limit to a curated subset of features (or do PCA)

**Outputs**

* `rad_*` columns (but keep feature count controlled)

---

# Practical recommended “Phase 1” target (feature count + stability)

With **405 subjects**, I’d aim for an initial subject-level model with **~50–150 features**, not 2,000.

**Phase 1 tiers:** 0 → 3 (maybe 4 with a small ROI list)

This usually yields a strong, defensible baseline:

* scanner/protocol controls
* QC controls
* TIV + ventricles + hippocampus + a few temporal ROIs
* baseline + slope + delta

---

# Suggested deliverables you can implement in order

1. `visits.parquet` with Tier 0–1 columns
2. `subjects_survival.parquet` with correct `event_time_years` / `event_observed`
3. Add Tier 2 QC (`qc_*`)
4. Add Tier 3 global morphometry (`seg_*`)
5. Add Tier 4 ROI volumes for a small ROI set (`roi_*`)
6. Add longitudinal summaries (`*_slope_yr`, `*_delta`, `*_pctchg`)
7. Only then consider intervals table or deformation/radiomics

---

If you want, I can also write out a **concrete ROI column list** (exact names) for a specific parcellation choice (e.g., FreeSurfer aseg/aparc, Desikan-Killiany, Harvard-Oxford), and a leakage-safe rule set for “which scans count” when building `*_last` / slopes for converters.

