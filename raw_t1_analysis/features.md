# Time-to-conversion (survival modeling).

## 1) Final flattened column schema (one row per subject; ~405 rows)

Below is a **single “subjects” table schema** that’s (a) leakage-safe for survival, (b) small enough to work with 405 subjects, and (c) extendable later.

### A. IDs, cohort bookkeeping, follow-up

* `subject_id` (string)
* `n_visits_total` (int) — total scans found
* `n_visits_used` (int) — scans used for features (up to event/censor; see leakage rule below)
* `followup_years` (float) — from baseline MCI to last scan
* `manufacturer_mode` (string) — most common across used visits
* `model_mode` (string) — most common across used visits
* `field_strength_mode_t` (float) — most common across used visits
* `site_mode` (string) — most common across used visits (if available)

### B. Survival label columns (what you train on)

* `event_observed` (int) — 1 if converts MCI→AD during follow-up, else 0
* `event_time_years` (float) — time from baseline MCI to first AD (if event) else to last scan (censor)
* `mci_bl_datetime` (datetime) — baseline MCI date/time
* `event_datetime` (datetime) — first AD date/time (null if censored)
* `censor_datetime` (datetime) — last available scan date/time

**Optional (for AFT survival objective)**

* `aft_y_lower` (float) — = `event_time_years` for all
* `aft_y_upper` (float) — = `event_time_years` if event else `+inf`

### C. Baseline protocol/geometry controls (from baseline MCI scan)

(JSON + NIfTI header)

* `meta_field_strength_t_bl` (float)
* `meta_tr_s_bl` (float)
* `meta_te_s_bl` (float)
* `meta_ti_s_bl` (float)
* `meta_flip_angle_deg_bl` (float)
* `hdr_dim_x_bl` (int)
* `hdr_dim_y_bl` (int)
* `hdr_dim_z_bl` (int)
* `hdr_vox_x_mm_bl` (float)
* `hdr_vox_y_mm_bl` (float)
* `hdr_vox_z_mm_bl` (float)
* `hdr_voxvol_mm3_bl` (float)
* `hdr_fov_x_mm_bl` (float)
* `hdr_fov_y_mm_bl` (float)
* `hdr_fov_z_mm_bl` (float)

### D. Baseline QC / intensity (from baseline MCI scan)

(Use MRIQC-style if you can; otherwise compute simple brain/background stats)

* `qc_brain_mask_vol_mm3_bl` (float)
* `qc_brain_mean_bl` (float)
* `qc_brain_std_bl` (float)
* `qc_brain_p01_bl` (float)
* `qc_brain_p50_bl` (float)
* `qc_brain_p99_bl` (float)
* `qc_bg_mean_bl` (float)
* `qc_bg_std_bl` (float)
* `qc_brain_bg_ratio_bl` (float)
* `qc_snr_bl` (float, optional)
* `qc_cnr_bl` (float, optional)
* `qc_efc_bl` (float, optional)
* `qc_cjv_bl` (float, optional)

### E. Baseline global morphometry (Tier 3+)

* `seg_tiv_mm3_bl` (float)
* `seg_gm_mm3_bl` (float)
* `seg_wm_mm3_bl` (float)
* `seg_csf_mm3_bl` (float)
* `seg_brain_mm3_bl` (float) — GM+WM
* `seg_bpf_bl` (float) — (GM+WM)/TIV
* `seg_ventricles_mm3_bl` (float)
* `seg_ventricles_norm_bl` (float) — ventricles/TIV

### F. Baseline AD-relevant ROI volumes/thickness (Tier 4/5+)

Pick **one** of these sets depending on what your parcellation produces:

**If you have volumes only**

* `roi_hippocampus_l_vol_norm_bl` (float)
* `roi_hippocampus_r_vol_norm_bl` (float)
* `roi_hippocampus_asym_bl` (float)
* `roi_amygdala_l_vol_norm_bl` (float)
* `roi_amygdala_r_vol_norm_bl` (float)
* `roi_entorhinal_l_vol_norm_bl` (float)
* `roi_entorhinal_r_vol_norm_bl` (float)
* `roi_parahippocampal_l_vol_norm_bl` (float)
* `roi_parahippocampal_r_vol_norm_bl` (float)
* `roi_inferior_temporal_l_vol_norm_bl` (float)
* `roi_inferior_temporal_r_vol_norm_bl` (float)
* `roi_middle_temporal_l_vol_norm_bl` (float)
* `roi_middle_temporal_r_vol_norm_bl` (float)
* `roi_fusiform_l_vol_norm_bl` (float)
* `roi_fusiform_r_vol_norm_bl` (float)
* `roi_precuneus_l_vol_norm_bl` (float)
* `roi_precuneus_r_vol_norm_bl` (float)
* `roi_posterior_cingulate_l_vol_norm_bl` (float)
* `roi_posterior_cingulate_r_vol_norm_bl` (float)
* `roi_inferior_parietal_l_vol_norm_bl` (float)
* `roi_inferior_parietal_r_vol_norm_bl` (float)
* `roi_lateral_ventricle_l_norm_bl` (float)
* `roi_lateral_ventricle_r_norm_bl` (float)
* `roi_third_ventricle_norm_bl` (float)

**If you have cortical thickness too**
Add:

* `roi_mean_cortical_thick_mm_bl` (float)
* `roi_entorhinal_l_thick_mm_bl` (float)
* `roi_entorhinal_r_thick_mm_bl` (float)
* `roi_inferior_temporal_l_thick_mm_bl` (float)
* `roi_inferior_temporal_r_thick_mm_bl` (float)
* `roi_middle_temporal_l_thick_mm_bl` (float)
* `roi_middle_temporal_r_thick_mm_bl` (float)
* `roi_fusiform_l_thick_mm_bl` (float)
* `roi_fusiform_r_thick_mm_bl` (float)
* `roi_precuneus_l_thick_mm_bl` (float)
* `roi_precuneus_r_thick_mm_bl` (float)
* `roi_posterior_cingulate_l_thick_mm_bl` (float)
* `roi_posterior_cingulate_r_thick_mm_bl` (float)
* `roi_inferior_parietal_l_thick_mm_bl` (float)
* `roi_inferior_parietal_r_thick_mm_bl` (float)
* `roi_ad_signature_thick_mm_bl` (float) — mean of a chosen AD-signature set

### G. Longitudinal change features (computed using scans up to event/censor)

For a **small biomarker set** `X ∈ {hippocampus_norm, entorhinal_thick (or volume), ventricles_norm, bpf, ad_signature_thick}` compute:

For each `X`, include:

* `long_<X>_last` (float)
* `long_<X>_delta` (float)
* `long_<X>_pctchg` (float)
* `long_<X>_slope_yr` (float)
* `long_<X>_mean` (float)
* `long_<X>_std` (float)
* `long_<X>_n_obs` (int)

Concrete recommended set (keeps features manageable):

* `long_hippocampus_norm_*` (use mean of L/R or include both separately)
* `long_entorhinal_*` (thickness if available, else volume norm)
* `long_ventricles_norm_*`
* `long_bpf_*`
* `long_ad_signature_thick_*` (if thickness available)

### Leakage rule (critical)

When building all `long_*` features:

* For **converters**: only use scans with `acq_datetime <= event_datetime` (choose `<=` or `<` and stick to it; I’d use `<= first AD scan` if you interpret that scan as “at conversion time”).
* For **censored**: use all scans up to `censor_datetime` (last scan).

This ensures you’re not using post-event information.

---

## 2) How do you create labels? Is this regression or classification?

### You are doing **survival analysis**

That means your “labels” are **(time, event)**:

* **`event_observed`**: did the subject convert from MCI → AD during observed follow-up?

  * `1` if they ever have an AD-labeled scan after baseline MCI
  * `0` if they never have AD during follow-up (censored)

* **`event_time_years`**: time from baseline MCI to:

  * first AD scan (if event_observed=1), else
  * last available scan (if event_observed=0)

That’s it. Those two columns define the survival target.

### Is it regression or classification?

It’s **neither in the usual sense**:

* **Not standard classification** because “stable MCI” isn’t truly negative forever — some are just not observed long enough (censoring).
* **Not standard regression** because the time is **censored** for non-converters (you don’t know true conversion time).

Survival learning handles this properly.

### What does the model output?

Depending on the survival method:

* Cox-style models output a **risk score** (relative hazard)
* Discrete-time hazard models output **probability of converting in each time bin**
* AFT models output a **distribution/estimate of time-to-event**

You can still turn it into a classification decision, e.g.:

> “Probability of conversion within 24 months > 0.5”

…but that’s derived from the survival output, not the training label.

---

### Minimal label-building algorithm (from your folder structure)

For each subject:

1. Sort visits by datetime folder name
2. Identify baseline:

   * first visit where label is MCI (dx_bin=0)
   * exclude subject if no MCI visit (for an “MCI→AD” study)
3. Find event:

   * first visit after baseline with label AD (dx_bin=1)
4. Set:

   * `event_observed = 1` if event exists else 0
   * `event_time_years = (event_datetime - baseline_datetime)/365.25` if observed
     else `(last_datetime - baseline_datetime)/365.25`

---

If you tell me whether you consider the **first AD-labeled scan** as the conversion time (common) or you want something like “midpoint between last MCI and first AD”, I can recommend the cleanest convention and how to keep it consistent across subjects.


Perfect — that’s the **cleanest and most defensible choice**, and it’s exactly what most ADNI survival papers do.

Below I’ll lock that choice in and give you a **precise, unambiguous labeling rule** you can implement and cite, plus a couple of sanity checks so you don’t accidentally leak information.

---

## Final label definition (locked)

### Conversion time definition

> **Conversion time = datetime of the first scan labeled AD after baseline MCI**

No midpoint, no interpolation.

---

## Step-by-step label construction (implementation-ready)

For **each subject**:

### 1) Sort visits

Sort all visits by `acq_datetime` (parsed from folder name or JSON).

---

### 2) Define baseline (time = 0)

* Baseline = **earliest scan labeled MCI**
* Denote its datetime as:

```text
mci_bl_datetime
```

**Exclusion rule**

* If subject has **no MCI scan**, exclude from this study.
* If subject has AD before their first MCI, exclude (or treat separately, but do not include here).

---

### 3) Define conversion event

* Find the **first scan strictly after baseline** where label == AD.
* If found:

  * `event_observed = 1`
  * `event_datetime = datetime_of_first_AD_scan`
* Else:

  * `event_observed = 0`
  * `event_datetime = null`

---

### 4) Define censoring time

* If `event_observed == 1`:

  * `censor_datetime = event_datetime`
* Else:

  * `censor_datetime = datetime_of_last_available_scan`

---

### 5) Compute survival time

```text
event_time_years =
    (censor_datetime - mci_bl_datetime).total_seconds()
    / (365.25 * 24 * 3600)
```

This definition ensures:

* Converters: time = MCI → first AD
* Non-converters: time = MCI → last observed scan (right-censored)

---

### 6) Optional AFT labels (if you use XGBoost AFT)

```text
aft_y_lower = event_time_years
aft_y_upper =
    event_time_years      if event_observed == 1
    +infinity             if event_observed == 0
```

---

## Leakage-safe feature usage (important)

Because you’ve defined conversion time as the **first AD scan**, the feature rule becomes very clean:

### Allowed scans for feature extraction

* **Converters**: use only scans with

  ```text
  acq_datetime ≤ event_datetime
  ```
* **Censored subjects**: use all scans up to last scan

This is consistent with:

> “At the moment the subject is diagnosed as AD, what information was available?”

You are *not* using future scans beyond diagnosis.

---

## Special edge cases & how to handle them

### Case 1: MCI → AD → MCI → AD

Very common in ADNI due to diagnostic noise.

**Rule (recommended)**

* **First AD locks the event**
* Ignore any later MCI reversion

This aligns with clinical practice and avoids ambiguous survival times.

---

### Case 2: AD at baseline

* Exclude from this study (not an MCI→AD progression problem).

---

### Case 3: Single MCI scan, no follow-up

* Valid censored subject
* `event_observed = 0`
* `event_time_years ≈ 0`
* Keep, but note these add limited information.

---

## Is this classification or regression? (final answer)

### Training problem

You are solving a **right-censored survival analysis problem**.

Formally:

* Target = `(event_time_years, event_observed)`
* This is **neither standard classification nor standard regression**.

### What XGBoost is doing

If you use:

* `objective = "survival:cox"` → proportional hazards model
* `objective = "survival:aft"` → accelerated failure time model

XGBoost learns:

* A **risk function over time**, not a hard label

---

## How to get a binary answer (if needed later)

After training, you can derive:

* Probability of conversion **within T years** (e.g. 24 months)
* Or risk score → threshold → binary decision

But that’s **post-hoc** and does not change how you train.

---

## Minimal checklist before you train

Before fitting the model, verify:

* [ ] Each subject has exactly **one row**
* [ ] `event_time_years > 0` for all subjects
* [ ] No features use scans **after** `event_datetime`
* [ ] Train/val/test split is **subject-level**
* [ ] Stratify splits by `event_observed` (and optionally site)

---

## Why this setup is scientifically strong

* Matches ADNI conventions
* Handles censored stable-MCI subjects correctly
* Avoids “stable MCI = negative forever” fallacy
* Produces time-dependent risk estimates
* Cleanly extensible to deep longitudinal models later

If you want, next I can:

* suggest **XGBoost survival hyperparameters** that work well at N≈400,
* show how to **evaluate** (C-index, time-dependent AUC),
* or sketch a **code template** that builds these labels directly from your folder structure.

Great questions — this is exactly the right moment to pause and get the concepts **rock-solid**. Survival analysis *looks* confusing at first because it doesn’t fit the usual “one y per row” mental model, but once the intuition clicks, it’s actually very clean.

I’ll answer your three questions **in order**, with concrete ADNI examples and minimal math.

---

# 1) What is **censoring**?

### Plain-English definition

**Censoring means you don’t know the true outcome time — only that it hasn’t happened yet (up to a point).**

In your ADNI project:

* Some MCI subjects **convert to AD** during the study
  → you know *when* conversion happened.
* Some MCI subjects **do not convert** before the study ends
  → you **do not know** when (or if) they will convert in the future.

Those second subjects are **right-censored**.

---

### Concrete ADNI example

| Subject | What you observe                  | What you *don’t* know                  |
| ------- | --------------------------------- | -------------------------------------- |
| A       | Converts to AD at year 2.3        | Nothing missing                        |
| B       | Followed for 4.1 years, still MCI | Will they convert at year 5? 8? Never? |

Subject B is **censored at 4.1 years**.

---

### Why classification fails here

If you do binary classification:

| Subject | Label            |
| ------- | ---------------- |
| A       | 1 (converter)    |
| B       | 0 (stable MCI) ❌ |

This assumes B is *truly* negative forever — which is wrong.
They’re only “negative **so far**”.

Survival analysis fixes this by saying:

> “B survived (remained MCI) **at least** 4.1 years.”

That’s censoring.

---

### Key rule

* **Event observed** → exact time known
* **Censored** → event time unknown, but **greater than last follow-up**

---

# 2) What is **AFT**?

AFT = **Accelerated Failure Time** model.

Think of survival models as answering one of two questions:

---

## Two ways to model time-to-event

### A) Cox model (most common)

> “How does a feature **change the risk** of converting at any moment?”

* Outputs a **risk score**
* Relative: “twice the risk”, “half the risk”
* Does **not** directly predict time

This is `survival:cox` in XGBoost.

---

### B) AFT model (more intuitive)

> “How do features **speed up or slow down the clock** until conversion?”

* Directly models **time to AD**
* Example interpretation:

  > “Smaller hippocampus → conversion happens sooner”

This is `survival:aft` in XGBoost.

---

### Why AFT is nice for you

* You can think of it as **regression on time**
* But it **correctly handles censored data**
* Output can be turned into:

  * expected time to conversion
  * probability of converting within N years

---

### Why AFT needs *two* numbers (this matters for Q3)

For censored subjects:

* You **don’t know** the exact conversion time
* You only know:

  ```text
  true_time > last_followup_time
  ```

So AFT models accept **intervals**:

| Subject   | True time lies in |
| --------- | ----------------- |
| Converter | [2.3, 2.3]        |
| Censored  | [4.1, +∞)         |

That’s why you give **lower and upper bounds**.

---

# 3) “Two labels?? I only know regression & classification”

This is the key confusion — and it’s a *very* common one.

### Short answer

> **You are not predicting two labels.**
> You are predicting **one thing: time to AD**, but some times are unknown.

The “two labels” are just **how we describe uncertainty** during training.

---

## The survival label is NOT two outputs

You are **not** training the model to predict:

```text
(y1, y2)
```

Instead, you are telling the model:

> “The true time lies in this interval.”

---

## Think of it this way (analogy)

### Ordinary regression

```text
y = 2.3 years
```

### Censored regression

```text
y > 4.1 years
```

AFT needs a way to express both cases **numerically**.

So we encode:

| Case           | Lower bound | Upper bound |
| -------------- | ----------- | ----------- |
| Event observed | 2.3         | 2.3         |
| Censored       | 4.1         | +∞          |

That’s it.

---

## What the model actually predicts

The model predicts **one value per subject**:

* A latent time-to-event distribution
* Or a point estimate of log(time)

It is **not predicting two numbers**.

---

## How XGBoost uses this (very concrete)

For AFT in XGBoost:

You pass:

```python
y_lower = [2.3, 4.1, 1.8, ...]
y_upper = [2.3, inf, 1.8, ...]
```

Internally, XGBoost:

* treats exact times as exact
* treats censored ones as “greater than”

During prediction:

```python
pred = model.predict(X)
```

You get:

* a **single prediction per subject**

---

## If this still feels weird, use this mental model

### Survival analysis = regression with missing targets

But:

* missingness is **structured**
* “missing” means “greater than this value”

Classification throws away that information. Survival keeps it.

---

## Mapping to what you already know

| You know             | Survival equivalent        |
| -------------------- | -------------------------- |
| Regression           | AFT (with censoring)       |
| Classification       | Discrete-time hazard model |
| Label = y            | Label = (time, event)      |
| Missing y → drop row | Missing y → censored       |

---

## Minimal example with your ADNI setup

### Your table (one row per subject)

| subject | event_observed | event_time_years |
| ------- | -------------- | ---------------- |
| S1      | 1              | 2.3              |
| S2      | 0              | 4.1              |
| S3      | 1              | 1.6              |

### AFT encoding

| subject | y_lower | y_upper |
| ------- | ------- | ------- |
| S1      | 2.3     | 2.3     |
| S2      | 4.1     | +∞      |
| S3      | 1.6     | 1.6     |

### Prediction

```text
model(X) → predicted_time_or_distribution
```

Still **one prediction per subject**.

---

## Final intuition checkpoint (important)

* You are predicting **when** AD happens
* Some people haven’t reached AD yet
* You must not pretend they never will
* Survival analysis lets you say:

  > “At least this long, maybe longer”

That’s censoring, and that’s why the labels look different.

---

If you want, next I can:

* walk through a **toy 5-subject example with numbers**,
* show **exact XGBoost code** for Cox vs AFT,
* or show how to convert survival predictions into
  **“probability of AD within 2 years”**, which often makes everything click.
