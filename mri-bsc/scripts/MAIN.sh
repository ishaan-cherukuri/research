#!/usr/bin/env bash
set -e

echo "==== ADNI PIPELINE START (S3 MODE) ===="

# ---------------
# CONFIG
# ---------------
S3_BASE="s3://ishaan-research/data"
WORK_DIR=".cache/mri-bsc"
mkdir -p "$WORK_DIR"

S3_MANIFESTS="$S3_BASE/manifests"
S3_DERIVS="$S3_BASE/derivatives/bsc/adni/atropos"
S3_INDEX="$S3_BASE/index"

# -------------------------
# 0. Building Sanity Check
# -------------------------
# echo "SANITY CHECK COMMENCING..."

# python3 -m code.ingest.adni_sanity_check \
#   --s3-root s3://ishaan-research/data/raw/adni_5/ \
#   --helper-csv s3://ishaan-research/data/raw/final_df.csv \
#   --out s3://ishaan-research/data/manifests/sanity_checks.csv

# -------------------------
# 1. Build ADNI TEST Manifest (CSV + IMAGE FOLDER)
# -------------------------
# echo "Building ADNI TEST manifest..."

# python3 -m code.ingest.build_manifest_adni \
#   --s3-root s3://ishaan-research/data/raw/adni_5/ \
#   --helper-csv s3://ishaan-research/data/raw/final_df.csv \
#   --out-manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --out-report s3://ishaan-research/data/manifests/adni_subject_report.csv \


# -------------------------
# 2. Preprocess Images
# -------------------------
# echo "Running preprocessing..."

# python3 -m code.preprocess.simple_preproc \
#   --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --out_root s3://ishaan-research/data/derivatives/preprocess/adni_5/ \
#   --temp_root data/splits



# # -------------------------
# # 3. Run BSC Batch (Atropos)
# # -------------------------
# echo "Running BSC (Atropos)..."

# python3 -m code.pipeline.run_batch \
#   --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --engine atropos \
#   --out_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
#   --preproc_root s3://ishaan-research/data/derivatives/preprocess/adni_5 \
#   --temp_root data/splits \

# echo " BSC (Atropos)..."

# echo "Postprocessing BSC maps (hard mask + overwrite in S3)..."

# # macOS-safe temp control (prevents /private/var/folders/... temp blowups)
# TMPROOT="$PWD/data/splits/tmp"
# mkdir -p "$TMPROOT"
# export TMPDIR="$TMPROOT"
# export TMP="$TMPROOT"
# export TEMP="$TMPROOT"

# # -------------------------
# # 3b. Postprocess BSC outputs in S3 (overwrite)
# # -------------------------
# # This step enforces that voxel-wise BSC maps are exactly zero outside the boundary mask.
# # It overwrites (per scan folder):
# #   - bsc_dir_map.nii.gz
# #   - bsc_mag_map.nii.gz (if present)
# # Optional overwrite:
# #   - boundary_band_mask.nii.gz  (WRITE_MASK=1)
# #   - bsc_metrics.json + subject_metrics.csv (WRITE_METRICS=1)

# MASK_MODE="interface"        # band|interface
# NO_SKIP=1                    # 1 => overwrite all scans even if already masked
# WRITE_MASK=1                 # 1 => overwrite boundary_band_mask.nii.gz with final mask
# WRITE_METRICS=1              # 1 => recompute + overwrite metrics after masking

# # Optional FreeSurfer constraints (ONLY if FS outputs already exist in S3):
# # FS_ROOT="s3://<bucket>/<prefix-to-freesurfer-recons>"
# # FS_ID="subject"            # image_id|subject
# # FS_MASK="cortex"           # interface|cortex|both

# POST_ARGS=(
#   --manifest "${S3_MANIFESTS}/adni_manifest.csv"
#   --out_root  "${S3_DERIVS}"
#   --temp_root data/splits
#   --mask_mode "${MASK_MODE}"
# )

# if [[ "$NO_SKIP" == "1" ]]; then
#   POST_ARGS+=(--no_skip_if_already_masked)
# fi

# if [[ "$WRITE_MASK" == "1" ]]; then
#   POST_ARGS+=(--write_mask)
# fi

# if [[ "$WRITE_METRICS" == "1" ]]; then
#   POST_ARGS+=(--write_metrics)
# fi

# if [[ -n "${FS_ROOT:-}" ]]; then
#   POST_ARGS+=(--fs_root "${FS_ROOT}")
#   POST_ARGS+=(--fs_id "${FS_ID:-image_id}")
#   POST_ARGS+=(--fs_mask "${FS_MASK:-interface}")
# fi

# python3 -m code.pipeline.postprocess_mask_bsc_s3 "${POST_ARGS[@]}"

# # -------------------------
# # 4. Build longitudinal index
# # -------------------------
# echo "Building longitudinal index (S3)..."

# python -m code.index.build_mci_conversion_labels \
#   --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --out_csv  s3://ishaan-research/data/labels/mci_conversion_labels.csv \
#   --out_unmatched_csv s3://ishaan-research/data/labels/mci_conversion_unmatched.csv

# python -m code.index.build_longitudinal_index \
#   --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --labels   s3://ishaan-research/data/labels/mci_conversion_labels.csv \
#   --derivatives_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
#   --out_csv  s3://ishaan-research/data/index/longitudinal_index.csv \
#   --out_unmatched_csv s3://ishaan-research/data/index/index_unmatched.csv

# -------------------------
# 5. LOCAL BSC v2 + FEATURES (YAAGL MODE)
# -------------------------
# macOS-safe temp control (recommended)
TMPROOT="$PWD/data/splits/tmp"; mkdir -p "$TMPROOT"; export TMPDIR="$TMPROOT" TMP="$TMPROOT" TEMP="$TMPROOT"

# (Optional) FreeSurfer volume-only recons must already exist locally for --fs_cortex
# export FS_SUBJECTS_DIR="/Volumes/YAAGL/derivatives/freesurfer/adni"

python3 -m code.pipeline.bsc_v2_local \
  --manifest "$PWD/data/manifests/adni_manifest_test.csv" \
  --preproc_root "/Volumes/YAAGL/derivatives/preprocess/adni_5" \
  --out_root     "/Volumes/YAAGL/derivatives/bsc/adni/atropos_v2" \
  --work_root    "/Volumes/YAAGL/tmp/bsc_v2" \
  --skip_done \
  --write_mask \
  --fs_subjects_dir "$FS_SUBJECTS_DIR" \
  --fs_cortex

python3 -m code.features.extract_bsc_features \
  --manifest "$PWD/data/manifests/adni_manifest_test.csv" \
  --bsc_root  "/Volumes/YAAGL/derivatives/bsc/adni/atropos_v2" \
  --out_csv   "$PWD/code/index/bsc_scan_features_v2.csv" \
  --bins 2,2,2

python3 -m code.features.build_longitudinal_features \
  --manifest "$PWD/data/manifests/adni_manifest_test.csv" \
  --scan_features "$PWD/code/index/bsc_scan_features_v2.csv" \
  --out_csv "$PWD/code/index/bsc_subject_features_v2.csv"

# echo "==== ADNI PIPELINE COMPLETE ===="
