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

python3 -m code.pipeline.run_batch \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --engine atropos \
  --out_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --preproc_root s3://ishaan-research/data/derivatives/preprocess/adni_5

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

# echo "==== ADNI PIPELINE COMPLETE ===="
