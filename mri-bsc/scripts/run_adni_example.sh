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
# 1. Build ADNI TEST Manifest (CSV + IMAGE FOLDER)
# -------------------------
# echo "Building ADNI TEST manifest..."

# python3 -m code.ingest.build_manifest_adni \
#   --s3_raw_root "$S3_BASE/raw/adni/adni_multi_model_3yr" \
#   --csv_path "$S3_BASE/raw/adni/Study_Key_Multi_Model_WITH_DIAGNOSIS-2.csv" \
#   --out_csv "$S3_MANIFESTS/adni_manifest.csv" \

# -------------------------
# 2. Preprocess Images
# -------------------------
# echo "Running preprocessing..."

# python3 -m code.preprocess.simple_preproc \
#   --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
#   --out_root s3://ishaan-research/data/derivatives/preprocess/adni


# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python3 -m code.pipeline.run_batch \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --engine atropos \
  --out_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --skip 96

# -------------------------
# 4. Build longitudinal index
# -------------------------
echo "Building longitudinal index (S3)..."

python -m code.index.build_mci_conversion_labels \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --out_csv  s3://ishaan-research/data/labels/mci_conversion_labels.csv \
  --out_unmatched_csv s3://ishaan-research/data/labels/mci_conversion_unmatched.csv

python -m code.index.build_longitudinal_index \
  --manifest s3://ishaan-research/data/manifests/adni_manifest.csv \
  --labels   s3://ishaan-research/data/labels/mci_conversion_labels.csv \
  --derivatives_root s3://ishaan-research/data/derivatives/bsc/adni/atropos \
  --out_csv  s3://ishaan-research/data/index/longitudinal_index.csv \
  --out_unmatched_csv s3://ishaan-research/data/index/index_unmatched.csv

echo "==== ADNI PIPELINE COMPLETE ===="
