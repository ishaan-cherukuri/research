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
#   --s3_raw_root "$S3_BASE/raw/adni/adni_3yr_3t" \
#   --csv_path "$S3_BASE/raw/adni/ADNI1_3Yr_3T_12_10_2025.csv" \
#   --out_csv "$S3_MANIFESTS/adni_manifest_test.csv" \

# # -------------------------
# # 2. Preprocess Images
# # -------------------------
echo "Running preprocessing..."

python3 -m code.preprocess.simple_preproc \
  --manifest s3://ishaan-research/data/manifests/adni_manifest_test.csv \
  --out_root s3://ishaan-research/data/derivatives/preprocess/adni


# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python3 -m code.pipeline.run_batch \
  --manifest "$S3_MANIFESTS/adni_manifest_test.csv" \
  --engine atropos \
  --out_root "$S3_DERIVS" \
  --work_dir "$WORK_DIR"

# -------------------------
# 4. Build longitudinal index
# -------------------------
echo "Building longitudinal index (S3)..."

python3 -m code.index.build_longitudinal_index \
  --manifest "$S3_MANIFESTS/adni_manifest_test.csv" \
  --labels "s3://ishaan-research/index/adni_mci_conversion_labels.csv" \
  --derivatives_root "$S3_DERIVS" \
  --out_csv "$S3_INDEX/adni_longitudinal_index.csv" \
  --out_unmatched_csv "$S3_INDEX/adni_longitudinal_index_unmatched.csv" \
  --require t1,gm,wm,mask

echo "==== ADNI PIPELINE COMPLETE ===="
