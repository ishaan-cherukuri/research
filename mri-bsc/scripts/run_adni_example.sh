#!/usr/bin/env bash
set -e

echo "==== ADNI PIPELINE START ===="

# -------------------------
# 1. Build ADNI TEST Manifest (CSV + IMAGE FOLDER)
# -------------------------
echo "Building ADNI TEST manifest..."

python3 -m code.ingest.build_manifest_adni \
  --adni_root data/raw/adni/adni_3yr_3t \
  --adni_csv data/raw/adin/ADNI1_3Yr_3T_12_10_2025.csv \
  --out data/manifests/adni_manifest_test.csv


# -------------------------
# 2. Preprocess Images
# -------------------------
echo "Running preprocessing..."

python3 -m code.preprocess.simple_preproc \
  --manifest data/manifests/adni_manifest_test.csv \
  --out_dir data/derivatives/preprocess/adni


# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python3 -m code.pipeline.run_batch \
  --manifest data/manifests/adni_manifest_test.csv \
  --engine atropos \
  --out_root data/derivatives/bsc/adni/atropos


echo "==== ADNI PIPELINE COMPLETE ===="
