#!/usr/bin/env bash
set -e

echo "==== ADNI PIPELINE START ===="

# -------------------------
# 1. Build ADNI Manifest
# -------------------------
echo "Building ADNI manifest..."

python3 -m code.ingest.build_manifest_adni \
  --zip_path data/raw/adni/Imaging.zip \
  --nifti_root data/raw/adni/nifti \
  --out_csv data/manifests/adni_manifest.csv

# -------------------------
# 2. Preprocess Images
# -------------------------
echo "Running preprocessing..."

python3 -m code.preprocess.monai_preproc \
  --manifest data/manifests/adni_manifest.csv \
  --out_dir data/derivatives/preprocess/adni

# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python3 -m code.pipeline.run_batch \
  --manifest data/manifests/adni_manifest.csv \
  --engine atropos \
  --out_root data/derivatives/bsc/adni/atropos

echo "==== ADNI PIPELINE COMPLETE ===="
