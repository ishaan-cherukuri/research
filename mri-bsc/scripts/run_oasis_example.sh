#!/usr/bin/env bash
set -e

echo "==== OASIS3 PIPELINE START ===="

# -------------------------
# 1. Build OASIS3 Manifest
# -------------------------
echo "Building OASIS3 manifest..."

python -m code.ingest.build_manifest_oasis \
  --scan_csv data/raw/oasis3/OASIS3_MR_scan_summary.csv \
  --zip_path data/raw/oasis3/OASIS3_derivative_files.zip \
#   --nifti_root data/raw/oasis3/nifti \
  --out_csv data/manifests/oasis3_manifest.csv

# -------------------------
# 2. Preprocess Images
# -------------------------
echo "Running preprocessing..."

python -m code.preprocess.monai_preproc \
  --manifest data/manifests/oasis3_manifest.csv \
  --out_dir data/derivatives/preprocess/oasis3

# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python -m code.pipeline.run_batch \
  --manifest data/manifests/oasis3_manifest.csv \
  --engine atropos \
  --out_root data/derivatives/bsc/oasis3/atropos

echo "==== OASIS3 PIPELINE COMPLETE ===="
