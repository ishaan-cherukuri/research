#!/usr/bin/env bash
set -e

echo "==== OASIS3 PIPELINE START ===="

# -------------------------
# 1. Build OASIS3 TEST Manifest (CSV + IMAGE FOLDER)
# -------------------------
echo "Building OASIS3 TEST manifest..."

python3 -m code.ingest.build_manifest_oasis \
  --image_root data/raw/oasis3/t1_mprage_sag/t1_mprage_sag_images \
  --csv_path data/raw/oasis3/t1_mprage_sag/t1_mprage_sag_12_07_2025.csv \
  --out_csv data/manifests/oasis3_manifest_test.csv


# -------------------------
# 2. Preprocess Images
# -------------------------
echo "Running preprocessing..."

python3 -m code.preprocess.monai_preproc \
  --manifest data/manifests/oasis3_manifest.csv \
  --out_dir data/derivatives/preprocess/oasis3

# -------------------------
# 3. Run BSC Batch (Atropos)
# -------------------------
echo "Running BSC (Atropos)..."

python3 -m code.pipeline.run_batch \
  --manifest data/manifests/oasis3_manifest.csv \
  --engine atropos \
  --out_root data/derivatives/bsc/oasis3/atropos

echo "==== OASIS3 PIPELINE COMPLETE ===="
