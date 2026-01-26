#!/usr/bin/env bash
set -e

echo "==== ADNI BSC LONGITUDINAL SLOPES PIPELINE ===="

# Step 1: Extract BSC features from all 1,824 scans on external drive
echo "Step 1: Extracting BSC features from external drive..."
python3 -m code.features.extract_bsc_features \
  --manifest /Users/ishu/Downloads/adni_manifest.csv \
  --bsc_root /Volumes/YAAGL/derivatives/bsc/adni/atropos_v2 \
  --out_csv data/index/bsc_simple_features.csv

# Step 2: Compute longitudinal slopes for all 456 subjects
echo "Step 2: Computing longitudinal BSC slopes..."
python3 -m code.features.extract_bsc_slopes \
  --features data/index/bsc_simple_features.csv \
  --manifest /Users/ishu/Downloads/adni_manifest.csv \
  --out_csv data/index/bsc_longitudinal_slopes.csv \
  --min_visits 4

echo "==== PIPELINE COMPLETE ===="
echo ""
echo "Outputs:"
echo "  - data/index/bsc_simple_features.csv (1824 scans, 187 features)"
echo "  - data/index/bsc_longitudinal_slopes.csv (456 subjects, 182 slope features)"
echo ""
echo "Next steps:"
echo "  1. Train classification model: python3 -m code.ml.train_diagnosis_classifier"
echo "  2. Train survival with slopes: Use bsc_longitudinal_slopes.csv as features"

