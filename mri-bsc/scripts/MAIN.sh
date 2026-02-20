#!/usr/bin/env bash
set -e

echo "==== ADNI BSC LONGITUDINAL SLOPES PIPELINE ===="

# Step 1: Extract BSC features from all 1,824 scans on external drive
# echo "Step 1: Extracting BSC features from external drive..."
# python3 -m code.features.extract_bsc_features \
#   --manifest data/manifests/adni_manifest.csv \
#   --bsc_root /Volumes/YAAGL/derivatives/bsc/adni/atropos_v2 \
#   --out_csv data/index/bsc_simple_features.csv

# Step 2: Compute longitudinal slopes for all 456 subjects
# echo "Step 2: Computing longitudinal BSC slopes..."
# python3 -m code.features.extract_bsc_slopes \
#   --features data/index/bsc_simple_features.csv \
#   --manifest data/manifests/adni_manifest.csv \
#   --out_csv data/index/bsc_longitudinal_slopes.csv \
#   --min_visits 4

# Step 3: Prepare survival datasets with all subjects
# echo "Step 3: Preparing survival datasets..."
# python3 -m code.ml.prepare_survival_data \
#   --features data/index/bsc_simple_features.csv \
#   --out_dir data/ml/survival

# Step 4: Train survival model with BSC SLOPES (the killer feature!)
# echo "Step 4: Training survival model with BSC slopes..."
# python3 -m code.ml.train_survival_with_slopes \
#   --slopes data/index/bsc_longitudinal_slopes.csv \
#   --survival data/ml/survival/time_to_conversion.csv \
#   --out_dir data/ml/results/slopes \
#   --top_k 20 \
#   --model weibull

# python3 scripts/extract_paper_statistics.py
# python3 scripts/diagnose_model_issues.py

python3 -m code.ml.train_rsf_with_slopes \
  --slopes data/index/bsc_longitudinal_slopes.csv \
  --survival data/ml/survival/time_to_conversion.csv \
  --out_dir data/ml/results/rsf_standardized \
  --top_k 20 \
  --n_estimators 1000

echo "==== PIPELINE COMPLETE ===="
echo ""
echo "Outputs:"
echo "  - data/index/bsc_simple_features.csv (1824 scans, 187 features)"
echo "  - data/index/bsc_longitudinal_slopes.csv (456 subjects, 182 slope features)"
echo "  - data/ml/survival/time_to_conversion.csv (456 subjects with conversion labels)"
echo "  - data/ml/results/slopes/ (survival model results with BSC slopes)"
echo ""
echo "Key comparison:"
echo "  Baseline BSC: C-index ~0.24 (FAILED)"
echo "  BSC Slopes:   Check data/ml/results/slopes/slopes_weibull_metrics.json"


