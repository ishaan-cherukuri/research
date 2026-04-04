#!/usr/bin/env bash
set -e

echo "==== ADNI BSC LONGITUDINAL SLOPES PIPELINE ===="

  
python3 -m code.ml.train_rsf_with_slopes \
  --slopes data/index/bsc_longitudinal_slopes.csv \
  --survival data/ml/survival/time_to_conversion.csv \
  --out_dir data/ml/results/rsf \
  --top_k 20 \
  --penalize_regex nboundary \
  --penalty_factor 0.1

echo "==== PIPELINE COMPLETE ===="

