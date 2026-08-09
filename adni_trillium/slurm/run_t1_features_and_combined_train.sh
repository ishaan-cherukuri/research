#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=adni_t1_combined
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log

# T1 morphometry/QC feature extraction (from already-computed preprocess/BSC outputs, pure
# I/O + numpy, no new segmentation) + combined BSC+T1 XGBoost survival training, reusing
# mri-bsc's original train_xgb_survival_combined.py unmodified (parametrized via
# --t1_features to point at our newly-built file instead of raw_t1_analysis/).
#
# Usage: sbatch run_t1_features_and_combined_train.sh <PROJECT_DIR> <SCRATCH_DERIV_DIR> <MANIFEST_CSV>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_t1_features_and_combined_train.sh PROJECT_DIR SCRATCH_DERIV_DIR MANIFEST_CSV}"
SCRATCH_DERIV_DIR="${2:?}"
MANIFEST_CSV="${3:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/mri-bsc:${PYTHONPATH:-}"

PREPROC_ROOT="${SCRATCH_DERIV_DIR}/preprocess"
BSC_ROOT="${SCRATCH_DERIV_DIR}/bsc"

N_JOBS="${SLURM_CPUS_ON_NODE:-192}"
TOTAL=$(($(wc -l < "$MANIFEST_CSV") - 1))
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

echo "manifest rows: $TOTAL, workers: $N_JOBS, chunk size: $CHUNK"

extract_chunk() {
    local skip="$1"
    python3 "${PROJECT_DIR}/extract_t1_scan_features.py" \
        --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
        --out_csv "${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_${skip}.csv" \
        --skip "$skip" --limit "$CHUNK"
}
export -f extract_chunk
export MANIFEST_CSV PREPROC_ROOT BSC_ROOT SCRATCH_DERIV_DIR PROJECT_DIR CHUNK

mkdir -p "${SCRATCH_DERIV_DIR}/t1_scan_features_parts"
seq 0 "$CHUNK" "$TOTAL" | parallel -j "$N_JOBS" extract_chunk {}

python3 -c "
import pandas as pd, glob
parts = sorted(glob.glob('${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_*.csv'))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv('${SCRATCH_DERIV_DIR}/t1_scan_features.csv', index=False)
print('merged', len(df), 'rows from', len(parts), 'parts')
"

echo "[DONE] T1 scan feature extraction complete"

python3 "${PROJECT_DIR}/build_t1_subject_features.py" \
    --scan_features "${SCRATCH_DERIV_DIR}/t1_scan_features.csv" \
    --survival "${PROJECT_DIR}/survival_labels.csv" \
    --out_csv "${SCRATCH_DERIV_DIR}/features_t1_all.csv"

echo "[DONE] T1 subject feature aggregation complete"

python3 "${PROJECT_DIR}/mri-bsc/code/ml/train_xgb_survival_combined.py" \
    --slopes "${PROJECT_DIR}/bsc_longitudinal_slopes.csv" \
    --survival "${PROJECT_DIR}/survival_labels.csv" \
    --t1_features "${SCRATCH_DERIV_DIR}/features_t1_all.csv" \
    --out_dir "${SCRATCH_DERIV_DIR}/results_combined" \
    --figs_dir "${SCRATCH_DERIV_DIR}/results_combined/figs"

echo "[DONE] combined BSC+T1 training complete -- copy ${SCRATCH_DERIV_DIR}/results_combined and features_t1_all.csv to \$PROJECT from a login node"
