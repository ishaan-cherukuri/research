#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=oasis_bsc_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# Whole-node preprocess + atropos-BSC + T1 feature extraction over the OASIS manifest,
# reusing every script from the ADNI pipeline unmodified (all manifest-driven).
#
# Usage: sbatch run_oasis_pipeline.sh <PROJECT_DIR> <SCRATCH_DERIV_DIR> <MANIFEST_CSV>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_oasis_pipeline.sh PROJECT_DIR SCRATCH_DERIV_DIR MANIFEST_CSV}"
SCRATCH_DERIV_DIR="${2:?}"
MANIFEST_CSV="${3:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/mri-bsc:${PYTHONPATH:-}"

PREPROC_ROOT="${SCRATCH_DERIV_DIR}/preprocess"
BSC_ROOT="${SCRATCH_DERIV_DIR}/bsc"
mkdir -p "$PREPROC_ROOT" "$BSC_ROOT"

N_JOBS="${SLURM_CPUS_ON_NODE:-192}"
TOTAL=$(($(wc -l < "$MANIFEST_CSV") - 1))
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

echo "manifest rows: $TOTAL, workers: $N_JOBS, chunk size: $CHUNK"

process_chunk() {
    local skip="$1"
    python3 "${PROJECT_DIR}/preprocess_local.py" \
        --manifest "$MANIFEST_CSV" --out_root "$PREPROC_ROOT" \
        --skip "$skip" --limit "$CHUNK"
    python3 "${PROJECT_DIR}/run_bsc_batch.py" \
        --manifest "$MANIFEST_CSV" \
        --preproc_root "$PREPROC_ROOT" --out_root "$BSC_ROOT" \
        --skip "$skip" --limit "$CHUNK"
}
export -f process_chunk
export MANIFEST_CSV PREPROC_ROOT BSC_ROOT PROJECT_DIR CHUNK

seq 0 "$CHUNK" "$TOTAL" | parallel -j "$N_JOBS" process_chunk {}

echo "[DONE] preprocess + BSC segmentation complete"

python3 "${PROJECT_DIR}/mri-bsc/code/features/extract_bsc_features.py" \
    --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
    --out_csv "${SCRATCH_DERIV_DIR}/bsc_simple_features.csv" --min_visits_per_subject 1

python3 "${PROJECT_DIR}/mri-bsc/code/features/extract_bsc_slopes.py" \
    --features "${SCRATCH_DERIV_DIR}/bsc_simple_features.csv" --manifest "$MANIFEST_CSV" \
    --out_csv "${SCRATCH_DERIV_DIR}/bsc_longitudinal_slopes.csv" --min_visits 4

echo "[DONE] BSC feature aggregation complete"

extract_t1_chunk() {
    local skip="$1"
    python3 "${PROJECT_DIR}/extract_t1_scan_features.py" \
        --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
        --out_csv "${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_${skip}.csv" \
        --skip "$skip" --limit "$CHUNK"
}
export -f extract_t1_chunk
mkdir -p "${SCRATCH_DERIV_DIR}/t1_scan_features_parts"
seq 0 "$CHUNK" "$TOTAL" | parallel -j "$N_JOBS" extract_t1_chunk {}

python3 -c "
import pandas as pd, glob
parts = sorted(glob.glob('${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_*.csv'))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv('${SCRATCH_DERIV_DIR}/t1_scan_features.csv', index=False)
print('merged', len(df), 'rows from', len(parts), 'parts')
"

python3 "${PROJECT_DIR}/build_t1_subject_features.py" \
    --scan_features "${SCRATCH_DERIV_DIR}/t1_scan_features.csv" \
    --survival "${PROJECT_DIR}/survival_labels_oasis.csv" \
    --out_csv "${SCRATCH_DERIV_DIR}/features_t1_all.csv"

echo "[DONE] OASIS pipeline complete -- copy ${SCRATCH_DERIV_DIR}/{bsc_longitudinal_slopes,features_t1_all}.csv to \$PROJECT from a login node"
