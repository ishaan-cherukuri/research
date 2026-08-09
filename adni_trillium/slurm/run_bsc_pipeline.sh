#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=adni_bsc_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# Whole-node preprocess + atropos-BSC run over the full manifest, fanned out across all
# cores on the node via GNU Parallel (Trillium is whole-node-only scheduling: 192 cores).
#
# Usage: sbatch run_bsc_pipeline.sh <PROJECT_DIR> <SCRATCH_DERIV_DIR> <MANIFEST_CSV>
#   PROJECT_DIR:       e.g. /project/rrg-mchakrav-ab/ishaan/adni_trillium
#   SCRATCH_DERIV_DIR: e.g. $SCRATCH/adni_trillium/derivatives
#   MANIFEST_CSV:      e.g. $PROJECT_DIR/manifest.csv

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_bsc_pipeline.sh PROJECT_DIR SCRATCH_DERIV_DIR MANIFEST_CSV}"
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

echo "[DONE] pipeline complete"

# Feature aggregation runs single-threaded at the end (cheap relative to the segmentation step).
# $PROJECT is read-only on compute nodes -- write to $SCRATCH_DERIV_DIR here, copy final CSVs
# to $PROJECT from a login node after the job finishes.
python3 "${PROJECT_DIR}/mri-bsc/code/features/extract_bsc_features.py" \
    --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
    --out_csv "${SCRATCH_DERIV_DIR}/bsc_simple_features.csv" --min_visits_per_subject 1

python3 "${PROJECT_DIR}/mri-bsc/code/features/extract_bsc_slopes.py" \
    --features "${SCRATCH_DERIV_DIR}/bsc_simple_features.csv" --manifest "$MANIFEST_CSV" \
    --out_csv "${SCRATCH_DERIV_DIR}/bsc_longitudinal_slopes.csv" --min_visits 4

echo "[DONE] feature aggregation complete -- outputs in ${SCRATCH_DERIV_DIR}, copy to \$PROJECT from a login node"
