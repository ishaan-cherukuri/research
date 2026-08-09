#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=adni_bsc_debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log

# Small end-to-end smoke test (~10 subjects, serial) before committing to the full whole-node
# run in run_bsc_pipeline.sh. Verifies the chain works on Trillium's actual environment
# (modules, FreeSurfer/ANTs availability, filesystem paths) before scaling up.
#
# Usage: sbatch run_debug_small.sh <PROJECT_DIR> <SCRATCH_DERIV_DIR> <MANIFEST_CSV>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_debug_small.sh PROJECT_DIR SCRATCH_DERIV_DIR MANIFEST_CSV}"
SCRATCH_DERIV_DIR="${2:?}"
MANIFEST_CSV="${3:?}"

module load cobralab

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/mri-bsc:${PYTHONPATH:-}"

PREPROC_ROOT="${SCRATCH_DERIV_DIR}/preprocess_debug"
BSC_ROOT="${SCRATCH_DERIV_DIR}/bsc_debug"
mkdir -p "$PREPROC_ROOT" "$BSC_ROOT"

python3 "${PROJECT_DIR}/preprocess_local.py" \
    --manifest "$MANIFEST_CSV" --out_root "$PREPROC_ROOT" --limit 10

python3 "${PROJECT_DIR}/run_bsc_batch.py" \
    --manifest "$MANIFEST_CSV" \
    --preproc_root "$PREPROC_ROOT" --out_root "$BSC_ROOT" --limit 10

python3 "${PROJECT_DIR}/mri-bsc/code/features/extract_bsc_features.py" \
    --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
    --out_csv "${SCRATCH_DERIV_DIR}/bsc_simple_features_debug.csv" --min_visits_per_subject 1

echo "[DONE] debug run complete -- inspect ${BSC_ROOT} and ${SCRATCH_DERIV_DIR}/bsc_simple_features_debug.csv by hand"
echo "[NOTE] \$PROJECT is read-only on compute nodes -- copy final results from a login node after the job finishes"
