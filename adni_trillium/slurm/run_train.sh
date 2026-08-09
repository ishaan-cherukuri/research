#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=adni_bsc_train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log

# XGBoost survival training -- cheap relative to the segmentation stage, but exceeds the
# login node's CPU time limit, so run it as a real (if tiny) job.
#
# Usage: sbatch run_train.sh <PROJECT_DIR> <SCRATCH_DIR>
# $PROJECT is read-only on compute nodes -- write results to $SCRATCH_DIR, copy to
# $PROJECT from a login node after the job finishes.

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_train.sh PROJECT_DIR SCRATCH_DIR}"
SCRATCH_DIR="${2:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"

python3 "${PROJECT_DIR}/train_xgb_bsc_survival.py" \
    --slopes "${PROJECT_DIR}/bsc_longitudinal_slopes.csv" \
    --survival "${PROJECT_DIR}/survival_labels.csv" \
    --out_dir "${SCRATCH_DIR}/results"

echo "[DONE] training complete -- results in ${SCRATCH_DIR}/results, copy to \$PROJECT from a login node"
