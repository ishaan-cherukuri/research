#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=oasis_external_eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log

set -euo pipefail
PROJECT_DIR="${1:?Usage: sbatch run_external_eval.sh PROJECT_DIR SCRATCH_DIR}"
SCRATCH_DIR="${2:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"

# $PROJECT is read-only on compute nodes -- write to $SCRATCH_DIR, copy from a login node.
python3 "${PROJECT_DIR}/eval_frozen_model_external.py" \
    --adni_slopes "${PROJECT_DIR}/bsc_longitudinal_slopes.csv" \
    --adni_survival "${PROJECT_DIR}/survival_labels.csv" \
    --adni_t1 "${PROJECT_DIR}/features_t1_all.csv" \
    --model "${PROJECT_DIR}/results_combined/xgb_model.json" \
    --ext_slopes "${PROJECT_DIR}/bsc_longitudinal_slopes_oasis.csv" \
    --ext_survival "${PROJECT_DIR}/survival_labels_oasis.csv" \
    --ext_t1 "${PROJECT_DIR}/features_t1_all_oasis.csv" \
    --out_dir "${SCRATCH_DIR}/results_external_oasis"

echo "[DONE] external eval complete -- copy ${SCRATCH_DIR}/results_external_oasis to \$PROJECT from a login node"
