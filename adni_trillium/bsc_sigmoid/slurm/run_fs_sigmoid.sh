#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=fs_sigmoid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# Olafson's sigmoid BSC fitted per native FreeSurfer vertex, for every scan in
# the vertex-wise cohort. Roughly 10 minutes per scan single-threaded, so about
# 375 core-hours over 2,251 scans; the CIVET route the companion pipeline
# assumes is budgeted at 26,000 to 52,000.
#
# Arguments are positional, not environment: Trillium submits with
# --export=NONE --get-user-env, so exported variables never reach the job.
#
# Usage: sbatch run_fs_sigmoid.sh <PROJECT_DIR> <WORK> <SCANS_CSV> <OUT_DIR> [WORKERS]

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_fs_sigmoid.sh PROJECT_DIR WORK SCANS_CSV OUT_DIR [WORKERS]}"
WORK="${2:?}"
SCANS_CSV="${3:?}"
OUT_DIR="${4:?}"
N_JOBS="${5:-150}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/bsc_sigmoid:${PYTHONPATH:-}"
# scipy/numpy would otherwise each grab the whole node per worker
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT_DIR" "$WORK/logs/sigmoid"
export SUBJECTS_DIR_ARG="$WORK/freesurfer"

fit_one() {
    local id="$1"
    python3 "${PROJECT_DIR}/bsc_sigmoid/fs_sigmoid_bsc.py" \
        --subjects_dir "$SUBJECTS_DIR_ARG" --image_id "$id" --out_dir "$OUT_DIR" \
        >> "$WORK/logs/sigmoid/${id}.log" 2>&1 \
        && echo "[OK] $id" || echo "[FAIL] $id"
}
export -f fit_one
# WORK is used inside fit_one for the per-scan log path; GNU Parallel runs it
# in a fresh shell, so every variable it reads has to be exported here.
export PROJECT_DIR OUT_DIR WORK

tail -n +2 "$SCANS_CSV" | cut -d, -f1 \
    | parallel -j "$N_JOBS" --joblog "$WORK/logs/sigmoid.joblog" fit_one {}

echo "[DONE] fitted $(ls "$OUT_DIR" | wc -l) scans"
