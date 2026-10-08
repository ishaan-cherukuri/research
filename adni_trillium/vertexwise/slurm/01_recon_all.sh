#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --output=%x_%A_%a.log

# Cross-sectional recon-all on every pre-outcome scan, one whole node per array
# task, 160 recon-alls at a time under GNU Parallel. Each array task takes a
# contiguous slice of vertex_scans.csv so that a task holds about one scan per
# core and finishes inside the 24 h wall limit. Re-running the array is safe:
# finished scans are skipped and half-finished ones resume.
#
# /project is read-only on compute nodes, so everything is written under WORK
# on scratch; submit from WORK/logs (run_vertexwise.sh does).
#
# Usage: sbatch --array=0-14 01_recon_all.sh <VW_DIR> <WORK>
#   VW_DIR: pipeline code, e.g. /project/rrg-mchakrav-ab/ishaan/adni_trillium/vertexwise
#   WORK:   e.g. $SCRATCH/adni_trillium/derivatives/vertexwise

set -euo pipefail

VW_DIR="${1:?Usage: sbatch 01_recon_all.sh VW_DIR WORK}"
WORK="${2:?}"
SCANS_CSV="$VW_DIR/cohort/vertex_scans.csv"
SUBJECTS_DIR_WANT="$WORK/freesurfer"
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$SUBJECTS_DIR_WANT"   # FreeSurferEnv.sh resets it, so set it afterwards
module load cobralab 2>/dev/null || true   # GNU parallel
export OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

LOG_DIR="$WORK/logs/recon"
mkdir -p "$SUBJECTS_DIR" "$LOG_DIR"

TOTAL=$(($(wc -l < "$SCANS_CSV") - 1))
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( TASK * CHUNK ))
N_JOBS="${RECON_PER_NODE:-160}"   # below the 192 cores: recon-all peaks near 4 GB and the node has 767 GB
echo "task $TASK/$N_TASKS: rows $START..$((START + CHUNK - 1)) of $TOTAL, $N_JOBS workers"

recon_one() {
    local id="$1" t1="$2"
    local log="$LOG_DIR/${id}.log"
    if [[ -f "$SUBJECTS_DIR/$id/scripts/recon-all.done" ]]; then
        echo "[DONE-SKIP] $id"; return 0
    fi
    if [[ ! -f "$t1" ]]; then
        echo "[MISSING-T1] $id $t1" | tee -a "$log"; return 0
    fi
    rm -f "$SUBJECTS_DIR/$id/scripts/IsRunning."*
    local t0=$SECONDS
    if [[ -d "$SUBJECTS_DIR/$id/mri" ]]; then
        # partially processed on an earlier run; -all without -i resumes in place
        recon-all -all -s "$id" -sd "$SUBJECTS_DIR" > "$log" 2>&1 && rc=0 || rc=$?
    else
        recon-all -all -i "$t1" -s "$id" -sd "$SUBJECTS_DIR" > "$log" 2>&1 && rc=0 || rc=$?
    fi
    echo "[$( [[ $rc -eq 0 ]] && echo OK || echo FAIL )] $id rc=$rc $(( (SECONDS - t0) / 60 )) min"
}
export -f recon_one
export SUBJECTS_DIR LOG_DIR

# image_id,subject,t1_path,... -> pass image_id and t1_path
tail -n +2 "$SCANS_CSV" | sed -n "$((START + 1)),$((START + CHUNK))p" \
    | awk -F, '{print $1"\t"$3}' \
    | parallel -j "$N_JOBS" --colsep '\t' --joblog "$WORK/logs/recon_task${TASK}.joblog" recon_one {1} {2}

echo "[DONE] task $TASK"
