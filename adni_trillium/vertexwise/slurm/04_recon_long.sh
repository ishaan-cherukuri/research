#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.log

# FreeSurfer's longitudinal stream on top of the cross-sectional runs 01 made.
#
# Why it matters here. The analysis is a slope, and a slope is a difference of
# measurements, so it inherits the measurement noise of every timepoint twice
# over. Running recon-all independently per scan gives each timepoint its own
# segmentation and surface placement, and the disagreement between those
# independent solutions goes straight into the slope. The longitudinal stream
# initialises every timepoint from one within-subject template, which removes
# most of that term. If gBSC carries a real longitudinal signal in this cohort,
# this is the condition under which it is visible; if the null survives here,
# it is not attributable to slope noise.
#
# Two phases, in order, because -long needs the base:
#   base   recon-all -base base_<subject> -tp <scan> ... -all     (one per subject)
#   long   recon-all -long <scan> base_<subject> -all             (one per scan)
# The long outputs land in $SUBJECTS_DIR/<scan>.long.base_<subject>, so the
# cross-sectional runs are left untouched and both can be sampled and compared.
#
# Budget, measured against the cross-sectional run: base is a full recon at
# roughly 8 h single-threaded, long roughly 3 h, so about 3,300 + 6,800 =
# 10,000 core-hours over 417 subjects and 2,251 scans. Submit base first,
# check the real timings in the joblog, then submit long.
#
# Re-running either phase is safe: finished stages are skipped.
#
# Usage: sbatch --array=0-2  04_recon_long.sh <VW_DIR> <WORK> base
#        sbatch --array=0-14 04_recon_long.sh <VW_DIR> <WORK> long

set -euo pipefail

VW_DIR="${1:?Usage: sbatch 04_recon_long.sh VW_DIR WORK {base|long}}"
WORK="${2:?}"
PHASE="${3:?base or long}"
SCANS_CSV="$VW_DIR/cohort/vertex_scans.csv"
SUBJECTS_DIR_WANT="$WORK/freesurfer"
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$SUBJECTS_DIR_WANT"   # FreeSurferEnv.sh resets it, so set it afterwards
module load cobralab 2>/dev/null || true   # GNU parallel
export OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

LOG_DIR="$WORK/logs/recon_$PHASE"
mkdir -p "$SUBJECTS_DIR" "$LOG_DIR"
N_JOBS="${RECON_PER_NODE:-140}"   # the base stage holds two volumes in memory, so fewer than 01's 160

# Only scans whose cross-sectional recon finished can enter the longitudinal
# stream; one scan (127_S_4197_m48) failed Talairach and is absent here.
WORKLIST="$WORK/logs/long_${PHASE}_worklist.txt"
if [[ "$TASK" == "0" || ! -s "$WORKLIST" ]]; then
    tail -n +2 "$SCANS_CSV" | awk -F, '{print $2"\t"$1}' | sort \
        | while IFS=$'\t' read -r subj id; do
              [[ -f "$SUBJECTS_DIR/$id/scripts/recon-all.done" ]] && echo -e "$subj\t$id"
          done > "${WORKLIST}.tmp"
    if [[ "$PHASE" == "base" ]]; then
        # one row per subject: subject then every timepoint, tab separated
        awk -F'\t' '{a[$1]=a[$1]"\t"$2} END {for (s in a) print s a[s]}' \
            "${WORKLIST}.tmp" | sort > "$WORKLIST"
    else
        sort "${WORKLIST}.tmp" > "$WORKLIST"
    fi
    rm -f "${WORKLIST}.tmp"
fi
# array tasks beyond 0 may start before task 0 has written the list
for _ in $(seq 1 60); do [[ -s "$WORKLIST" ]] && break; sleep 10; done

TOTAL=$(wc -l < "$WORKLIST")
CHUNK=$(( (TOTAL + N_TASKS - 1) / N_TASKS ))
START=$(( TASK * CHUNK ))
echo "phase $PHASE, task $TASK/$N_TASKS: rows $START..$((START + CHUNK - 1)) of $TOTAL, $N_JOBS workers"

base_one() {
    # subject, then one argument per timepoint
    local subj="$1"; shift
    local base="base_${subj}"
    local log="$LOG_DIR/${base}.log"
    if [[ -f "$SUBJECTS_DIR/$base/scripts/recon-all.done" ]]; then
        echo "[DONE-SKIP] $base"; return 0
    fi
    local tps=()
    for id in "$@"; do tps+=(-tp "$id"); done
    rm -f "$SUBJECTS_DIR/$base/scripts/IsRunning."*
    local t0=$SECONDS
    recon-all -base "$base" "${tps[@]}" -all -sd "$SUBJECTS_DIR" > "$log" 2>&1 && rc=0 || rc=$?
    echo "[$( [[ $rc -eq 0 ]] && echo OK || echo FAIL )] $base rc=$rc ${#} tp $(( (SECONDS - t0) / 60 )) min"
}

long_one() {
    local subj="$1" id="$2"
    local base="base_${subj}"
    local out="${id}.long.${base}"
    local log="$LOG_DIR/${out}.log"
    if [[ -f "$SUBJECTS_DIR/$out/scripts/recon-all.done" ]]; then
        echo "[DONE-SKIP] $out"; return 0
    fi
    if [[ ! -f "$SUBJECTS_DIR/$base/scripts/recon-all.done" ]]; then
        echo "[NO-BASE] $out"; return 0
    fi
    rm -f "$SUBJECTS_DIR/$out/scripts/IsRunning."*
    local t0=$SECONDS
    recon-all -long "$id" "$base" -all -sd "$SUBJECTS_DIR" > "$log" 2>&1 && rc=0 || rc=$?
    echo "[$( [[ $rc -eq 0 ]] && echo OK || echo FAIL )] $out rc=$rc $(( (SECONDS - t0) / 60 )) min"
}
export -f base_one long_one
export SUBJECTS_DIR LOG_DIR

JOBLOG="$WORK/logs/long_${PHASE}_task${TASK}.joblog"
sed -n "$((START + 1)),$((START + CHUNK))p" "$WORKLIST" \
    | if [[ "$PHASE" == "base" ]]; then
          parallel -j "$N_JOBS" --joblog "$JOBLOG" --colsep '\t' base_one
      else
          parallel -j "$N_JOBS" --joblog "$JOBLOG" --colsep '\t' long_one {1} {2}
      fi

echo "[DONE] phase $PHASE task $TASK"
