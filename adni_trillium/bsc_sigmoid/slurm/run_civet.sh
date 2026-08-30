#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=bsc_sigmoid_civet
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

# CIVET 2.1.0 over the spec_v3 manifest, fanned out across the whole node with
# GNU Parallel (Trillium schedules whole nodes: 192 cores).
#
# This is the expensive stage by a wide margin: CIVET is roughly 6-12 core-hours
# per scan, so 4309 scans is 26k-52k core-hours, or 140-270 whole-node hours.
# It will not finish in one job. The walltime is 24h rather than the 6h used by
# the Atropos pipeline because a single CIVET run can exceed 6h on its own, and
# a scan killed mid-run leaves a partial output directory. Resubmit until
# run_civet.sh reports nothing left to do; --skip-if-done makes reruns cheap.
#
# Usage: sbatch run_civet.sh <PROJECT_DIR> <SCRATCH_DIR> <MANIFEST_CSV>
#   PROJECT_DIR: e.g. /project/rrg-mchakrav-ab/ishaan/adni_trillium
#   SCRATCH_DIR: e.g. $SCRATCH/bsc_sigmoid
#   MANIFEST_CSV: e.g. $PROJECT_DIR/from_cluster/manifest_v3.csv

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_civet.sh PROJECT_DIR SCRATCH_DIR MANIFEST_CSV}"
SCRATCH_DIR="${2:?}"
MANIFEST_CSV="${3:?}"
PREFIX="${PREFIX:-adni}"

module load cobralab

source "${PROJECT_DIR}/.venv/bin/activate"

SRC_DIR="${SCRATCH_DIR}/civet_input"
CIVET_OUT="${SCRATCH_DIR}/civet_out"
mkdir -p "$SRC_DIR" "$CIVET_OUT"

# Stage NIfTI -> MINC and assign each scan a CIVET-safe id. Idempotent, so this
# is cheap on resubmission.
python3 "${PROJECT_DIR}/bsc_sigmoid/prepare_civet_inputs.py" \
    --manifest "$MANIFEST_CSV" --out_root "$SCRATCH_DIR" --prefix "$PREFIX"

N_JOBS="${SLURM_CPUS_ON_NODE:-192}"

# One CIVET invocation per scan. -run without a queue keeps it in-process so
# GNU Parallel controls concurrency rather than CIVET's own scheduler, which
# expects a cluster queue this node does not expose.
run_one() {
    local cid="$1"
    if [ -f "${CIVET_OUT}/${cid}/surfaces/${PREFIX}_${cid}_mid_surface_right_81920.obj" ]; then
        return 0
    fi
    CIVET_Processing_Pipeline \
        -sourcedir "$SRC_DIR" \
        -targetdir "$CIVET_OUT" \
        -prefix "$PREFIX" \
        -N3-distance 75 \
        -lsq12 \
        -thickness tlink 20 \
        -resample-surfaces \
        -area-fwhm 20 -volume-fwhm 20 \
        -spawn -no-queue \
        -run "$cid" \
        > "${CIVET_OUT}/${cid}.log" 2>&1 || echo "[FAIL] $cid"
}
export -f run_one
export SRC_DIR CIVET_OUT PREFIX

parallel -j "$N_JOBS" run_one :::: "${SCRATCH_DIR}/civet_ids.txt"

DONE=$(find "$CIVET_OUT" -name "*_mid_surface_right_81920.obj" | wc -l)
TOTAL=$(wc -l < "${SCRATCH_DIR}/civet_ids.txt")
echo "[STATUS] CIVET complete for ${DONE}/${TOTAL} scans"
if [ "$DONE" -lt "$TOTAL" ]; then
    echo "[STATUS] resubmit this script to continue"
fi
