#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=bsc_sigmoid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# Everything downstream of CIVET: build the ten sampling surfaces, sample T1w
# intensity on them, fit the sigmoid per vertex, smooth, and resample.
#
# Cheap next to CIVET. The sigmoid fit is ~82 s per scan for both hemispheres
# (measured), so roughly 100 core-hours for the cohort; the surface algebra and
# MINC calls dominate wall time instead.
#
# Requires run_civet.sh to have finished. Every stage skips scans it has already
# done, so this is safe to resubmit.
#
# Usage: sbatch run_sigmoid_bsc.sh <PROJECT_DIR> <SCRATCH_DIR>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_sigmoid_bsc.sh PROJECT_DIR SCRATCH_DIR}"
SCRATCH_DIR="${2:?}"
PREFIX="${PREFIX:-adni}"

module load cobralab

source "${PROJECT_DIR}/.venv/bin/activate"

BSC_DIR="${PROJECT_DIR}/bsc_sigmoid"
CIVET_OUT="${SCRATCH_DIR}/civet_out"
MODEL_DIR="${CIVET_MODEL_DIR:-${QUARANTINE_PATH}/CIVET/2.1.0/build/CIVET-2.1.0/models/icbm}"

if [ ! -f "${MODEL_DIR}/icbm_avg_mid_sym_mc_left.obj" ]; then
    echo "[FATAL] ICBM model surfaces not found under ${MODEL_DIR}"
    echo "        set CIVET_MODEL_DIR to the CIVET models/icbm directory"
    exit 1
fi

# Built once here rather than per worker; the vendored binary in the reference
# repo was compiled elsewhere and will not necessarily run on this node.
if [ ! -x "${BSC_DIR}/methods/move_surface_along_flipped_vector" ]; then
    echo "[INFO] compiling move_surface_along_flipped_vector"
    gcc -o "${BSC_DIR}/methods/move_surface_along_flipped_vector" \
        "${BSC_DIR}/methods/move_surface_along_flipped_vector.c" \
        $(pkg-config --cflags --libs bicpl minc2) -lm
fi

N_JOBS="${SLURM_CPUS_ON_NODE:-192}"

process_one() {
    local cid="$1"
    python3 "${BSC_DIR}/run_surfaces.py" \
        --civet_root "$CIVET_OUT" --work_root "$SCRATCH_DIR" \
        --civet_id "$cid" --prefix "$PREFIX" || { echo "[FAIL surf] $cid"; return 0; }

    for hemi in left right; do
        python3 "${BSC_DIR}/fit_sigmoid.py" \
            --samples_dir "${SCRATCH_DIR}/samples" \
            --image_id "$cid" --hemi "$hemi" \
            --out_dir "${SCRATCH_DIR}/sigmoid_fit/unsmoothed" \
            || { echo "[FAIL fit] $cid $hemi"; return 0; }
    done

    python3 "${BSC_DIR}/postprocess_vertex.py" \
        --work_root "$SCRATCH_DIR" --civet_root "$CIVET_OUT" \
        --civet_id "$cid" --prefix "$PREFIX" --model_dir "$MODEL_DIR" \
        || echo "[FAIL post] $cid"
}
export -f process_one
export BSC_DIR CIVET_OUT SCRATCH_DIR PREFIX MODEL_DIR

# Only scans CIVET actually finished.
find "$CIVET_OUT" -name "*_mid_surface_right_81920.obj" \
    | sed -E 's|.*/(scan[0-9]+)/surfaces/.*|\1|' | sort -u \
    > "${SCRATCH_DIR}/civet_done_ids.txt"
echo "[INFO] $(wc -l < "${SCRATCH_DIR}/civet_done_ids.txt") scans with CIVET output"

parallel -j "$N_JOBS" process_one :::: "${SCRATCH_DIR}/civet_done_ids.txt"

echo "[DONE] per-scan stages complete"
echo "[NEXT] on a login node, run gather_curvature.py then extract_sigmoid_features.py"
