#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_slopes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

# Per-subject annualized slope maps for one measure, plus the slope-precision
# table the weighted variants need. Wraps compute_vertex_slopes.py so the
# measure, the directory its fsaverage6 maps live in, and the normalization can
# vary without editing anything.
#
# Positional, not environment: Trillium submits with --export=NONE.
#
# Usage: sbatch 06_slopes.sh <VW_DIR> <WORK> <OUT_SUBDIR> <MEASURE> <MEASURE_SUBDIR> \
#                           <NORMALIZE> <QC> [SURF_SUBDIR]
#
#   OUT_SUBDIR      e.g. slopes_long, slopes_sigfree
#   MEASURE         gbsc | model_c_free_log | model_c_log | model_ratio
#   MEASURE_SUBDIR  where that measure's maps are, e.g. surf_long, sigmoid_surf
#   NORMALIZE       global (divide by the scan's cortical mean) | demean
#                   (subtract it, for a log-valued measure) | none
#   QC              gbsc | finite | none
#   SURF_SUBDIR     where thickness and curv come from (default surf)
#
# Examples:
#   sbatch 06_slopes.sh "$VW" "$WORK" slopes_long    gbsc             surf_long    global gbsc   surf_long
#   sbatch 06_slopes.sh "$VW" "$WORK" slopes_sigfree model_c_free_log sigmoid_surf demean finite

set -euo pipefail

VW_DIR="${1:?see header for usage}"
WORK="${2:?}"
OUT_SUBDIR="${3:?}"
MEASURE="${4:?}"
MEASURE_SUBDIR="${5:?}"
NORMALIZE="${6:?}"
QC="${7:?}"
SURF_SUBDIR="${8:-surf}"

PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
export OMP_NUM_THREADS=1

echo "== slopes: measure=$MEASURE from $MEASURE_SUBDIR, thickness/curv from $SURF_SUBDIR, "\
"normalize=$NORMALIZE qc=$QC -> $OUT_SUBDIR"

$PY "$VW_DIR/compute_vertex_slopes.py" \
    --scans "$VW_DIR/cohort/vertex_scans.csv" \
    --surf_root "$WORK/$SURF_SUBDIR" \
    --measure "$MEASURE" --measure_root "$WORK/$MEASURE_SUBDIR" \
    --out_root "$WORK/$OUT_SUBDIR" \
    --normalize "$NORMALIZE" --qc "$QC" \
    --fs_subjects_dir "$SUBJECTS_DIR"

echo "[DONE] slopes $OUT_SUBDIR"
