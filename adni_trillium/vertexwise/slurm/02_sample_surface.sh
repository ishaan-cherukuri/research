#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_sample
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log

# For each scan with a finished recon-all: recompute the dense gBSC field
# (compute_gbsc_field.py; the stored bsc_dir_map is masked to a sparse band
# that cannot be sampled on a surface), evaluate it at the FreeSurfer white
# surface averaged over 0 to 1.5 mm outward along the normal (where the field
# peaks), and carry cortical thickness and mean curvature, all onto fsaverage6
# with FWHM smoothing on the target surface. One .mgh per measure per
# hemisphere per scan.
#
# MEASURE is positional, not an environment variable: Trillium submits with
# --export=NONE --get-user-env, so exported variables do not reach the job.
# "cos" keeps its outputs in surf_cos/ and field_cos/ beside the gradient run.
#
# Usage: sbatch 02_sample_surface.sh <VW_DIR> <WORK> [FWHM] [MEASURE]
#   reads WORK/freesurfer, writes WORK/surf/<image_id>/; FWHM in mm (default 10)

set -euo pipefail

VW_DIR="${1:?Usage: sbatch 02_sample_surface.sh VW_DIR WORK [FWHM]}"
WORK="${2:?}"
export FWHM="${3:-10}"
export MEASURE="${4:-dir}"            # dir = |grad I . n|; cos = that over |grad I|
SUFFIX=""; [[ "$MEASURE" == "cos" ]] && SUFFIX="_cos"
export PROJ="${PROJ:---projdist-avg 0 1.5 0.5}"
export PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"
export VW_DIR
SCANS_CSV="$VW_DIR/cohort/vertex_scans.csv"
SUBJECTS_DIR_WANT="$WORK/freesurfer"
export SURF_OUT="$WORK/surf${SUFFIX}"
export FIELD_OUT="$WORK/field${SUFFIX}"
export TRG=fsaverage6

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$SUBJECTS_DIR_WANT"   # FreeSurferEnv.sh resets it, so set it afterwards
module load cobralab 2>/dev/null || true
export OMP_NUM_THREADS=1

# fsaverage6 has to be visible inside SUBJECTS_DIR for --trgsubject
[[ -e "$SUBJECTS_DIR/$TRG" ]] || ln -s "$FREESURFER_HOME/subjects/$TRG" "$SUBJECTS_DIR/$TRG"
export LOG_DIR="$WORK/logs/sample"
mkdir -p "$SURF_OUT" "$FIELD_OUT" "$LOG_DIR"

sample_one() {
    local id="$1" bsc_dir="$2"
    local out="$SURF_OUT/$id" log="$LOG_DIR/${id}.log"
    if [[ ! -f "$SUBJECTS_DIR/$id/scripts/recon-all.done" ]]; then
        echo "[NO-RECON] $id"; return 0
    fi
    if [[ ! -f "$bsc_dir/gm_prob.nii.gz" ]]; then echo "[NO-BSC] $id"; return 0; fi
    mkdir -p "$out"
    : > "$log"
    local mov="$FIELD_OUT/${id}.nii.gz"
    if [[ ! -f "$mov" ]]; then
        $PY "$VW_DIR/compute_gbsc_field.py" "$bsc_dir" "$mov" --measure "$MEASURE" >> "$log" 2>&1 \
            || { echo "[FAIL-field] $id"; return 0; }
    fi
    for hemi in lh rh; do
        if [[ -f "$out/${hemi}.curv.${TRG}.mgh" ]]; then continue; fi
        # gBSC field -> native white surface, averaged 0..1.5 mm outward, unsmoothed
        mri_vol2surf --mov "$mov" --regheader "$id" --hemi "$hemi" \
            --surf white $PROJ --interp trilinear \
            --o "$out/${hemi}.gbsc.native.mgh" >> "$log" 2>&1 || { echo "[FAIL-vol2surf] $id $hemi"; return 0; }
        # all three measures: native -> fsaverage6 with the same target-surface smoothing
        for meas in gbsc thickness curv; do
            if [[ "$meas" == gbsc ]]; then
                sval="$out/${hemi}.gbsc.native.mgh"; fmt=""
            else
                sval="$SUBJECTS_DIR/$id/surf/${hemi}.${meas}"; fmt="--sfmt curv"
            fi
            mri_surf2surf --srcsubject "$id" --trgsubject "$TRG" --hemi "$hemi" \
                --sval "$sval" $fmt --fwhm-trg "$FWHM" \
                --tval "$out/${hemi}.${meas}.${TRG}.mgh" >> "$log" 2>&1 \
                || { echo "[FAIL-surf2surf] $id $hemi $meas"; return 0; }
        done
    done
    echo "[OK] $id"
}
export -f sample_one

tail -n +2 "$SCANS_CSV" | awk -F, '{print $1"\t"$4}' \
    | parallel -j "${SLURM_CPUS_ON_NODE:-192}" --colsep '\t' sample_one {1} {2}

echo "[DONE] sampling"
