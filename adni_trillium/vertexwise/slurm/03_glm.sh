#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_glm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# Vertex-wise group tests on fsaverage6 with cluster-wise permutation correction.
# Model A: converter vs non-converter gBSC slope, adjusted for age, sex, APOE4, 3T.
# Model B: Model A with thickness slope and curvature slope as per-vertex regressors.
#
# Both models run through vertex_glm.py (Freedman-Lane residual permutation,
# CFT p<0.01 two-sided, cluster area on the white surface, null = per-iteration
# max over both hemispheres, cluster-wise p<0.05). mri_glmfit refuses per-vertex
# regressors under permutation, so it cannot run Model B; it is run for Model A
# only, with mri_glmfit-sim --perm, as an independent cross-check.
#
# Usage: sbatch 03_glm.sh <VW_DIR> <WORK> [NPERM] [GLM_SUBDIR]
#   runs in WORK/<GLM_SUBDIR> (default "glm", written by make_design.py).
#   Pass "glm_cos" for the scale-invariant variant. Positional, not an env
#   var: Trillium submits with --export=NONE so exports do not reach the job.

set -euo pipefail

VW_DIR="${1:?Usage: sbatch 03_glm.sh VW_DIR WORK [NPERM]}"
NPERM="${3:-10000}"
GLM_DIR="${2:?}/${4:-glm}"
TRG=fsaverage6
CFT=2          # -log10(0.01), for mri_glmfit-sim
CWP=0.05
PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$GLM_DIR"

# ---- primary analysis: four (model, hemi) runs side by side, same seed ----
WORKERS=$(( ${SLURM_CPUS_ON_NODE:-192} / 4 - 2 ))
for model in A B; do
    for hemi in lh rh; do
        $PY "$VW_DIR/vertex_glm.py" --glm_dir . --model "$model" --hemi "$hemi" \
            --out_dir "py${model}_${hemi}" --nperm "$NPERM" --seed 0 --cft 0.01 \
            --workers "$WORKERS" --fs_subjects_dir "$SUBJECTS_DIR" \
            > "log_py${model}_${hemi}.txt" 2>&1 &
    done
done
wait
for model in A B; do
    $PY "$VW_DIR/combine_clusters.py" --glm_dir . --model "$model" --alpha "$CWP" \
        | tee "log_combine_${model}.txt"
done

# ---- cross-check: Model A through mri_glmfit / mri_glmfit-sim ----
for hemi in lh rh; do
    mri_glmfit --y "${hemi}.gbsc_slope.4d.mgh" --X X.mat --C contrast_A.mtx \
        --surf "$TRG" "$hemi" --cortex --eres-save --glmdir "fsA_${hemi}" > "log_fsA_${hemi}.txt" 2>&1
    mri_glmfit-sim --glmdir "fsA_${hemi}" --perm "$NPERM" "$CFT" abs \
        --cwp "$CWP" --2spaces --bg 64 --overwrite >> "log_fsA_${hemi}.txt" 2>&1 &
done
wait

echo "== mri_glmfit-sim Model A cluster summaries (cross-check)"
for hemi in lh rh; do
    echo "--- $hemi"
    grep -v "^#" "fsA_${hemi}/contrast_A/perm.th${CFT}0.abs.sig.cluster.summary" || echo "(none)"
done
echo "[DONE] glm"
