#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_var
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# One named vertex-wise variant end to end: design, Models A and B on both
# hemispheres, cluster-wise correction over the pair. 03_glm.sh is the primary
# analysis and also cross-checks Model A against mri_glmfit; this is for the
# variants that mri_glmfit cannot express at all (weights, a restricted search
# space, a one-sided tail), so there is nothing to cross-check against and it
# is left out.
#
# Every option is positional. Trillium submits with --export=NONE
# --get-user-env, so an exported variable never reaches the job.
#
# Usage: sbatch 05_glm_variant.sh <VW_DIR> <WORK> <NAME> <SLOPES_SUBDIR> \
#                <Y_MEASURE> <COVARIATES> <FIELD_SUBSET> <WEIGHTS> <MASK> <TAIL> [NPERM]
#
#   NAME           output goes to WORK/glm_<NAME>
#   SLOPES_SUBDIR  e.g. slopes, slopes_long, slopes_sigfree
#   Y_MEASURE      gbsc | bscsigfree | bscsig | ...
#   COVARIATES     comma-separated, or "none" for group alone
#   FIELD_SUBSET   all | stable | 3t | 15t
#   WEIGHTS        yes | no      (slope-precision weights)
#   MASK           signature | none
#   TAIL           two | neg | pos
#   NPERM          permutations, default 10000
#   DROP_PCT       drop subjects above this percentile of cortex-mean |slope|,
#                  outcome-blind; 0 (the default) keeps everyone
#   WITHIN_GROUP   yes takes DROP_PCT inside each group instead of over the
#                  pooled sample. A pooled cut on |slope| is outcome-blind but
#                  not outcome-neutral: it eats the fast changers, who are
#                  enriched for converters, so it can remove the effect it is
#                  meant to test.
#   SEED           permutation seed, default 0; vary it to check that a
#                  cluster-wise p is not an artefact of one permutation stream
#
# Example, the maximally favourable pre-specified configuration:
#   sbatch 05_glm_variant.sh "$VW" "$WORK" conf_15t slopes_long bscsigfree \
#       age,female 15t yes signature neg

set -euo pipefail

VW_DIR="${1:?see header for usage}"
WORK="${2:?}"
NAME="${3:?}"
SLOPES_SUBDIR="${4:?}"
Y_MEASURE="${5:?}"
COVARIATES="${6:?}"
FIELD_SUBSET="${7:?}"
WEIGHTS="${8:?}"
MASK="${9:?}"
TAIL="${10:?}"
NPERM="${11:-10000}"
DROP_PCT="${12:-0}"
SEED="${13:-0}"
WITHIN_GROUP="${14:-no}"   # yes = take DROP_PCT inside each group, not pooled

GLM_DIR="$WORK/glm_${NAME}"
SLOPES_ROOT="$WORK/${SLOPES_SUBDIR}"
CWP=0.05
PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"

# The seven bilateral regions pre-specified as the AD signature, the same set
# compute_regional_bsc.py and fs_sigmoid_features.py use, under aparc names.
SIGNATURE="entorhinal,parahippocampal,inferiortemporal,middletemporal,precuneus,posteriorcingulate,fusiform"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$GLM_DIR"

DESIGN_ARGS=(--field_subset "$FIELD_SUBSET" --y_measure "$Y_MEASURE")
[[ "$COVARIATES" != "none" ]] && DESIGN_ARGS+=(--covariates "$COVARIATES")
[[ "$COVARIATES" == "none" ]] && DESIGN_ARGS+=(--covariates "")
[[ "$WEIGHTS" == "yes" ]] && DESIGN_ARGS+=(--weights)
[[ "$DROP_PCT" != "0" ]] && DESIGN_ARGS+=(--drop_slope_outlier_pct "$DROP_PCT")
[[ "$WITHIN_GROUP" == "yes" ]] && DESIGN_ARGS+=(--outlier_within_group)

$PY "$VW_DIR/make_design.py" \
    --subjects "$VW_DIR/cohort/vertex_subjects.csv" \
    --scans "$VW_DIR/cohort/vertex_scans.csv" \
    --slopes_root "$SLOPES_ROOT" --out_dir "$GLM_DIR" \
    "${DESIGN_ARGS[@]}" | tee "$GLM_DIR/log_design.txt"

GLM_ARGS=(--tail "$TAIL" --y_measure "$Y_MEASURE")
[[ "$MASK" == "signature" ]] && GLM_ARGS+=(--mask_labels "$SIGNATURE")
[[ "$WEIGHTS" == "yes" ]] && GLM_ARGS+=(--weights "$GLM_DIR/weights.txt")

cd "$GLM_DIR"
echo "== variant $NAME: y=$Y_MEASURE slopes=$SLOPES_SUBDIR cov=$COVARIATES "\
"field=$FIELD_SUBSET weights=$WEIGHTS mask=$MASK tail=$TAIL nperm=$NPERM "\
"drop_pct=$DROP_PCT within_group=$WITHIN_GROUP seed=$SEED"

WORKERS=$(( ${SLURM_CPUS_ON_NODE:-192} / 4 - 2 ))
for model in A B; do
    for hemi in lh rh; do
        $PY "$VW_DIR/vertex_glm.py" --glm_dir . --model "$model" --hemi "$hemi" \
            --out_dir "py${model}_${hemi}" --nperm "$NPERM" --seed "$SEED" --cft 0.01 \
            --workers "$WORKERS" --fs_subjects_dir "$SUBJECTS_DIR" \
            "${GLM_ARGS[@]}" > "log_py${model}_${hemi}.txt" 2>&1 &
    done
done
wait

for model in A B; do
    $PY "$VW_DIR/combine_clusters.py" --glm_dir . --model "$model" --alpha "$CWP" \
        | tee "log_combine_${model}.txt"
done
echo "[DONE] variant $NAME"
