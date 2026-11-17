#!/bin/bash
# Submit the vertex-wise variants, from the login node. Each stage is a
# separate invocation so the timings of the expensive one can be checked before
# the next is committed.
#
# What these answer, and why each exists.
#
# The three-term model (group, age, sex) put one cluster through correction:
# right middle temporal, 1,168 mm2, cwp 0.024. The same region with APOE4 and
# field strength in the model was 329 mm2 at cwp 0.25. Dropping covariates
# cannot create signal, so either the covariates were absorbing a real effect
# or the simple model is reading a confound as a group difference. Field
# strength is the candidate: 3 T covers 75 per cent of non-converters and 50
# per cent of converters.
#
#   stage prec      regenerate the gradient slopes to get slope_precision.csv,
#                   which the earlier run did not write. Values must match the
#                   existing slopes/ bitwise; verify_slopes.py checks that.
#   stage field     the confound test. 3 T and 1.5 T fitted separately, 67 and
#                   66 events, so a replication pair of near-equal power rather
#                   than a main result and a footnote. Also the field-stable
#                   subset, which is a power check and not a confound test:
#                   dropping the 50 mid-series switchers removes slope noise
#                   but leaves the between-subject imbalance untouched.
#   stage sigmoid   Olafson's sigmoid growth rate, vertex-wise. The fits exist
#                   for all 2,251 scans and have never been looked at
#                   spatially. The unbounded variant is the one carried here:
#                   the published estimator's floor binds on 87 per cent of
#                   cortex vertices.
#   stage base      } the longitudinal stream. Cross-sectional recon gives every
#   stage long      } timepoint an independent segmentation, and a slope is a
#   stage sample    } difference, so that noise enters twice. About 10,000
#   stage longslopes} core-hours in total. This is the one condition under which
#   stage longglm   } a real longitudinal effect would become visible, and the
#                   } condition a reviewer will ask about first.
#
# Every variant below is weighted by slope precision, restricted to the seven
# pre-specified AD-signature parcels, and tested one-sided in the hypothesised
# direction (converters blurring faster). Those three choices are fixed in
# advance for all of them, so this is one pre-specified configuration applied
# to several samples, not a search over configurations.
#
# Usage: ./run_variants.sh <stage> [NPERM]

set -euo pipefail

STAGE="${1:?Usage: ./run_variants.sh {prec|field|sigmoid|base|long|sample|longslopes|longglm|status} [NPERM]}"
NPERM="${2:-10000}"

VW="/project/rrg-mchakrav-ab/ishaan/adni_trillium/vertexwise"
WORK="$SCRATCH/adni_trillium/derivatives/vertexwise"
S="$VW/slurm"
LOGS="$WORK/logs/variants"
mkdir -p "$LOGS"
cd "$LOGS"

# weights, signature mask and one-sided tail, fixed for every variant
CONF=(yes signature neg)

case "$STAGE" in

prec)
    sbatch "$S/06_slopes.sh" "$VW" "$WORK" slopes_prec gbsc surf global gbsc surf
    ;;

field)
    for sub in 3t 15t stable; do
        sbatch "$S/05_glm_variant.sh" "$VW" "$WORK" "grad_$sub" slopes_prec gbsc \
            age,female "$sub" "${CONF[@]}" "$NPERM"
    done
    # the same configuration on the whole sample, so the strata have something
    # to be compared against under identical settings
    sbatch "$S/05_glm_variant.sh" "$VW" "$WORK" grad_all slopes_prec gbsc \
        age,female all "${CONF[@]}" "$NPERM"
    ;;

sigmoid)
    jid=$(sbatch --parsable "$S/06_slopes.sh" "$VW" "$WORK" \
          slopes_sigfree model_c_free_log sigmoid_surf demean finite surf)
    echo "slopes job $jid"
    for sub in all 3t 15t; do
        sbatch --dependency=afterok:"$jid" "$S/05_glm_variant.sh" "$VW" "$WORK" \
            "sig_$sub" slopes_sigfree model_c_free_log age,female "$sub" \
            "${CONF[@]}" "$NPERM"
    done
    ;;

base)
    sbatch --array=0-2 "$S/04_recon_long.sh" "$VW" "$WORK" base
    ;;

long)
    sbatch --array=0-14 "$S/04_recon_long.sh" "$VW" "$WORK" long
    ;;

sample)
    sbatch "$S/02_sample_surface.sh" "$VW" "$WORK" 10 dir long
    ;;

longslopes)
    sbatch "$S/06_slopes.sh" "$VW" "$WORK" slopes_long gbsc surf_long global gbsc surf_long
    ;;

longglm)
    for sub in all 3t 15t; do
        sbatch "$S/05_glm_variant.sh" "$VW" "$WORK" "long_$sub" slopes_long gbsc \
            age,female "$sub" "${CONF[@]}" "$NPERM"
    done
    # the published three-term model on longitudinal slopes, unweighted, whole
    # cortex, two-sided: the like-for-like comparison that isolates what the
    # longitudinal stream alone changed
    sbatch "$S/05_glm_variant.sh" "$VW" "$WORK" long_plain slopes_long gbsc \
        age,female all no none two "$NPERM"
    ;;

status)
    squeue -u "$USER" -o "%.10i %.20j %.9T %.10M %.6D %R"
    echo
    for d in "$WORK"/glm_*/; do
        [[ -d "$d" ]] || continue
        for m in A B; do
            f="$d/model${m}_cluster_summary.csv"
            [[ -f "$f" ]] && printf '%-28s %s\n' "$(basename "$d") model$m" \
                "$(tail -n +2 "$f" | head -1 | awk -F, '{print "best cwp "$9", "$4" mm2, "$8}')"
        done
    done
    ;;

*)
    echo "unknown stage $STAGE"; exit 1 ;;
esac
