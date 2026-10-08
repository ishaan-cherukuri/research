#!/bin/bash
# Driver for the vertex-wise complementary-information analysis. Run on a
# Trillium login node from this directory. Each stage is idempotent; rerun a
# stage to pick up scans that failed or were added.
#
#   bash run_vertexwise.sh debug        1 h debug-queue recon-all check (4 scans)
#   bash run_vertexwise.sh recon        recon-all on all 2,252 scans (15-node array)
#   bash run_vertexwise.sh recon-status count finished / failed / pending scans
#   bash run_vertexwise.sh sample       gBSC, thickness, curvature -> fsaverage6
#   bash run_vertexwise.sh slopes       per-subject slope maps (login node)
#   bash run_vertexwise.sh design       stack slopes, write X.mat and contrasts (login node)
#   bash run_vertexwise.sh glm          Models A and B + permutation correction
#   bash run_vertexwise.sh fetch        pull GLM results to $VW_DIR/results/ on /project
#
# Environment overrides: WORK (scratch root), NPERM, FWHM, N_RECON_NODES, PY,
# NORMALIZE (global|none, for the slopes stage).
set -euo pipefail

VW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$SCRATCH/adni_trillium/derivatives/vertexwise}"
NPERM="${NPERM:-10000}"
FWHM="${FWHM:-10}"
N_RECON_NODES="${N_RECON_NODES:-15}"
PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"
SCANS="$VW_DIR/cohort/vertex_scans.csv"
FS_SUBJECTS="${FS_SUBJECTS:-/cvmfs/restricted.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/freesurfer/7.4.1/subjects}"
SUBJECTS="$VW_DIR/cohort/vertex_subjects.csv"

mkdir -p "$WORK/logs"
stage="${1:?stage required: debug|recon|recon-status|sample|slopes|design|glm|fetch}"

submit() {  # sbatch from WORK/logs so job logs land on scratch
    (cd "$WORK/logs" && sbatch "$@")
}

case "$stage" in
    debug)
        submit "$VW_DIR/slurm/00_recon_debug.sh" "$VW_DIR" "$WORK" 4 ;;
    recon)
        submit --array=0-$((N_RECON_NODES - 1)) "$VW_DIR/slurm/01_recon_all.sh" "$VW_DIR" "$WORK" ;;
    recon-status)
        total=$(($(wc -l < "$SCANS") - 1))
        done_n=$(find "$WORK/freesurfer" -maxdepth 3 -name recon-all.done 2>/dev/null | wc -l)
        err_n=$(find "$WORK/freesurfer" -maxdepth 3 -name recon-all.error 2>/dev/null | wc -l)
        running=$(find "$WORK/freesurfer" -maxdepth 3 -name 'IsRunning.*' 2>/dev/null | wc -l)
        echo "scans: $total  done: $done_n  error: $err_n  running-flag: $running  pending: $((total - done_n))"
        [[ $err_n -gt 0 ]] && find "$WORK/freesurfer" -maxdepth 3 -name recon-all.error -exec dirname {} \; \
            | xargs -n1 dirname | xargs -n1 basename | head -20
        squeue -u "$USER" -h | wc -l | xargs echo "queued/running jobs:" ;;
    sample)
        submit "$VW_DIR/slurm/02_sample_surface.sh" "$VW_DIR" "$WORK" "$FWHM" ;;
    slopes)
        $PY "$VW_DIR/compute_vertex_slopes.py" --scans "$SCANS" \
            --surf_root "$WORK/surf" --out_root "$WORK/slopes" --normalize "${NORMALIZE:-global}" \
            --fs_subjects_dir "$FS_SUBJECTS" ;;
    design)
        $PY "$VW_DIR/make_design.py" --subjects "$SUBJECTS" --scans "$SCANS" \
            --slopes_root "$WORK/slopes" --out_dir "$WORK/glm" ;;
    glm)
        submit "$VW_DIR/slurm/03_glm.sh" "$VW_DIR" "$WORK" "$NPERM" ;;
    fetch)
        out="$VW_DIR/results"; mkdir -p "$out"
        cp "$WORK/glm"/model?_cluster_summary.csv "$WORK/glm"/*.sig_clusters.mgh \
           "$WORK/glm"/*.cwp.mgh "$WORK/glm"/subjects_used.csv "$WORK/glm"/design_columns.txt "$out/"
        for d in "$WORK/glm"/py?_?h; do
            b=$(basename "$d"); mkdir -p "$out/$b"
            cp "$d"/t.mgh "$d"/beta.mgh "$d"/sig.mgh "$d"/cluster_id.mgh "$d"/clusters_obs.csv \
               "$d"/info.json "$d"/perm_max_area.npy "$out/$b/"
        done
        for d in "$WORK/glm"/fsA_?h; do
            b=$(basename "$d"); mkdir -p "$out/$b"
            cp "$d"/contrast_A/perm.th20.abs.sig.cluster.summary "$out/$b/" 2>/dev/null || true
        done
        cp "$WORK/slopes/subjects_done.csv" "$WORK/slopes/missing.csv" "$out/" 2>/dev/null || true
        echo "results in $out" ;;
    *) echo "unknown stage $stage"; exit 1 ;;
esac
