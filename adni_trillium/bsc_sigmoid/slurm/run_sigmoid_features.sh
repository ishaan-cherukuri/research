#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=sig_feat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# Resample the per-vertex sigmoid fits to fsaverage6 and aggregate them by the
# FreeSurfer aparc parcellation, producing the per-scan feature CSV that
# build_spec_cohort.py reads. Chunks are fanned across the node; each writes its
# own CSV and they are concatenated at the end.
#
# Positional, not environment: Trillium submits with --export=NONE.
#
# SURF_CACHE is the directory of resampled fsaverage6 maps. It is a positional
# argument and not a fixed path because fs_sigmoid_features.py treats an
# existing target map as done and returns it untouched, so a cache written by an
# older resampling method is silently reused. The first cache, sigmoid_surf, was
# built by plain smoothing and holds maps that are 66 per cent NaN; the
# normalized-convolution rebuild has to go somewhere else or it would read that
# one straight back.
#
# Usage: sbatch run_sigmoid_features.sh <PROJECT_DIR> <WORK> <NATIVE_ROOT> <OUT_DIR> [WORKERS] [SURF_CACHE]

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_sigmoid_features.sh PROJECT_DIR WORK NATIVE_ROOT OUT_DIR [WORKERS]}"
WORK="${2:?}"
NATIVE_ROOT="${3:?}"
OUT_DIR="${4:?}"
N_JOBS="${5:-60}"
SURF_CACHE="${6:-$WORK/sigmoid_surf}"
SCANS_CSV="$PROJECT_DIR/vertexwise/cohort/vertex_scans.csv"

module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
module load cobralab 2>/dev/null || true
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/bsc_sigmoid:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

FS_SUBJECTS="$FREESURFER_HOME/subjects"
mkdir -p "$OUT_DIR/parts" "$SURF_CACHE"

TOTAL=$(($(wc -l < "$SCANS_CSV") - 1))
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))
echo "scans: $TOTAL, workers: $N_JOBS, chunk: $CHUNK, cache: $SURF_CACHE"

feat_chunk() {
    local skip="$1"
    local out="${OUT_DIR}/parts/per_scan_sigmoid_${skip}.csv"
    if [[ -s "$out" ]]; then echo "[SKIP-DONE] $skip"; return 0; fi
    python3 "${PROJECT_DIR}/bsc_sigmoid/fs_sigmoid_features.py" \
        --native_root "$NATIVE_ROOT" \
        --subjects_dir "$WORK/freesurfer" \
        --scans "$SCANS_CSV" \
        --fs_subjects_dir "$FS_SUBJECTS" \
        --work "$SURF_CACHE" \
        --out_dir "${OUT_DIR}/parts" \
        --skip "$skip" --limit "$CHUNK" > /dev/null 2>&1 \
        && mv "${OUT_DIR}/parts/per_scan_sigmoid_${skip}.csv" "$out" 2>/dev/null || true
    [[ -s "$out" ]] && echo "[OK] $skip" || echo "[FAIL] $skip"
}
export -f feat_chunk
export PROJECT_DIR WORK NATIVE_ROOT OUT_DIR CHUNK SCANS_CSV FS_SUBJECTS SURF_CACHE

seq 0 "$CHUNK" $(( TOTAL - 1 )) | parallel -j "$N_JOBS" feat_chunk {}

python3 - <<PYEOF
import glob, pandas as pd
parts = sorted(glob.glob("${OUT_DIR}/parts/per_scan_sigmoid_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.drop_duplicates("image_id").sort_values(["subject", "acq_date"]).reset_index(drop=True)
df.to_csv("${OUT_DIR}/per_scan_sigmoid.csv", index=False)
print(f"merged {len(parts)} parts -> {len(df)} scans, {df['subject'].nunique()} subjects, "
      f"{df['image_id'].nunique()} unique image_id")
PYEOF
echo "[DONE] sigmoid features"
