#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=cos_bsc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log

# Scale-invariant boundary sharpness, cos(theta) = |grad I . n| / |grad I|,
# per scan: global summaries plus DKT parcel means. Fans subjects across the
# node with GNU Parallel; each worker writes its own CSV and they are merged
# at the end.
#
# Usage: sbatch run_cos_bsc.sh <PROJECT_DIR> <BSC_ROOT> <MANIFEST> <OUT_DIR>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_cos_bsc.sh PROJECT_DIR BSC_ROOT MANIFEST OUT_DIR}"
BSC_ROOT="${2:?}"
MANIFEST="${3:?}"
OUT_DIR="${4:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/mri-bsc:${PYTHONPATH:-}"
# ANTsXNet and ANTs each try to grab every core; one thread per worker instead.
export OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p "$OUT_DIR/parts"

N_SUBJ=$(python3 -c "
import pandas as pd; print(pd.read_csv('$MANIFEST')['subject'].nunique())")
# Concurrency and chunk size are set separately. Each ANTsXNet worker loads
# its own TensorFlow models and peaks near 10 GB, so the node's memory, not its
# core count, is the binding constraint: 120 workers was killed OOM. Keeping
# CHUNK fixed across runs means finished part files stay valid, so a rerun only
# picks up what is missing.
N_JOBS="${COS_WORKERS:-40}"
CHUNK="${COS_CHUNK:-$(( (N_SUBJ + 119) / 120 ))}"
echo "subjects: $N_SUBJ, workers: $N_JOBS, chunk: $CHUNK"

run_chunk() {
    local skip="$1"
    local out="${OUT_DIR}/parts/cos_${skip}.csv"
    if [[ -s "$out" ]]; then echo "[SKIP-DONE] chunk $skip"; return 0; fi
    python3 "${PROJECT_DIR}/compute_cos_bsc.py" \
        --manifest "$MANIFEST" --bsc_root "$BSC_ROOT" \
        --out_csv "$out" \
        --skip "$skip" --limit "$CHUNK"
}
export -f run_chunk
export PROJECT_DIR MANIFEST BSC_ROOT OUT_DIR CHUNK

seq 0 "$CHUNK" $(( N_SUBJ - 1 )) | parallel -j "$N_JOBS" --joblog "$OUT_DIR/cos.joblog" run_chunk {}

python3 - <<PYEOF
import glob, pandas as pd
parts = sorted(glob.glob("${OUT_DIR}/parts/cos_*.csv"))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df = df.sort_values(["subject", "acq_date"]).reset_index(drop=True)
df.to_csv("${OUT_DIR}/per_scan_cos.csv", index=False)
print(f"merged {len(parts)} parts -> {len(df)} scan-rows, {df['subject'].nunique()} subjects")
PYEOF
echo "[DONE] cos features"
