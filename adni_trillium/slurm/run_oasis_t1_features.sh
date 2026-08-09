#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=oasis_t1_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

# T1 morphometry/QC feature extraction for OASIS, reusing already-computed BSC outputs.
# Usage: sbatch run_oasis_t1_features.sh <PROJECT_DIR> <SCRATCH_DERIV_DIR> <MANIFEST_CSV>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sbatch run_oasis_t1_features.sh PROJECT_DIR SCRATCH_DERIV_DIR MANIFEST_CSV}"
SCRATCH_DERIV_DIR="${2:?}"
MANIFEST_CSV="${3:?}"

module load cobralab
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/mri-bsc:${PYTHONPATH:-}"

BSC_ROOT="${SCRATCH_DERIV_DIR}/bsc"

N_JOBS="${SLURM_CPUS_ON_NODE:-192}"
TOTAL=$(($(wc -l < "$MANIFEST_CSV") - 1))
CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))

extract_t1_chunk() {
    local skip="$1"
    python3 "${PROJECT_DIR}/extract_t1_scan_features.py" \
        --manifest "$MANIFEST_CSV" --bsc_root "$BSC_ROOT" \
        --out_csv "${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_${skip}.csv" \
        --skip "$skip" --limit "$CHUNK"
}
export -f extract_t1_chunk
export MANIFEST_CSV BSC_ROOT SCRATCH_DERIV_DIR PROJECT_DIR CHUNK
mkdir -p "${SCRATCH_DERIV_DIR}/t1_scan_features_parts"
seq 0 "$CHUNK" "$TOTAL" | parallel -j "$N_JOBS" extract_t1_chunk {}

python3 -c "
import pandas as pd, glob
parts = sorted(glob.glob('${SCRATCH_DERIV_DIR}/t1_scan_features_parts/part_*.csv'))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
df.to_csv('${SCRATCH_DERIV_DIR}/t1_scan_features.csv', index=False)
print('merged', len(df), 'rows from', len(parts), 'parts')
"

python3 "${PROJECT_DIR}/build_t1_subject_features.py" \
    --scan_features "${SCRATCH_DERIV_DIR}/t1_scan_features.csv" \
    --survival "${PROJECT_DIR}/survival_labels_oasis.csv" \
    --out_csv "${SCRATCH_DERIV_DIR}/features_t1_all.csv"

echo "[DONE] OASIS T1 features complete"
