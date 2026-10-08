#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_recon_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --output=%x_%j.log

# One-hour debug-queue check that recon-all runs in the compute-node
# environment: autorecon1 on the first few scans of the list, into a throwaway
# SUBJECTS_DIR under WORK. Submit from WORK/logs (the driver does).
# Usage: sbatch 00_recon_debug.sh <VW_DIR> <WORK> [N]
set -euo pipefail
VW_DIR="${1:?}"; WORK="${2:?}"; N="${3:-4}"
SCANS_CSV="$VW_DIR/cohort/vertex_scans.csv"; SUBJECTS_DIR_WANT="$WORK/fs_debug"; LOG_DIR="$WORK/logs/recon_debug"
module load freesurfer/7.4.1
set +eu +o pipefail; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu -o pipefail
export SUBJECTS_DIR="$SUBJECTS_DIR_WANT"   # FreeSurferEnv.sh resets it, so set it afterwards
module load cobralab 2>/dev/null || true
which parallel recon-all
export OMP_NUM_THREADS=1
mkdir -p "$SUBJECTS_DIR" "$LOG_DIR"
one() {
    local id="$1" t1="$2"; local t0=$SECONDS
    recon-all -autorecon1 -i "$t1" -s "$id" -sd "$SUBJECTS_DIR" > "$LOG_DIR/${id}.log" 2>&1 \
        && echo "[OK] $id $(( (SECONDS - t0) / 60 )) min" || echo "[FAIL] $id rc=$?"
}
export -f one; export SUBJECTS_DIR LOG_DIR
tail -n +2 "$SCANS_CSV" | head -n "$N" | awk -F, '{print $1"\t"$3}' | parallel -j "$N" --colsep '\t' one {1} {2}
ls "$SUBJECTS_DIR"/*/mri | head -20
