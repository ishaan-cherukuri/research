#!/bin/bash
#SBATCH --account=rrg-mchakrav-ab
#SBATCH --job-name=vw_collect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log

# Gather every variant's result into one file, so the findings can be fetched in
# a single read instead of polling the queue. Submit it with a dependency on the
# jobs it summarises; SciNet's two-factor login makes repeated connections
# expensive, and a loop of them will knock the SSH control master over.
#
# Usage: sbatch --dependency=afterany:J1:J2:... 99_collect.sh <WORK> <OUT_FILE>

set -euo pipefail

W="${1:?Usage: sbatch 99_collect.sh WORK OUT_FILE}"
OUT="${2:?}"

{
  echo "collected $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  for d in "$W"/glm_grad_* "$W"/glm_sig_* "$W"/glm_long_*; do
      [[ -d "$d" ]] || continue
      echo "=================================================================="
      echo "## $(basename "$d")"
      sed -n '1,3p' "$d/log_design.txt" 2>/dev/null | grep -E "weights:|design:|field subset|slope-outlier" || true
      for m in A B; do
          f="$d/model${m}_cluster_summary.csv"
          [[ -f "$f" ]] || continue
          grep -E "survive" "$d/log_combine_${m}.txt" 2>/dev/null || true
          # the three strongest clusters by corrected p
          head -4 "$f" | column -s, -t
      done
      echo
  done

  echo "=================================================================="
  echo "## split-half validation"
  for f in "$W"/glm_grad_*/split_half_model*.csv; do
      [[ -f "$f" ]] || continue
      echo "--- $f"
      python3 - "$f" <<'PYEOF' 2>/dev/null || echo "(no python summary)"
import sys, csv
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r.get("d_test") not in (None, "", "nan")]
if not rows:
    print("  no splits formed a cluster")
else:
    import statistics as st
    tr = sorted(float(r["d_train"]) for r in rows)
    te = sorted(float(r["d_test"]) for r in rows)
    def q(v, p):
        return v[min(len(v) - 1, int(p * len(v)))]
    neg = sum(1 for r in rows if float(r["d_test"]) < 0) / len(rows)
    sig = sum(1 for r in rows if float(r["p_test"]) < 0.05
              and float(r["d_test"]) < 0) / len(rows)
    print(f"  splits {len(rows)}")
    print(f"  d train   median {st.median(tr):+.3f}  [{q(tr,.025):+.3f}, {q(tr,.975):+.3f}]")
    print(f"  d heldout median {st.median(te):+.3f}  [{q(te,.025):+.3f}, {q(te,.975):+.3f}]")
    print(f"  heldout in hypothesised direction {100*neg:.0f}%, p<0.05 {100*sig:.0f}%")
    print(f"  shrinkage {st.median(te)/st.median(tr):.2f}")
PYEOF
  done

  echo
  echo "=================================================================="
  echo "## sigmoid surface coverage (normalized convolution vs plain smoothing)"
  for root in sigmoid_surf sigmoid_surf_nn; do
      [[ -d "$W/$root" ]] || continue
      echo "--- $root"
      ls "$W/$root" | wc -l | sed 's/^/  scan dirs: /'
  done
} > "$OUT" 2>&1

echo "[DONE] wrote $OUT"
