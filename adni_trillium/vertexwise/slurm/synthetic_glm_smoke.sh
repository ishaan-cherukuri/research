#!/bin/bash
# Smoke test for the GLM stage on random data: builds a 30-subject design on
# fsaverage6 with two per-vertex regressors, runs Models A and B for one
# hemisphere, and a 20-permutation mri_glmfit-sim. Meant for a login node
# (a minute or two). Usage: bash synthetic_glm_smoke.sh <SCRATCH_DIR>
set -eu
VW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?}/vw_smoke"; rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
module load freesurfer/7.4.1; set +eu; source "$EBROOTFREESURFER/FreeSurferEnv.sh" >/dev/null; set -eu
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
PY="${PY:-/project/rrg-mchakrav-ab/ishaan/adni_trillium/.venv/bin/python}"
$PY - <<'PYEOF'
import numpy as np, nibabel as nib
rng=np.random.default_rng(0); n=417; nv=40962
def save(name, arr): nib.save(nib.MGHImage(arr.astype(np.float32).reshape(nv,1,1,-1), np.eye(4)), name)
y=rng.normal(size=(nv,n)); y[1000:1400, :133]+=0.6; save("lh.gbsc_slope.4d.mgh", y)
save("lh.thickness_slope.4d.mgh", rng.normal(size=(nv,n)))
save("lh.curv_slope.4d.mgh", rng.normal(size=(nv,n)))
conv=(np.arange(n)<133).astype(float); cov=rng.normal(size=(n,4))
X=np.column_stack([conv,1-conv,cov]); np.savetxt("X.mat",X,fmt="%.6f"); open("design_columns.txt","w").write("converter\nnonconverter\nage_c\nfemale_c\napoe4_c\nfield_3t_c\n")
cA=np.zeros((1,6)); cA[0,:2]=[1,-1]; cB=np.zeros((1,8)); cB[0,:2]=[1,-1]
np.savetxt("contrast_A.mtx",cA,fmt="%.1f"); np.savetxt("contrast_B.mtx",cB,fmt="%.1f")
PYEOF
mri_glmfit --y lh.gbsc_slope.4d.mgh --X X.mat --C contrast_A.mtx --surf fsaverage6 lh --cortex --eres-save --glmdir modelA_lh > glmA.log 2>&1
mri_glmfit --y lh.gbsc_slope.4d.mgh --X X.mat --C contrast_B.mtx --pvr lh.thickness_slope.4d.mgh --pvr lh.curv_slope.4d.mgh --surf fsaverage6 lh --cortex --eres-save --glmdir modelB_lh > glmB.log 2>&1
ls modelA_lh modelB_lh/contrast_B
mri_glmfit-sim --glmdir modelA_lh --perm 20 2 abs --cwp 0.05 --2spaces --overwrite > simA.log 2>&1 || { echo SIM-FAIL; tail -30 simA.log; exit 1; }
grep -v "^#" modelA_lh/contrast_A/perm.th20.abs.sig.cluster.summary | head -3
for model in A B; do
  $PY "$VW/vertex_glm.py" --glm_dir . --model $model --hemi lh --out_dir py${model}_lh --nperm 64 --seed 0 --workers 16 --fs_subjects_dir "$SUBJECTS_DIR"
  cp -r py${model}_lh py${model}_rh   # stand-in for the other hemisphere
  $PY "$VW/combine_clusters.py" --glm_dir . --model $model | head -6
done
echo SMOKE-OK
