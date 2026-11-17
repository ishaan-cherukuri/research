"""False positive rate of the weighted, masked, one-sided variant under the null.

The planted effect is zero and the converter labels are reshuffled each trial,
so every cluster that survives correction is a false positive. A calibrated
procedure rejects at the nominal rate; one-sided testing inside a restricted
mask with subject weights is exactly the combination that could quietly fail to.
"""
import subprocess, sys, numpy as np, pandas as pd
from pathlib import Path

W, VW, PY = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
NTRIAL, NPERM = int(sys.argv[4]), int(sys.argv[5])
SIG = ("entorhinal,parahippocampal,inferiortemporal,middletemporal,"
       "precuneus,posteriorcingulate,fusiform")
subj = pd.read_csv(W / "cohort" / "vertex_subjects.csv")
best = []
for trial in range(NTRIAL):
    rng = np.random.default_rng(1000 + trial)
    s = subj.copy()
    s["event"] = rng.permutation(s["event"].to_numpy())
    cohort = W / "cohort" / "shuffled.csv"
    s.to_csv(cohort, index=False)
    g = W / "glm_cal"
    subprocess.run([PY, str(VW / "make_design.py"), "--subjects", str(cohort),
                    "--scans", str(W / "cohort" / "vertex_scans.csv"),
                    "--slopes_root", str(W / "slopes"), "--out_dir", str(g),
                    "--y_measure", "gbsc", "--covariates", "age,female",
                    "--field_subset", "all", "--weights"],
                   check=True, capture_output=True)
    for hemi in ("lh", "rh"):
        subprocess.run([PY, str(VW / "vertex_glm.py"), "--glm_dir", str(g), "--model", "A",
                        "--hemi", hemi, "--out_dir", str(g / f"pyA_{hemi}"),
                        "--nperm", str(NPERM), "--seed", "0", "--cft", "0.01",
                        "--workers", "4", "--fs_subjects_dir", str(W / "fsdir"),
                        "--tail", "neg", "--y_measure", "gbsc",
                        "--mask_labels", SIG, "--weights", str(g / "weights.txt")],
                       check=True, capture_output=True)
    subprocess.run([PY, str(VW / "combine_clusters.py"), "--glm_dir", str(g),
                    "--model", "A", "--alpha", "0.05"], check=True, capture_output=True)
    summ = pd.read_csv(g / "modelA_cluster_summary.csv")
    best.append(float(summ["cwp"].min()) if len(summ) else 1.0)
    print(f"  trial {trial + 1}/{NTRIAL}: best cwp {best[-1]:.4f}", flush=True)

b = np.array(best)
print(f"\n{NTRIAL} null trials, {NPERM} permutations each")
for a in (0.05, 0.10, 0.20):
    print(f"  rejected at alpha={a:.2f}: {int((b < a).sum())}/{NTRIAL} "
          f"= {100 * (b < a).mean():.0f}% (nominal {100 * a:.0f}%)")
print(f"  median best cwp {np.median(b):.3f} (uniform would give 0.50 "
      f"only if a cluster always forms; here many trials form none)")
