"""Check that the regenerated gradient slopes match the published ones.

compute_vertex_slopes.py was extended to emit slope_precision.csv and to take
an arbitrary measure, which meant touching the loop that writes the slope maps.
The gradient run has to come out unchanged, or every published vertex-wise
result silently moves. This compares two slopes roots map by map and reports
the largest absolute difference found.

Usage:
    python3 verify_slopes.py --a .../slopes --b .../slopes_prec
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

HEMIS = ("lh", "rh")
MEASURES = ("gbsc", "thickness", "curv")


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float64).reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference slopes root")
    ap.add_argument("--b", required=True, help="regenerated slopes root")
    ap.add_argument("--limit", type=int, default=0, help="only the first N subjects")
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)

    sa = pd.read_csv(A / "subjects_done.csv")
    sb = pd.read_csv(B / "subjects_done.csv")
    print(f"subjects: {len(sa)} in a, {len(sb)} in b")
    only_a = sorted(set(sa["subject"]) - set(sb["subject"]))
    only_b = sorted(set(sb["subject"]) - set(sa["subject"]))
    if only_a or only_b:
        print(f"MISMATCHED COHORT: only in a {only_a}; only in b {only_b}")

    shared = sorted(set(sa["subject"]) & set(sb["subject"]))
    if args.limit:
        shared = shared[:args.limit]
    worst = {m: 0.0 for m in MEASURES}
    worst_at = {m: "" for m in MEASURES}
    for i, s in enumerate(shared):
        for hemi in HEMIS:
            for meas in MEASURES:
                f = f"{hemi}.{meas}_slope.mgh"
                d = float(np.nanmax(np.abs(load(A / s / f) - load(B / s / f))))
                if d > worst[meas]:
                    worst[meas], worst_at[meas] = d, f"{s} {f}"
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(shared)}", flush=True)

    print(f"compared {len(shared)} subjects")
    for m in MEASURES:
        verdict = "identical" if worst[m] == 0 else f"DIFFERS at {worst_at[m]}"
        print(f"  {m:10s} max |a-b| = {worst[m]:.3e}  {verdict}")
    if all(v == 0 for v in worst.values()):
        print("PASS: regenerated slopes are bitwise identical")
    else:
        raise SystemExit("FAIL: slopes moved; do not use the new root")


if __name__ == "__main__":
    main()
