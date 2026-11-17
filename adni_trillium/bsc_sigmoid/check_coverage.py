"""How much cortex each sigmoid resampling actually keeps.

The native fits do not converge everywhere, so the fsaverage6 maps inherit
some missing data. The question is how much, because plain smoothing spreads
every NaN across its whole kernel: at 20 mm FWHM that turned 7 per cent
missing into 66 per cent, and every ROI mean became an average over whichever
part of the parcel sat furthest from a failed fit. Normalized convolution
should bring coverage back close to native.

Reports the finite fraction per stage, and for the vertices both resamplings
keep, how far apart their values are. A large discrepancy there means the old
features were not merely sparser but biased.

Usage:
    python3 check_coverage.py --native_root .../sigmoid/native \
        --old .../vertexwise/sigmoid_surf --new .../vertexwise/sigmoid_surf_nn [--limit 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

HEMIS = ("lh", "rh")
MEASURES = ("model_c_log", "model_c_free_log", "model_ratio")
TRG = "fsaverage6"


def finite_frac(p: Path) -> tuple[float, np.ndarray]:
    v = np.asarray(nib.load(str(p)).dataobj, dtype=np.float64).reshape(-1)
    return float(np.isfinite(v).mean()), v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native_root", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    nat, old, new = Path(args.native_root), Path(args.old), Path(args.new)

    ids = [d.name for d in sorted(new.iterdir()) if d.is_dir()][: args.limit]
    if not ids:
        raise SystemExit(f"nothing in {new}")
    print(f"{len(ids)} scans\n")
    print(f"{'measure':18s} {'native':>8s} {'old':>8s} {'new':>8s} "
          f"{'shared':>8s} {'median |old-new|':>17s} {'r':>6s}")
    for meas in MEASURES:
        fn, fo, fnew, sh, diff, cors = [], [], [], [], [], []
        for iid in ids:
            for hemi in HEMIS:
                pn = nat / iid / f"{hemi}.{meas}.mgh"
                po = old / iid / f"{hemi}.{meas}.{TRG}.mgh"
                pw = new / iid / f"{hemi}.{meas}.{TRG}.mgh"
                if not (pn.exists() and pw.exists()):
                    continue
                a, _ = finite_frac(pn)
                fn.append(a)
                c, vw = finite_frac(pw)
                fnew.append(c)
                if po.exists():
                    b, vo = finite_frac(po)
                    fo.append(b)
                    both = np.isfinite(vo) & np.isfinite(vw)
                    sh.append(float(both.mean()))
                    if both.sum() > 100:
                        diff.append(float(np.median(np.abs(vo[both] - vw[both]))))
                        cors.append(float(np.corrcoef(vo[both], vw[both])[0, 1]))
        if not fn:
            continue
        print(f"{meas:18s} {100*np.mean(fn):7.1f}% "
              f"{100*np.mean(fo) if fo else float('nan'):7.1f}% "
              f"{100*np.mean(fnew):7.1f}% "
              f"{100*np.mean(sh) if sh else float('nan'):7.1f}% "
              f"{np.median(diff) if diff else float('nan'):17.5f} "
              f"{np.mean(cors) if cors else float('nan'):6.3f}")

    print("\nnative is the per-vertex fit; old is plain smoothing, which spreads NaN; "
          "new is normalized convolution.\nshared is the fraction of vertices both "
          "resamplings keep, and the last two columns compare them there.")


if __name__ == "__main__":
    main()
