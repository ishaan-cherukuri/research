"""Olafson's sigmoid BSC, driven from FreeSurfer surfaces instead of CIVET.

The published estimator needs one thing: ten T1 intensity samples per vertex at
fixed percentages of cortical thickness spanning the gray/white boundary. CIVET
is how the reference produces them, but nothing about the measure requires it.
FreeSurfer gives the same three ingredients (a white surface, a pial surface
and a per-vertex thickness), and this cohort already has a cross-sectional
recon-all for all 2,251 pre-outcome scans, so the profiles can be sampled
directly and handed to the unmodified reference fit in fit_sigmoid.py.

That matters practically: the CIVET stage in the companion pipeline is budgeted
at 26,000 to 52,000 core-hours, against roughly 200 here.

Sampling. For each vertex, the unit vector from the white surface toward the
pial surface defines the profile axis, and positions are placed at
X_PERCENT percent of that vertex's own cortical thickness along it, negative
values running into white matter. Intensities come from nu.mgz, which is bias
corrected but not intensity normalised, by trilinear interpolation in the
conformed voxel grid. Intensity normalisation is not needed: the growth rate c
is a rate in percent-thickness, and a multiplicative rescaling of the profile is
absorbed by the fitted a and exp(k), which is the property the gradient measure
in mri-bsc/code/seg/atropos_bsc.py lacks.

Known departures from the reference, beyond the surface source:
  - FreeSurfer native meshes carry roughly 130k vertices per hemisphere against
    CIVET's 40,962. The fit runs per native vertex and the result is resampled
    afterwards, which mirrors the reference's order (fit, log, smooth, resample)
    rather than its vertex count.
  - FreeSurfer's aparc has an entorhinal parcel, so the AD-signature composite
    here is the full seven regions rather than the six the CIVET atlas allows.

Usage:
    python3 fs_sigmoid_bsc.py --subjects_dir $SCRATCH/.../freesurfer \
        --image_id <id> --out_dir $SCRATCH/.../sigmoid/native
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer import read_geometry, read_morph_data
from scipy.ndimage import map_coordinates

from fit_sigmoid import X_PERCENT, fit_hemisphere

HEMIS = ("lh", "rh")
SAVE = ("model_c", "model_c_log", "model_c_free", "model_c_free_log",
        "model_ratio", "rmse", "converged")


def sample_profiles(subj_dir: Path, hemi: str, vol_name: str = "nu.mgz"):
    """Ten intensity samples per vertex, ordered white matter -> pial.

    Returns (10, n_vertices) and the vertex count.
    """
    vol = nib.load(str(subj_dir / "mri" / vol_name))
    data = np.asarray(vol.dataobj, dtype=np.float32)
    # Surfaces are stored in tkrRAS, which is not the volume's world frame.
    inv = np.linalg.inv(vol.header.get_vox2ras_tkr())

    white, _ = read_geometry(str(subj_dir / "surf" / f"{hemi}.white"))
    pial, _ = read_geometry(str(subj_dir / "surf" / f"{hemi}.pial"))
    thick = read_morph_data(str(subj_dir / "surf" / f"{hemi}.thickness"))

    axis = pial - white
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    # Medial wall vertices can have coincident white and pial points; they carry
    # zero thickness too and are masked out of every downstream statistic.
    unit = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 1e-6)

    n = white.shape[0]
    profiles = np.empty((len(X_PERCENT), n), dtype=np.float64)
    for i, pct in enumerate(X_PERCENT):
        pos = white + (pct / 100.0) * thick[:, None] * unit
        vox = (inv[:3, :3] @ pos.T) + inv[:3, 3:4]
        profiles[i] = map_coordinates(data, vox, order=1, mode="nearest")
    return profiles, n, thick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects_dir", required=True)
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--volume", default="nu.mgz")
    args = ap.parse_args()

    subj_dir = Path(args.subjects_dir) / args.image_id
    out = Path(args.out_dir) / args.image_id
    if (out / f"rh.model_c.mgh").exists():
        print(f"[SKIP-DONE] {args.image_id}")
        return
    if not (subj_dir / "scripts" / "recon-all.done").exists():
        print(f"[NO-RECON] {args.image_id}")
        return
    out.mkdir(parents=True, exist_ok=True)

    for hemi in HEMIS:
        profiles, n, thick = sample_profiles(subj_dir, hemi, args.volume)
        res = fit_hemisphere(profiles)
        # Zero thickness means no cortex to profile; mark rather than fit.
        bad = thick <= 0
        for k in ("model_c", "model_c_log", "model_c_free",
                  "model_c_free_log", "model_ratio"):
            res[k] = np.where(bad, np.nan, res[k])
        for k in SAVE:
            img = nib.MGHImage(np.asarray(res[k], dtype=np.float32).reshape(-1, 1, 1),
                               np.eye(4))
            nib.save(img, str(out / f"{hemi}.{k}.mgh"))
        ok = 100.0 * np.mean(res["converged"][~bad]) if (~bad).any() else 0.0
        print(f"[OK] {args.image_id} {hemi} n={n} converged={ok:.1f}% "
              f"median_c={np.nanmedian(res['model_c']):.4f}")


if __name__ == "__main__":
    main()
