"""Recompute the dense gBSC field for one scan, unmasked by the boundary band.

The stored bsc_dir_map.nii.gz is |dI/dn| (image gradient projected on the
tissue-probability normal) zeroed outside the atropos boundary band, which is
the thin set of voxels with GM probability within eps of 0.5. That band is
sparse, covers a few percent of the gray-white interface, and is empty in a
quarter of scans, so it cannot be sampled onto a surface. The field itself is
defined everywhere. This recomputes it with the same smoothing (sigma 1 mm)
from the preprocessed T1 and GM probability that the volumetric pipeline
stored, restricted to the brain mask, so that the surface stage can evaluate
it at the FreeSurfer white surface, which is where the boundary sits.

Two measures are available. "dir" is the gradient projection |grad I . n|,
which is what the volumetric pipeline stores and which scales linearly with
image intensity. "cos" divides it by the full gradient magnitude, giving
|grad I . n| / |grad I|, the cosine of the angle between the intensity
gradient and the tissue boundary. Because n is a unit vector, that ratio is
bounded in [0, 1] and is invariant to any rescaling of image intensity, which
is the property the gradient measure lacks.

Usage: python3 compute_gbsc_field.py <bsc_dir> <out_nii> [--measure dir|cos]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def gradient_phys(vol: np.ndarray, spacing):
    gx, gy, gz = np.gradient(vol.astype(np.float32), *spacing)
    return gx, gy, gz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bsc_dir")
    ap.add_argument("out")
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    ap.add_argument("--measure", choices=["dir", "cos"], default="dir")
    args = ap.parse_args()
    d = Path(args.bsc_dir)
    t1 = nib.load(str(d / "t1w_preproc.nii.gz"))
    gm = np.asarray(nib.load(str(d / "gm_prob.nii.gz")).dataobj, dtype=np.float32)
    mask = np.asarray(nib.load(str(d / "brain_mask.nii.gz")).dataobj) > 0
    spacing = t1.header.get_zooms()[:3]
    sig = [args.sigma_mm / max(s, 1e-6) for s in spacing]
    I_s = gaussian_filter(np.asarray(t1.dataobj, dtype=np.float32), sigma=sig, mode="nearest")
    P_s = gaussian_filter(gm, sigma=sig, mode="nearest")
    gIx, gIy, gIz = gradient_phys(I_s, spacing)
    gPx, gPy, gPz = gradient_phys(P_s, spacing)
    gP = np.sqrt(gPx * gPx + gPy * gPy + gPz * gPz) + 1e-8
    field = np.abs(gIx * gPx / gP + gIy * gPy / gP + gIz * gPz / gP).astype(np.float32)
    if args.measure == "cos":
        # |grad I . n| <= |grad I| analytically, so the ratio needs no clipping
        # beyond float guard; voxels with no gradient at all have no angle and
        # are set to zero, matching how the volumetric pipeline treats them.
        gI = np.sqrt(gIx * gIx + gIy * gIy + gIz * gIz).astype(np.float32)
        field = np.divide(field, gI, out=np.zeros_like(field), where=gI > 0)
        np.clip(field, 0.0, 1.0, out=field)
    field[~mask] = 0.0
    nib.save(nib.Nifti1Image(field, t1.affine, t1.header), args.out)


if __name__ == "__main__":
    main()
