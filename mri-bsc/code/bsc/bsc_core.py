\
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def load_img(path):
    nii = nib.load(path)
    data = nii.get_fdata(dtype=np.float32)
    aff  = nii.affine
    zooms = nib.affines.voxel_sizes(aff)
    return data, aff, tuple(float(z) for z in zooms[:3])

def wm_zscore(t1_arr, wm_prob=None, brain_mask=None, thr=0.5):
    """
    Normalize intensities using WM voxels if available, else brain.
    Returns normalized array, mean, std used.
    """
    if wm_prob is not None:
        mask = wm_prob >= thr
    elif brain_mask is not None:
        mask = brain_mask > 0
    else:
        raise ValueError("Need wm_prob or brain_mask for normalization")
    vals = t1_arr[mask]
    mu = float(np.mean(vals)); sd = float(np.std(vals) + 1e-8)
    return (t1_arr - mu) / sd, mu, sd

def gradient_phys(arr, spacing):
    dx, dy, dz = spacing
    gx, gy, gz = np.gradient(arr, dx, dy, dz)
    return gx, gy, gz

def directional_bsc(t1_norm, gm_prob, brain_mask, spacing, band_eps=0.05, sigma_mm=1.0):
    """
    Probability-based BSC (directional): |∇Iσ · n̂| at the GM/WM boundary band (P≈0.5).
    n̂ = ∇P / ||∇P|| where P = gm_prob (smoothed).
    Returns dict with BSC_dir, BSC_mag, Nboundary.
    """
    # 1) Boundary band at ~0.5 probability
    band = (gm_prob > (0.5 - band_eps)) & (gm_prob < (0.5 + band_eps))
    if brain_mask is not None:
        band &= (brain_mask > 0)

    # 2) Smooth before gradients (convert mm sigma to voxel sigma per axis)
    dx, dy, dz = spacing
    sig = (sigma_mm / max(dx,1e-6), sigma_mm / max(dy,1e-6), sigma_mm / max(dz,1e-6))
    I_s = gaussian_filter(t1_norm, sigma=sig, mode="nearest")
    P_s = gaussian_filter(gm_prob, sigma=sig, mode="nearest")

    # 3) Physical gradients
    gIx, gIy, gIz = gradient_phys(I_s, spacing)
    gPx, gPy, gPz = gradient_phys(P_s, spacing)

    # 4) Unit normal from P
    gP_norm = np.sqrt(gPx*gPx + gPy*gPy + gPz*gPz) + 1e-8
    nx, ny, nz = gPx/gP_norm, gPy/gP_norm, gPz/gP_norm

    # 5) Directional derivative of intensity along n̂
    dI_dn = gIx*nx + gIy*ny + gIz*nz

    vals = np.abs(dI_dn[band])
    if vals.size == 0:
        return dict(BSC_dir=np.nan, BSC_mag=np.nan, Nboundary=0)

    BSC_dir = float(np.median(vals))
    gI_mag = np.sqrt(gIx*gIx + gIy*gIy + gIz*gIz)
    BSC_mag = float(np.median(gI_mag[band]))
    return dict(BSC_dir=BSC_dir, BSC_mag=BSC_mag, Nboundary=int(band.sum()))


def directional_bsc_voxelwise(t1_norm, gm_prob, brain_mask, spacing, band_eps=0.05, sigma_mm=1.0):
    """
    Returns:
      bsc_dir_map: voxel-wise |∇Iσ · n̂| masked to boundary band
      bsc_mag_map: voxel-wise |∇Iσ| masked to boundary band
      band_mask: boundary band mask (uint8)
      summary dict (BSC_dir, BSC_mag, Nboundary)
    """
    # 1) Boundary band
    band = (gm_prob > (0.5 - band_eps)) & (gm_prob < (0.5 + band_eps))
    if brain_mask is not None:
        band &= (brain_mask > 0)

    # 2) Smooth before gradients (mm -> voxel sigma)
    dx, dy, dz = spacing
    sig = (
        sigma_mm / max(dx, 1e-6),
        sigma_mm / max(dy, 1e-6),
        sigma_mm / max(dz, 1e-6),
    )

    I_s = gaussian_filter(t1_norm, sigma=sig, mode="nearest")
    P_s = gaussian_filter(gm_prob, sigma=sig, mode="nearest")

    # 3) Physical gradients
    gIx, gIy, gIz = gradient_phys(I_s, spacing)
    gPx, gPy, gPz = gradient_phys(P_s, spacing)

    # 4) Unit normal from P
    gP_norm = np.sqrt(gPx * gPx + gPy * gPy + gPz * gPz) + 1e-8
    nx, ny, nz = gPx / gP_norm, gPy / gP_norm, gPz / gP_norm

    # 5) Directional derivative
    dI_dn = gIx * nx + gIy * ny + gIz * nz
    bsc_dir_map = np.abs(dI_dn).astype(np.float32)

    # 6) Gradient magnitude
    gI_mag = np.sqrt(gIx * gIx + gIy * gIy + gIz * gIz).astype(np.float32)
    bsc_mag_map = gI_mag

    # 7) Mask outside boundary band
    band_mask = band.astype(np.uint8)
    bsc_dir_map[~band] = 0.0
    bsc_mag_map[~band] = 0.0

    vals_dir = bsc_dir_map[band]
    vals_mag = bsc_mag_map[band]

    if vals_dir.size == 0:
        summary = dict(BSC_dir=np.nan, BSC_mag=np.nan, Nboundary=0)
    else:
        summary = dict(
            BSC_dir=float(np.median(vals_dir)),
            BSC_mag=float(np.median(vals_mag)),
            Nboundary=int(band.sum()),
        )

    return bsc_dir_map, bsc_mag_map, band_mask, summary
