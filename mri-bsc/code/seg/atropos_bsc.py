import os, json, argparse, tempfile, numpy as np, ants, pandas as pd
from pathlib import Path
from nibabel import Nifti1Image, save as nib_save

from code.bsc.bsc_core import load_img, wm_zscore, directional_bsc_voxelwise
from code.io.s3 import upload_file


def _is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def run_atropos_bsc(t1_path, out_dir, eps=0.05, sigma_mm=1.0, work_dir=None):
    """
    Always run locally inside work_dir, then upload results to out_dir if out_dir is S3.
    """
    is_s3_out = _is_s3_path(out_dir)

    # ✅ Local work directory (you control it)
    if work_dir is None:
        local_work_dir = tempfile.mkdtemp()
    else:
        os.makedirs(work_dir, exist_ok=True)
        local_work_dir = tempfile.mkdtemp(dir=work_dir)

    os.makedirs(local_work_dir, exist_ok=True)

    # --- read + N4 ---
    t1 = ants.image_read(str(t1_path))
    t1_n4 = ants.n4_bias_field_correction(t1)
    t1_n4.set_spacing(t1.spacing)

    # ✅ Better mask (Otsu + cleanup)
    brain_mask = ants.threshold_image(t1_n4, "Otsu", 1)
    brain_mask = ants.iMath(brain_mask, "GetLargestComponent")
    brain_mask = ants.iMath(brain_mask, "FillHoles")

    # ✅ Apply mask before resample / Atropos
    t1_brain = t1_n4 * brain_mask

    # --- resample to 1mm ---
    t1_iso = ants.resample_image(t1_brain, (1.0, 1.0, 1.0), use_voxels=False, interp_type=1)
    mask_iso = ants.resample_image(brain_mask, (1.0, 1.0, 1.0), use_voxels=False, interp_type=0)

    # --- write preproc + mask locally ---
    preproc_path = os.path.join(local_work_dir, "t1w_preproc.nii.gz")
    mask_path = os.path.join(local_work_dir, "brain_mask.nii.gz")
    ants.image_write(t1_iso, preproc_path)
    ants.image_write(mask_iso, mask_path)

    # --- Atropos segmentation ---
    seg = ants.atropos(a=t1_iso, x=mask_iso, i="kmeans[3]", c="[5,0]", m="[0.1,1x1x1]")
    probs = seg["probabilityimages"]

    # Identify GM vs WM by intensity
    means = [
        float(t1_iso[probs[i] > 0.6].mean()) if (probs[i] > 0.6).sum() > 0 else 0
        for i in range(3)
    ]
    order = np.argsort(means)  # CSF, GM, WM
    gm_idx, wm_idx = order[1], order[2]

    gm_prob_local = os.path.join(local_work_dir, "gm_prob.nii.gz")
    wm_prob_local = os.path.join(local_work_dir, "wm_prob.nii.gz")
    ants.image_write(probs[gm_idx], gm_prob_local)
    ants.image_write(probs[wm_idx], wm_prob_local)

    # --- voxel-wise BSC maps ---
    t1_arr, aff, spacing = load_img(preproc_path)
    gm_prob, _, _ = load_img(gm_prob_local)
    wm_prob, _, _ = load_img(wm_prob_local)
    brain_mask_arr, _, _ = load_img(mask_path)

    t1_norm, mu, sd = wm_zscore(t1_arr, wm_prob=wm_prob)

    bsc_dir_map, bsc_mag_map, band_mask, metrics = directional_bsc_voxelwise(
        t1_norm,
        gm_prob,
        brain_mask_arr,
        spacing,
        band_eps=eps,
        sigma_mm=sigma_mm,
    )

    # save maps locally
    bsc_dir_map_path = os.path.join(local_work_dir, "bsc_dir_map.nii.gz")
    bsc_mag_map_path = os.path.join(local_work_dir, "bsc_mag_map.nii.gz")
    band_mask_path = os.path.join(local_work_dir, "boundary_band_mask.nii.gz")

    nib_save(Nifti1Image(bsc_dir_map.astype(np.float32), aff), bsc_dir_map_path)
    nib_save(Nifti1Image(bsc_mag_map.astype(np.float32), aff), bsc_mag_map_path)
    nib_save(Nifti1Image(band_mask.astype(np.uint8), aff), band_mask_path)

    # metrics files
    metrics.update(
        dict(
            engine="atropos",
            units="SD/mm",
            spacing=spacing,
            mu_wm=mu,
            sd_wm=sd,
            eps=eps,
            sigma_mm=sigma_mm,
            t1=str(t1_path),
        )
    )

    metrics_json_path = os.path.join(local_work_dir, "bsc_metrics.json")
    metrics_csv_path = os.path.join(local_work_dir, "subject_metrics.csv")

    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)

    if is_s3_out:
        for filename in [
            "t1w_preproc.nii.gz",
            "brain_mask.nii.gz",
            "gm_prob.nii.gz",
            "wm_prob.nii.gz",
            "bsc_dir_map.nii.gz",
            "bsc_mag_map.nii.gz",
            "boundary_band_mask.nii.gz",
            "bsc_metrics.json",
            "subject_metrics.csv",
        ]:
            local_file = os.path.join(local_work_dir, filename)
            if os.path.exists(local_file):
                upload_file(Path(local_file), f"{out_dir.rstrip('/')}/{filename}")

        import shutil
        shutil.rmtree(local_work_dir, ignore_errors=True)

    return metrics
