import os, json, argparse, tempfile, numpy as np, ants, pandas as pd
from pathlib import Path
from nibabel import Nifti1Image, save as nib_save
from code.bsc.bsc_core import load_img, wm_zscore, directional_bsc_voxelwise
from code.io.s3 import upload_file, parse_s3_uri


def _is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URI."""
    return isinstance(path, str) and path.startswith("s3://")


def _write_ants_image(image, target_path: str):
    """Write ANTsPy image, uploading to S3 if needed."""
    if _is_s3_path(target_path):
        # Write to temporary file, then upload
        _, ext = os.path.splitext(target_path)
        if target_path.endswith(".nii.gz"):
            suffix = ".nii.gz"
        else:
            suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        try:
            ants.image_write(image, tmp_path)
            upload_file(Path(tmp_path), target_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        ants.image_write(image, target_path)


def run_atropos_bsc(t1_path, out_dir, eps=0.05, sigma_mm=1.0):
    is_s3_out = _is_s3_path(out_dir)
    local_work_dir = out_dir if not is_s3_out else tempfile.mkdtemp()

    os.makedirs(local_work_dir, exist_ok=True)

    # Preprocess (N4 + resample 1mm) and mask
    t1 = ants.image_read(str(t1_path))
    t1_n4 = ants.n4_bias_field_correction(t1)
    # Preserve spacing metadata from original image to avoid ITK errors
    t1_n4.set_spacing(t1.spacing)
    brain_mask = ants.get_mask(t1_n4)
    print(f"Image spacing: {t1_n4.spacing}")
    t1_iso = ants.resample_image(
        t1_n4, (1.0, 1.0, 1.0), use_voxels=False, interp_type=1
    )
    print(f"Image spacing: {brain_mask.spacing}")
    mask_iso = ants.resample_image(
        brain_mask, (1.0, 1.0, 1.0), use_voxels=False, interp_type=0
    )

    preproc_path = os.path.join(local_work_dir, "t1w_preproc.nii.gz")
    mask_path = os.path.join(local_work_dir, "brain_mask.nii.gz")
    ants.image_write(t1_iso, preproc_path)
    ants.image_write(mask_iso, mask_path)

    # 3-class Atropos
    seg = ants.atropos(a=t1_iso, x=mask_iso, i="kmeans[3]", c="[5,0]", m="[0.1,1x1x1]")
    probs = seg["probabilityimages"]

    # Identify GM vs WM by intensity
    means = [
        float(t1_iso[probs[i] > 0.6].mean()) if (probs[i] > 0.6).sum() > 0 else 0
        for i in range(3)
    ]
    order = np.argsort(means)  # low->high : CSF, GM, WM
    gm_idx, wm_idx = order[1], order[2]

    gm_prob_local = os.path.join(local_work_dir, "gm_prob.nii.gz")
    wm_prob_local = os.path.join(local_work_dir, "wm_prob.nii.gz")
    ants.image_write(probs[gm_idx], gm_prob_local)
    ants.image_write(probs[wm_idx], wm_prob_local)

    # Compute BSC
    t1_arr, _, spacing = load_img(preproc_path)
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

    metrics.update(
        dict(
            engine="atropos",
            units="SD/mm",
            spacing=spacing,
            mu_wm=mu,
            sd_wm=sd,
            t1=str(t1_path),
        )
    )
    # Save voxel-wise maps locally (same affine as preproc)
    _, aff, _ = load_img(preproc_path)

    bsc_dir_map_path = os.path.join(local_work_dir, "bsc_dir_map.nii.gz")
    bsc_mag_map_path = os.path.join(local_work_dir, "bsc_mag_map.nii.gz")
    band_mask_path = os.path.join(local_work_dir, "boundary_band_mask.nii.gz")

    nib_save(Nifti1Image(bsc_dir_map.astype(np.float32), aff), bsc_dir_map_path)
    nib_save(Nifti1Image(bsc_mag_map.astype(np.float32), aff), bsc_mag_map_path)
    nib_save(Nifti1Image(band_mask.astype(np.uint8), aff), band_mask_path)
    
    # Save metrics and upload to S3 if needed
    metrics_json_path = os.path.join(local_work_dir, "bsc_metrics.json")
    metrics_csv_path = os.path.join(local_work_dir, "subject_metrics.csv")

    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([dict(subject=os.path.basename(local_work_dir), **metrics)]).to_csv(
        metrics_csv_path, index=False
    )

    # If output was S3, upload all files and clean up temp directory
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
                s3_path = os.path.join(out_dir, filename)
                upload_file(Path(local_file), s3_path)

        # Clean up temporary directory
        import shutil

        shutil.rmtree(local_work_dir, ignore_errors=True)

    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True, help="Preprocessed or raw T1 (NIfTI).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    args = ap.parse_args()
    m = run_atropos_bsc(args.t1, args.out_dir, eps=args.eps, sigma_mm=args.sigma_mm)
    print(json.dumps(m, indent=2))
