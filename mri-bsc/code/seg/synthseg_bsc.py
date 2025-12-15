import os, json, argparse, tempfile, numpy as np, nibabel as nib, pandas as pd
from pathlib import Path
from bsc.bsc_core import load_img, wm_zscore, directional_bsc
from code.io.s3 import upload_file


def _is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URI."""
    return isinstance(path, str) and path.startswith("s3://")


def load_posteriors_stack(post_path):
    nii = nib.load(post_path)
    arr = nii.get_fdata(dtype=np.float32)
    return arr, nii


def combine_gm_wm_from_posteriors(post_arr, label_map=None):
    """
    Combine posterior channels into GM and WM. label_map is a dict {label_id: "GM"/"WM"}.
    """
    if label_map is None:
        raise ValueError(
            "Provide gm_prob_path/wm_prob_path or a label_map for posteriors."
        )
    gm_list, wm_list = [], []
    K = post_arr.shape[-1]
    for k_str, role in label_map.items():
        try:
            lab_id = int(k_str)
        except Exception:
            continue
        if lab_id < 0 or lab_id >= K:
            continue
        if role.upper() == "GM":
            gm_list.append(post_arr[..., lab_id])
        elif role.upper() == "WM":
            wm_list.append(post_arr[..., lab_id])
    if not gm_list or not wm_list:
        raise ValueError("label_map does not produce valid GM/WM sets.")
    gm_prob = np.clip(np.sum(gm_list, axis=0), 0, 1)
    wm_prob = np.clip(np.sum(wm_list, axis=0), 0, 1)
    return gm_prob, wm_prob


def run_synthseg_bsc(
    t1_path,
    out_dir,
    gm_prob_path=None,
    wm_prob_path=None,
    posteriors_path=None,
    label_map_json=None,
    brain_mask_path=None,
    eps=0.05,
    sigma_mm=1.0,
):
    is_s3_out = _is_s3_path(out_dir)
    local_work_dir = out_dir if not is_s3_out else tempfile.mkdtemp()

    os.makedirs(local_work_dir, exist_ok=True)

    t1_arr, _, spacing = load_img(t1_path)

    if posteriors_path is not None:
        post, post_nii = load_posteriors_stack(posteriors_path)
        label_map = None
        if label_map_json and os.path.exists(label_map_json):
            label_map = json.loads(open(label_map_json, "r").read())
        gm_prob, wm_prob = combine_gm_wm_from_posteriors(post, label_map=label_map)
        nib.save(
            nib.Nifti1Image(gm_prob.astype(np.float32), post_nii.affine),
            os.path.join(local_work_dir, "gm_prob.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(wm_prob.astype(np.float32), post_nii.affine),
            os.path.join(local_work_dir, "wm_prob.nii.gz"),
        )
    else:
        if gm_prob_path is None or wm_prob_path is None:
            raise ValueError(
                "Provide posteriors+label_map_json OR gm_prob_path and wm_prob_path."
            )
        gm_prob, _, _ = load_img(gm_prob_path)
        wm_prob, _, _ = load_img(wm_prob_path)

    if brain_mask_path and os.path.exists(brain_mask_path):
        brain_mask, _, _ = load_img(brain_mask_path)
    else:
        brain_mask = ((gm_prob + wm_prob) > 0.1).astype(np.uint8)

    t1_norm, mu, sd = wm_zscore(t1_arr, wm_prob=wm_prob)
    metrics = directional_bsc(
        t1_norm, gm_prob, brain_mask, spacing, band_eps=eps, sigma_mm=sigma_mm
    )

    metrics.update(
        dict(
            engine="synthseg",
            units="SD/mm",
            spacing=spacing,
            mu_wm=mu,
            sd_wm=sd,
            t1=t1_path,
        )
    )

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
            "gm_prob.nii.gz",
            "wm_prob.nii.gz",
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
    ap.add_argument("--t1", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gm_prob")
    ap.add_argument("--wm_prob")
    ap.add_argument("--posteriors")
    ap.add_argument("--label_map_json")
    ap.add_argument("--brain_mask")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sigma_mm", type=float, default=1.0)
    args = ap.parse_args()

    m = run_synthseg_bsc(
        t1_path=args.t1,
        out_dir=args.out_dir,
        gm_prob_path=args.gm_prob,
        wm_prob_path=args.wm_prob,
        posteriors_path=args.posteriors,
        label_map_json=args.label_map_json,
        brain_mask_path=args.brain_mask,
        eps=args.eps,
        sigma_mm=args.sigma_mm,
    )
    print(json.dumps(m, indent=2))
