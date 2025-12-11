\
import os, json, argparse, numpy as np, ants, pandas as pd
from nibabel import Nifti1Image, save as nib_save
from code.bsc.bsc_core import load_img, wm_zscore, directional_bsc

def run_atropos_bsc(t1_path, out_dir, eps=0.05, sigma_mm=1.0):
    os.makedirs(out_dir, exist_ok=True)

    # Preprocess (N4 + resample 1mm) and mask
    t1 = ants.image_read(t1_path)
    t1_n4 = ants.n4_bias_field_correction(t1)
    brain_mask = ants.get_mask(t1_n4)
    t1_iso = ants.resample_image(t1_n4, (1.0,1.0,1.0), use_voxels=False, interp_type=1)
    mask_iso = ants.resample_image(brain_mask, (1.0,1.0,1.0), use_voxels=False, interp_type=0)

    preproc_path = os.path.join(out_dir, 't1w_preproc.nii.gz')
    mask_path = os.path.join(out_dir, 'brain_mask.nii.gz')
    ants.image_write(t1_iso, preproc_path)
    ants.image_write(mask_iso, mask_path)

    # 3-class Atropos
    seg = ants.atropos(a=t1_iso, x=mask_iso, i='kmeans[3]', c='[5,0]', m='[0.1,1x1x1]')
    probs = seg['probabilityimages']

    # Identify GM vs WM by intensity
    means = [float(t1_iso[probs[i]>.6].mean()) if (probs[i]>.6).sum()>0 else 0 for i in range(3)]
    order = np.argsort(means)  # low->high : CSF, GM, WM
    gm_idx, wm_idx = order[1], order[2]

    gm_prob_path = os.path.join(out_dir, 'gm_prob.nii.gz')
    wm_prob_path = os.path.join(out_dir, 'wm_prob.nii.gz')
    ants.image_write(probs[gm_idx], gm_prob_path)
    ants.image_write(probs[wm_idx], wm_prob_path)

    # Compute BSC
    t1_arr, _, spacing = load_img(preproc_path)
    gm_prob, _, _ = load_img(gm_prob_path)
    wm_prob, _, _ = load_img(wm_prob_path)
    brain_mask_arr, _, _ = load_img(mask_path)

    t1_norm, mu, sd = wm_zscore(t1_arr, wm_prob=wm_prob)
    metrics = directional_bsc(t1_norm, gm_prob, brain_mask_arr, spacing, band_eps=eps, sigma_mm=sigma_mm)

    metrics.update(dict(engine='atropos', units='SD/mm', spacing=spacing, mu_wm=mu, sd_wm=sd, t1=t1_path))
    with open(os.path.join(out_dir, 'bsc_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([dict(subject=os.path.basename(out_dir), **metrics)]).to_csv(os.path.join(out_dir, 'subject_metrics.csv'), index=False)
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--t1', required=True, help='Preprocessed or raw T1 (NIfTI).')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--sigma_mm', type=float, default=1.0)
    args = ap.parse_args()
    m = run_atropos_bsc(args.t1, args.out_dir, eps=args.eps, sigma_mm=args.sigma_mm)
    print(json.dumps(m, indent=2))
