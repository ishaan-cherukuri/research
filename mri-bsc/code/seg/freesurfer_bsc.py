\
import os, json, argparse, subprocess, tempfile, numpy as np, nibabel as nib, pandas as pd

def run_vol2surf_series(fs_subjects_dir, subject_id, hemi, vol_path, offsets_mm):
    env = os.environ.copy()
    env['SUBJECTS_DIR'] = fs_subjects_dir
    out = {}
    for off in offsets_mm:
        with tempfile.NamedTemporaryFile(suffix='.mgh', delete=False) as tmp:
            tmp_out = tmp.name
        cmd = [
            'mri_vol2surf',
            '--hemi', hemi,
            '--src', vol_path,
            '--out', tmp_out,
            '--surf', 'white',
            '--projdist', str(off),
            '--sd', fs_subjects_dir,
            '--surfreg', 'sphere.reg',
            '--cortex'
        ]
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        m = nib.load(tmp_out).get_fdata().squeeze()
        out[float(off)] = np.asarray(m, dtype=np.float32)
        os.remove(tmp_out)
    return out

def central_slope(vals_at_offsets, offsets_mm):
    offs = np.array(sorted(offsets_mm), dtype=np.float32)  # e.g., [-2,-1,0,1,2]
    X = np.vstack([offs, np.ones_like(offs)]).T  # [K,2]
    XtX_inv = np.linalg.inv(X.T @ X)
    # Build Y[K, Nverts]
    N = None
    for k, off in enumerate(offs):
        v = vals_at_offsets[float(off)]
        if N is None:
            N = v.shape[0]
            Y = np.zeros((offs.size, N), dtype=np.float32)
        Y[k, :] = v
    m_b = XtX_inv @ X.T @ Y   # [2, N]
    slopes = m_b[0, :]        # intensity change per mm along normal
    return slopes

def run_fs_bsc(fs_subjects_dir, subject_id, t1_mgz='mri/brain.mgz',
               offsets_mm=(-2.0, -1.0, 0.0, 1.0, 2.0),
               out_dir='.'):
    os.makedirs(out_dir, exist_ok=True)
    vol_path = os.path.join(fs_subjects_dir, subject_id, t1_mgz)

    all_slopes = []
    hemi_metrics = {}
    for hemi in ('lh','rh'):
        vals = run_vol2surf_series(fs_subjects_dir, subject_id, hemi, vol_path, list(offsets_mm))
        slopes = central_slope(vals, list(offsets_mm))
        all_slopes.append(slopes)
        bsc_surf = float(np.median(np.abs(slopes[np.isfinite(slopes)])))
        hemi_metrics[hemi] = dict(BSC_surf=bsc_surf, Nverts=int(slopes.size))

    all_slopes = np.concatenate(all_slopes)
    BSC_surf_both = float(np.median(np.abs(all_slopes[np.isfinite(all_slopes)])))

    metrics = dict(engine='freesurfer', subject=subject_id,
                   offsets_mm=list(offsets_mm),
                   lh=hemi_metrics['lh'], rh=hemi_metrics['rh'],
                   BSC_surf_global=BSC_surf_both,
                   units='SD/mm',
                   vol=vol_path)
    with open(os.path.join(out_dir, 'bsc_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([dict(subject=subject_id, **metrics)]).to_csv(os.path.join(out_dir, 'subject_metrics.csv'), index=False)
    return metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects_dir', required=True)
    ap.add_argument('--subject_id', required=True)
    ap.add_argument('--t1_mgz', default='mri/brain.mgz')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--offsets', default='-2,-1,0,1,2')
    args = ap.parse_args()
    offs = tuple(float(x) for x in args.offsets.split(','))
    m = run_fs_bsc(args.subjects_dir, args.subject_id, t1_mgz=args.t1_mgz,
                   offsets_mm=offs, out_dir=args.out_dir)
    print(json.dumps(m, indent=2))
