\
import argparse, os, pandas as pd
from pathlib import Path
from tqdm import tqdm
from importlib import import_module

def run_batch(manifest_csv, engine, out_root, limit=None, **kwargs):
    df = pd.read_csv(manifest_csv)
    if limit is not None:
        df = df.head(int(limit))
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)

    if engine == 'atropos':
        mod = import_module('code.seg.atropos_bsc')
        for _, row in tqdm(df.iterrows(), total=len(df)):
            t1 = row['path']
            sub_dir = out_root / f"{Path(t1).stem}"
            mod.run_atropos_bsc(str(t1), str(sub_dir), eps=kwargs.get('eps',0.05), sigma_mm=kwargs.get('sigma_mm',1.0))

    elif engine == 'synthseg':
        mod = import_module('code.seg.synthseg_bsc')
        for _, row in tqdm(df.iterrows(), total=len(df)):
            t1 = row['path']
            sub_dir = out_root / f"{Path(t1).stem}"
            gm = kwargs.get('gm_prob'); wm = kwargs.get('wm_prob')
            post = kwargs.get('posteriors'); lbl = kwargs.get('label_map_json')
            mod.run_synthseg_bsc(str(t1), str(sub_dir), gm_prob_path=gm, wm_prob_path=wm,
                                 posteriors_path=post, label_map_json=lbl,
                                 brain_mask_path=kwargs.get('brain_mask'),
                                 eps=kwargs.get('eps',0.05), sigma_mm=kwargs.get('sigma_mm',1.0))

    elif engine == 'freesurfer':
        mod = import_module('code.seg.freesurfer_bsc')
        subjects_dir = kwargs.get('subjects_dir')
        if not subjects_dir:
            raise ValueError('Provide --subjects_dir for FreeSurfer engine')
        for _, row in tqdm(df.iterrows(), total=len(df)):
            subject_id = os.path.splitext(os.path.basename(row['path']))[0]
            sub_dir = out_root / f"{subject_id}"
            mod.run_fs_bsc(subjects_dir, subject_id, t1_mgz=kwargs.get('t1_mgz','mri/brain.mgz'),
                           offsets_mm=tuple(map(float, kwargs.get('offsets','-2,-1,0,1,2').split(','))),
                           out_dir=str(sub_dir))
    else:
        raise ValueError('Unknown engine')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--engine', required=True, choices=['atropos','synthseg','freesurfer'])
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--sigma_mm', type=float, default=1.0)
    ap.add_argument('--subjects_dir')
    ap.add_argument('--t1_mgz', default='mri/brain.mgz')
    ap.add_argument('--offsets', default='-2,-1,0,1,2')
    ap.add_argument('--gm_prob'); ap.add_argument('--wm_prob'); ap.add_argument('--posteriors'); ap.add_argument('--label_map_json'); ap.add_argument('--brain_mask')
    args = ap.parse_args()
    run_batch(args.manifest, args.engine, args.out_root, limit=args.limit,
              eps=args.eps, sigma_mm=args.sigma_mm, subjects_dir=args.subjects_dir,
              t1_mgz=args.t1_mgz, offsets=args.offsets,
              gm_prob=args.gm_prob, wm_prob=args.wm_prob,
              posteriors=args.posteriors, label_map_json=args.label_map_json,
              brain_mask=args.brain_mask)
