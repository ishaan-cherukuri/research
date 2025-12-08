import argparse, os
from pathlib import Path
import pandas as pd
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, N4BiasFieldCorrectiond, SaveImaged
from tqdm import tqdm

def run_preproc(manifest_csv, out_dir, n=None):
    df = pd.read_csv(manifest_csv)
    if n is not None:
        df = df.head(int(n))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx = Compose([
        LoadImaged(keys=['image']),
        EnsureChannelFirstd(keys=['image']),
        Orientationd(keys=['image'], axcodes='RAS'),
        Spacingd(keys=['image'], pixdim=(1.0,1.0,1.0), mode=('bilinear',)),
        N4BiasFieldCorrectiond(keys=['image']),
        SaveImaged(keys=['image'], output_dir=str(out_dir), output_postfix='preproc', separate_folder=False),
    ])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row['path']
        _ = tx({'image': path})
    print(f"Preprocessed {len(df)} scans -> {out_dir}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n', type=int, default=None)
    args = ap.parse_args()
    run_preproc(args.manifest, args.out_dir, n=args.n)
