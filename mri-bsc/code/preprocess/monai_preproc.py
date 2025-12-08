import argparse, os
from pathlib import Path
import pandas as pd
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, SaveImaged)

try:
    from monai.transforms import N4BiasFieldCorrectiond
    USE_DICT_N4 = True
except ImportError:
    from monai.transforms import N4BiasFieldCorrection
    USE_DICT_N4 = False
from tqdm import tqdm

def run_preproc(manifest_csv, out_dir, n=None):
    df = pd.read_csv(manifest_csv)
    if n is not None:
        df = df.head(int(n))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tx = [
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=('bilinear',)),
    ]

    if USE_DICT_N4:
        tx.append(N4BiasFieldCorrectiond(keys=["image"]))
    else:
        tx.append(N4BiasFieldCorrection())

    tx.append(SaveImaged(keys=["image"], output_dir=out_dir))

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
