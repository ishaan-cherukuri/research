\
import argparse, glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_subject_metrics(csv_glob):
    files = []
    for pat in (csv_glob if isinstance(csv_glob, list) else [csv_glob]):
        files.extend(glob.glob(pat))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f
            rows.append(df)
        except Exception:
            pass
    if not rows:
        raise RuntimeError('No subject_metrics.csv found')
    big = pd.concat(rows, ignore_index=True)
    return big

def plot_by_site(df, out_dir, bsc_col='BSC_dir'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if 'sitekey' not in df.columns:
        df['sitekey'] = 'unknown'

    plt.figure(figsize=(10,6))
    order = sorted(df['sitekey'].unique())
    data = [df[df['sitekey']==s][bsc_col].dropna().values for s in order]
    plt.violinplot(data, showmedians=True)
    plt.xticks(np.arange(1, len(order)+1), order, rotation=45, ha='right')
    plt.ylabel(f'{bsc_col} (SD/mm)')
    plt.title('BSC by site')
    plt.tight_layout()
    plt.savefig(out / f'bsc_by_site_{bsc_col}.png', dpi=160)
    plt.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bsc_csv_glob', required=True, help='e.g., "data/derivatives/bsc/*/*/*/subject_metrics.csv"')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--bsc_col', default='BSC_dir')
    args = ap.parse_args()
    df = load_subject_metrics(args.bsc_csv_glob)
    plot_by_site(df, args.out_dir, bsc_col=args.bsc_col)
