\
import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import glob

def merge_labels(bsc_csv_glob, labels_csv, key_cols=('subject',)):
    rows = []
    for f in glob.glob(bsc_csv_glob):
        try:
            df = pd.read_csv(f)
            df['source_file'] = f
            rows.append(df)
        except Exception:
            pass
    bsc = pd.concat(rows, ignore_index=True)
    labels = pd.read_csv(labels_csv)
    merged = pd.merge(bsc, labels, on=list(key_cols), how='inner')
    return merged

def run_classification(df, feature_cols, target_col='label', out_dir='.'):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros_like(y, dtype=np.float32)
    truth = y.copy()
    for tr, te in cv.split(X, y):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[tr], y[tr])
        probs[te] = model.predict_proba(X[te])[:,1]
    auc = roc_auc_score(truth, probs)
    brier = brier_score_loss(truth, probs)
    frac_pos, mean_pred = calibration_curve(truth, probs, n_bins=10)
    plt.figure(); plt.plot(mean_pred, frac_pos, 'o-'); plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title(f'Calibration (AUC={auc:.3f}, Brier={brier:.3f})')
    plt.savefig(out/'calibration.png', dpi=160); plt.close()
    with open(out/'metrics.txt','w') as f:
        f.write(f'AUC: {auc:.4f}\\nBrier: {brier:.4f}\\n')
    print(f'AUC={auc:.4f} Brier={brier:.4f} -> {out}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--bsc_csv_glob', required=True)
    ap.add_argument('--labels_csv', required=True, help='CSV with columns: subject, label (0/1)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--feature_cols', default='BSC_dir')
    args = ap.parse_args()
    feats = [c.strip() for c in args.feature_cols.split(',')]
    df = merge_labels(args.bsc_csv_glob, args.labels_csv, key_cols=('subject',))
    run_classification(df, feats, target_col='label', out_dir=args.out_dir)
