"""
BSC-only adaptation of mri-bsc/code/ml/train_xgb_survival_combined.py: same XGBoost
survival:aft methodology (hyperparameters, 70/30 stratified split + 5-fold stratified CV,
sksurv concordance_index_censored evaluation, signed-log1p -> winsorize -> MinMax feature
preprocessing, train-only robust-variance slope feature selection) but trained on BSC
longitudinal slopes alone -- the T1-morphometry merge is intentionally dropped (scope
decision: replicate the BSC half of the study on the new Trillium cohort without
reproducing raw_t1_analysis/features_all_456.csv-equivalent features).

Usage:
    python3 train_xgb_bsc_survival.py --slopes bsc_longitudinal_slopes.csv \
        --survival survival_labels.csv --out_dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.metrics import concordance_index_censored


def load_and_merge(slopes_path: str, survival_path: str) -> pd.DataFrame:
    slopes = pd.read_csv(slopes_path)
    survival = pd.read_csv(survival_path)

    print(f"  BSC slopes: {len(slopes)} subjects")
    print(f"  Survival:   {len(survival)} subjects")

    merged = survival.merge(slopes, on="subject", how="inner")
    print(f"  After merge: {len(merged)} subjects  "
          f"({merged['event'].sum()} converters, {(merged['event'] == 0).sum()} stable)")
    return merged


def signed_log1p_df(X: pd.DataFrame) -> pd.DataFrame:
    return np.sign(X) * np.log1p(np.abs(X))


def fit_winsor_limits(X: pd.DataFrame, lower_q: float, upper_q: float) -> dict:
    return {"lower": X.quantile(lower_q), "upper": X.quantile(upper_q)}


def apply_winsor_limits(X: pd.DataFrame, limits: dict) -> pd.DataFrame:
    return X.clip(lower=limits["lower"], upper=limits["upper"], axis=1)


def minmax_scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xtr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return Xtr, Xte, scaler


def select_slope_features_train_only(
    X_train: pd.DataFrame,
    top_k: int = 20,
    penalize_regex: str = "nboundary",
    penalty_factor: float = 0.10,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
    quiet: bool = False,
) -> list:
    X_log = signed_log1p_df(X_train)
    limits = fit_winsor_limits(X_log, winsor_q_low, winsor_q_high)
    X_robust = apply_winsor_limits(X_log, limits)

    raw_var = X_robust.var()
    score = raw_var.copy()

    mask = pd.Series(
        score.index.str.contains(penalize_regex, case=False, regex=True), index=score.index
    )
    score.loc[mask] = score.loc[mask] * penalty_factor

    top_k = min(top_k, len(score))
    top_features = score.nlargest(top_k).index.tolist()

    if not quiet:
        print(f"\n{'=' * 80}")
        print("BSC SLOPE FEATURE SELECTION (TRAIN ONLY)")
        print(f"  Selecting top {top_k} BSC slope features")
        print(f"{'=' * 80}")
        for i, feat in enumerate(top_features, 1):
            pen = " (PENALISED)" if mask.loc[feat] else ""
            print(f"  {i:2d}. {feat:45s} var={raw_var[feat]:.6f}  score={score[feat]:.6f}{pen}")

    return top_features


def make_aft_labels(y: pd.DataFrame):
    y_lower = y["time_years"].values.astype(float)
    y_upper = np.where(y["event"].values == 1, y["time_years"].values, np.inf).astype(float)
    return y_lower, y_upper


def c_index(y_df: pd.DataFrame, pred: np.ndarray) -> float:
    event = y_df["event"].values.astype(bool)
    time = y_df["time_years"].values
    ci, *_ = concordance_index_censored(event, time, -pred)
    return float(ci)


def train_xgb_model(X_train, X_test, y_train, y_test, params, n_estimators):
    y_lower_tr, y_upper_tr = make_aft_labels(y_train)
    y_lower_te, y_upper_te = make_aft_labels(y_test)

    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info("label_lower_bound", y_lower_tr)
    dtrain.set_float_info("label_upper_bound", y_upper_tr)

    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info("label_lower_bound", y_lower_te)
    dtest.set_float_info("label_upper_bound", y_upper_te)

    print(f"\nFitting XGBoost AFT (n_estimators={n_estimators}, "
          f"max_depth={params.get('max_depth')}, lr={params.get('learning_rate')}) ...")

    model = xgb.train(
        params, dtrain, num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dtest, "test")], verbose_eval=50,
    )

    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)

    train_c = c_index(y_train, pred_train)
    test_c = c_index(y_test, pred_test)

    metrics = {
        "train_c_index": train_c,
        "test_c_index": test_c,
        "overfitting_gap": round(train_c - test_c, 4),
        "n_features": X_train.shape[1],
        "n_estimators": n_estimators,
    }

    print(f"\n{'=' * 80}")
    print("RESULTS: XGBoost AFT Survival (BSC slopes only)")
    print(f"{'=' * 80}")
    print(f"  Train C-index : {train_c:.4f}")
    print(f"  Test  C-index : {test_c:.4f}")
    print(f"  Overfitting   : {train_c - test_c:.4f}")

    return model, metrics, pred_train, pred_test


def run_5fold_cv(df, slope_cols, y_all, params, n_estimators, top_k,
                  penalize_regex, penalty_factor, winsor_low, winsor_high):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, test_scores = [], []

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())

    print(f"\n{'=' * 80}")
    print("5-FOLD CROSS-VALIDATION")
    print(f"{'=' * 80}")

    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(df.index, y_all["event"]), 1):
        tr_idx = df.index[tr_idx]
        te_idx = df.index[te_idx]
        y_tr = y_all.loc[tr_idx]
        y_te = y_all.loc[te_idx]

        sel = select_slope_features_train_only(
            X_slopes_all.loc[tr_idx], top_k, penalize_regex, penalty_factor,
            winsor_low, winsor_high, quiet=True,
        )

        X_tr = X_slopes_all.loc[tr_idx, sel]
        X_te = X_slopes_all.loc[te_idx, sel]

        X_tr_log = signed_log1p_df(X_tr)
        X_te_log = signed_log1p_df(X_te)
        lims = fit_winsor_limits(X_tr_log, winsor_low, winsor_high)
        X_tr_r = apply_winsor_limits(X_tr_log, lims)
        X_te_r = apply_winsor_limits(X_te_log, lims)
        X_tr_s, X_te_s, _ = minmax_scale_train_test(X_tr_r, X_te_r)

        y_lower_tr, y_upper_tr = make_aft_labels(y_tr)
        y_lower_te, y_upper_te = make_aft_labels(y_te)

        dtr = xgb.DMatrix(X_tr_s)
        dtr.set_float_info("label_lower_bound", y_lower_tr)
        dtr.set_float_info("label_upper_bound", y_upper_tr)
        dte = xgb.DMatrix(X_te_s)
        dte.set_float_info("label_lower_bound", y_lower_te)
        dte.set_float_info("label_upper_bound", y_upper_te)

        m = xgb.train(params, dtr, num_boost_round=n_estimators, verbose_eval=False)

        c_tr = c_index(y_tr, m.predict(dtr))
        c_te = c_index(y_te, m.predict(dte))
        train_scores.append(c_tr)
        test_scores.append(c_te)
        print(f"  Fold {fold_i}  train={c_tr:.4f}  test={c_te:.4f}  gap={c_tr - c_te:.4f}")

    gaps = [tr - te for tr, te in zip(train_scores, test_scores)]
    print(f"\n  Mean  train={np.mean(train_scores):.4f}  test={np.mean(test_scores):.4f}  "
          f"gap={np.mean(gaps):.4f}")
    print(f"  Std   train={np.std(train_scores):.4f}  test={np.std(test_scores):.4f}  "
          f"gap={np.std(gaps):.4f}")

    return {"train": train_scores, "test": test_scores, "gap": gaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slopes", required=True, help="bsc_longitudinal_slopes.csv")
    parser.add_argument("--survival", required=True, help="survival_labels.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--penalize_regex", type=str, default="nboundary")
    parser.add_argument("--penalty_factor", type=float, default=0.10)
    parser.add_argument("--winsor_low", type=float, default=0.01)
    parser.add_argument("--winsor_high", type=float, default=0.99)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}\nLOADING DATA\n{'=' * 80}")
    df = load_and_merge(args.slopes, args.survival)

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().mean() > 0.8]

    y_all = df[["time_years", "event"]].copy()

    print(f"\n{'=' * 80}\nDATA SPLIT (70/30, stratified by event)\n{'=' * 80}")
    print(f"  Total: {len(df)}  |  events: {y_all['event'].sum()} ({y_all['event'].mean() * 100:.1f}%)")

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=y_all["event"]
    )
    y_train = y_all.loc[idx_train]
    y_test = y_all.loc[idx_test]
    print(f"  Train: {len(idx_train)}  |  events: {y_train['event'].sum()}")
    print(f"  Test:  {len(idx_test)}  |  events: {y_test['event'].sum()}")

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())
    selected_slopes = select_slope_features_train_only(
        X_slopes_all.loc[idx_train], top_k=args.top_k, penalize_regex=args.penalize_regex,
        penalty_factor=args.penalty_factor, winsor_q_low=args.winsor_low, winsor_q_high=args.winsor_high,
    )

    X_all = X_slopes_all[selected_slopes]

    print(f"\n{'=' * 80}\nBSC SLOPE FEATURES: {len(selected_slopes)} total\n{'=' * 80}")

    X_train_raw = X_all.loc[idx_train].copy()
    X_test_raw = X_all.loc[idx_test].copy()

    X_train_log = signed_log1p_df(X_train_raw)
    X_test_log = signed_log1p_df(X_test_raw)
    limits = fit_winsor_limits(X_train_log, args.winsor_low, args.winsor_high)
    X_train_robust = apply_winsor_limits(X_train_log, limits)
    X_test_robust = apply_winsor_limits(X_test_log, limits)
    X_train_scaled, X_test_scaled, _ = minmax_scale_train_test(X_train_robust, X_test_robust)

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    model, metrics, pred_train, pred_test = train_xgb_model(
        X_train_scaled, X_test_scaled, y_train, y_test, params, args.n_estimators
    )

    model.save_model(str(out_dir / "xgb_model.json"))
    with open(out_dir / "xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "xgb_features.txt", "w") as f:
        for feat in selected_slopes:
            f.write(f"{feat}\n")

    test_df = pd.DataFrame({
        "subject": df.loc[idx_test, "subject"].values,
        "true_time": y_test["time_years"].values,
        "event": y_test["event"].values,
        "predicted_risk": pred_test,
    })
    test_df.to_csv(out_dir / "xgb_predictions.csv", index=False)

    cv_results = run_5fold_cv(
        df, slope_cols, y_all, params, args.n_estimators, args.top_k,
        args.penalize_regex, args.penalty_factor, args.winsor_low, args.winsor_high,
    )
    with open(out_dir / "xgb_cv_results.json", "w") as f:
        json.dump({k: [round(v, 4) for v in vs] for k, vs in cv_results.items()}, f, indent=2)

    with open(out_dir / "xgb_summary.txt", "w") as f:
        f.write("XGBoost AFT Survival -- BSC Slopes Only (Trillium cohort)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Subjects: {len(df)}  |  converters: {y_all['event'].sum()}\n")
        f.write(f"BSC slope features selected: {len(selected_slopes)}\n\n")
        f.write("XGBoost params:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  n_estimators: {args.n_estimators}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Train C-index : {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test  C-index : {metrics['test_c_index']:.4f}\n")
        f.write(f"  Overfitting   : {metrics['overfitting_gap']:.4f}\n")
        f.write(f"  5-fold CV mean test C-index: {np.mean(cv_results['test']):.4f} "
                f"+/- {np.std(cv_results['test']):.4f}\n")

    print(f"\nResults saved to: {out_dir}")
    print(f"\n{'=' * 80}")
    print(f"  TRAIN C-index : {metrics['train_c_index']:.4f}")
    print(f"  TEST  C-index : {metrics['test_c_index']:.4f}")
    print(f"  5-fold CV mean test C-index: {np.mean(cv_results['test']):.4f} "
          f"+/- {np.std(cv_results['test']):.4f}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
