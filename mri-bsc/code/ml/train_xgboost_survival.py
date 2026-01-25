"""code.ml.train_xgboost_survival

Train XGBoost survival model for AD conversion prediction.

Uses XGBoost with survival objective (AFT loss) for time-to-event prediction.
Compares directly with parametric AFT models using same metrics.

Metrics reported:
  - C-index: concordance between predicted and actual survival times
  - MSE: mean squared error on survival times
  - MAE: mean absolute error on survival times

Usage:
  python3 -m code.ml.train_xgboost_survival \
    --data data/ml/survival/time_to_conversion.csv \
    --out_dir data/ml/results \
    --top_k 30

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_survival_data(csv_path: str) -> pd.DataFrame:
    """Load and validate survival data."""
    df = pd.read_csv(csv_path)

    required = {"subject", "event", "time_years"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["time_years"] > 0].copy()

    print(f"[INFO] Loaded {len(df)} subjects")
    print(f"  Events: {df['event'].sum()}")
    print(f"  Censored: {(1 - df['event']).sum()}")
    print(f"  Mean follow-up: {df['time_years'].mean():.2f} years")
    print(f"  Median follow-up: {df['time_years'].median():.2f} years")

    return df


def select_features(
    df: pd.DataFrame, exclude_cols: set[str], top_k: Optional[int] = None
) -> list[str]:
    """Select feature columns by variance."""
    feat_cols = [c for c in df.columns if c not in exclude_cols]

    if top_k and len(feat_cols) > top_k:
        print(f"[INFO] Selecting top {top_k} features by variance...")
        variances = df[feat_cols].var()
        valid = variances[variances > 0].sort_values(ascending=False)
        feat_cols = valid.head(top_k).index.tolist()
        print(f"  Selected: {len(feat_cols)} features")

    return feat_cols


def prepare_data(
    df: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    """Prepare features for training."""
    df_clean = df.copy()

    # Handle missing/infinite
    for col in feat_cols:
        if df_clean[col].isna().any() or np.isinf(df_clean[col]).any():
            median = df_clean[col].replace([np.inf, -np.inf], np.nan).median()
            df_clean[col] = (
                df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(median)
            )

    # Standardize
    scaler = StandardScaler()
    df_clean[feat_cols] = scaler.fit_transform(df_clean[feat_cols])
    print("[INFO] Features standardized")

    return df_clean, scaler


def train_xgboost_survival(
    X_train: np.ndarray,
    y_train_lower: np.ndarray,
    y_train_upper: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val_lower: Optional[np.ndarray] = None,
    y_val_upper: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
):
    """Train XGBoost AFT survival model."""

    default_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    if params:
        default_params.update(params)

    # Create DMatrix with survival labels
    # For AFT: lower = upper = observed time for events, lower < upper for censored
    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info("label_lower_bound", y_train_lower)
    dtrain.set_float_info("label_upper_bound", y_train_upper)

    evals = [(dtrain, "train")]

    if X_val is not None:
        dval = xgb.DMatrix(X_val)
        dval.set_float_info("label_lower_bound", y_val_lower)
        dval.set_float_info("label_upper_bound", y_val_upper)
        evals.append((dval, "val"))

    print("[INFO] Training XGBoost AFT model...")

    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    print(f"[INFO] Training complete (best iteration: {model.best_iteration})")

    return model


def evaluate_model(
    model, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray, name: str = "Test"
) -> dict:
    """Evaluate model with survival metrics."""

    dmat = xgb.DMatrix(X)
    predictions = model.predict(dmat)

    # C-index (higher is better, 0.5 = random, 1.0 = perfect)
    c_index = concordance_index(y_time, -predictions, y_event)

    # MSE and MAE on predicted vs actual times (only for events)
    event_mask = y_event == 1
    if event_mask.sum() > 0:
        mse = mean_squared_error(y_time[event_mask], predictions[event_mask])
        mae = mean_absolute_error(y_time[event_mask], predictions[event_mask])
        rmse = np.sqrt(mse)
    else:
        mse = mae = rmse = float("nan")

    # MSE on all subjects (including censored)
    mse_all = mean_squared_error(y_time, predictions)
    mae_all = mean_absolute_error(y_time, predictions)
    rmse_all = np.sqrt(mse_all)

    print(f"\n{name} Set Metrics:")
    print(f"  C-index: {c_index:.4f}")
    print(f"  MSE (events only): {mse:.4f}")
    print(f"  RMSE (events only): {rmse:.4f}")
    print(f"  MAE (events only): {mae:.4f}")
    print(f"  MSE (all): {mse_all:.4f}")
    print(f"  RMSE (all): {rmse_all:.4f}")
    print(f"  MAE (all): {mae_all:.4f}")

    return {
        "c_index": c_index,
        "mse_events": mse,
        "rmse_events": rmse,
        "mae_events": mae,
        "mse_all": mse_all,
        "rmse_all": rmse_all,
        "mae_all": mae_all,
    }


def cross_validate(
    X: np.ndarray,
    y_time: np.ndarray,
    y_event: np.ndarray,
    n_folds: int = 5,
    params: Optional[dict] = None,
) -> dict:
    """Perform k-fold cross-validation."""

    print(f"\n[INFO] Running {n_folds}-fold cross-validation...")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_results = {
        "c_index": [],
        "mse_events": [],
        "mae_events": [],
        "mse_all": [],
        "mae_all": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n  Fold {fold}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train_time, y_val_time = y_time[train_idx], y_time[val_idx]
        y_train_event, y_val_event = y_event[train_idx], y_event[val_idx]

        # Create survival labels
        y_train_lower = y_train_time.copy()
        y_train_upper = np.where(y_train_event == 1, y_train_time, np.inf)

        model = train_xgboost_survival(
            X_train, y_train_lower, y_train_upper, params=params
        )

        metrics = evaluate_model(
            model, X_val, y_val_time, y_val_event, name=f"Fold {fold}"
        )

        for k, v in metrics.items():
            if not np.isnan(v):
                cv_results[k].append(v)

    # Compute means
    mean_results = {k: np.mean(v) for k, v in cv_results.items()}
    std_results = {k: np.std(v) for k, v in cv_results.items()}

    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*80}")
    for k in ["c_index", "mse_events", "mae_events", "mse_all", "mae_all"]:
        print(f"{k}: {mean_results[k]:.4f} ± {std_results[k]:.4f}")

    return mean_results, std_results


def get_feature_importance(model, feat_cols: list[str]) -> pd.DataFrame:
    """Extract feature importance."""
    importance = model.get_score(importance_type="gain")

    # Map feature names
    feat_importance = []
    for i, col in enumerate(feat_cols):
        score = importance.get(f"f{i}", 0)
        feat_importance.append({"feature": col, "importance": score})

    df_imp = pd.DataFrame(feat_importance).sort_values("importance", ascending=False)
    return df_imp


def make_predictions(
    model, df: pd.DataFrame, X: np.ndarray, feat_cols: list[str], out_path: Path
):
    """Generate predictions."""

    dmat = xgb.DMatrix(X)
    predictions = model.predict(dmat)

    pred_df = pd.DataFrame(
        {
            "subject": df["subject"].values,
            "actual_time": df["time_years"].values,
            "event": df["event"].values,
            "predicted_time": predictions,
        }
    )

    pred_df.to_csv(out_path, index=False)
    print(f"\n[OK] Predictions saved: {out_path}")

    return pred_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to time_to_conversion.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--top_k", type=int, default=None, help="Use only top K features by variance"
    )
    ap.add_argument(
        "--cv_folds", type=int, default=5, help="Number of cross-validation folds"
    )
    ap.add_argument(
        "--test_size", type=float, default=0.2, help="Test set size (default 0.2)"
    )
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_survival_data(args.data)

    # Select features
    exclude = {"subject", "event", "time_years", "image_id"}
    feat_cols = select_features(df, exclude, args.top_k)
    print(f"[INFO] Using {len(feat_cols)} features")

    # Prepare data
    df_clean, scaler = prepare_data(df, feat_cols)

    # Split train/test
    train_df, test_df = train_test_split(
        df_clean, test_size=args.test_size, random_state=42, stratify=df_clean["event"]
    )

    print(f"\n[INFO] Train: {len(train_df)}, Test: {len(test_df)}")

    X_train = train_df[feat_cols].values
    y_train_time = train_df["time_years"].values
    y_train_event = train_df["event"].values

    X_test = test_df[feat_cols].values
    y_test_time = test_df["time_years"].values
    y_test_event = test_df["event"].values

    # Create survival labels (lower and upper bounds)
    y_train_lower = y_train_time.copy()
    y_train_upper = np.where(y_train_event == 1, y_train_time, np.inf)

    # Train model
    params = {"max_depth": args.max_depth, "learning_rate": args.learning_rate}

    model = train_xgboost_survival(X_train, y_train_lower, y_train_upper, params=params)

    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train_time, y_train_event, "Train")
    test_metrics = evaluate_model(model, X_test, y_test_time, y_test_event, "Test")

    # Cross-validation on full data
    X_full = df_clean[feat_cols].values
    y_full_time = df_clean["time_years"].values
    y_full_event = df_clean["event"].values

    cv_mean, cv_std = cross_validate(
        X_full, y_full_time, y_full_event, args.cv_folds, params
    )

    # Feature importance
    feat_imp = get_feature_importance(model, feat_cols)

    print(f"\n{'='*80}")
    print("TOP 20 FEATURES BY IMPORTANCE")
    print(f"{'='*80}")
    print(feat_imp.head(20).to_string(index=False))

    # Save outputs
    results = {
        "model": "XGBoost_AFT",
        "n_features": len(feat_cols),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "params": params,
    }

    results_path = out_dir / "xgboost_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {results_path}")

    # Save feature importance
    imp_path = out_dir / "xgboost_feature_importance.csv"
    feat_imp.to_csv(imp_path, index=False)
    print(f"[OK] Feature importance saved: {imp_path}")

    # Make predictions
    pred_path = out_dir / "xgboost_predictions.csv"
    make_predictions(model, df_clean, X_full, feat_cols, pred_path)

    # Save model
    model_path = out_dir / "xgboost_model.json"
    model.save_model(str(model_path))
    print(f"[OK] Model saved: {model_path}")

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Test C-index: {test_metrics['c_index']:.4f}")
    print(f"Test MSE (events): {test_metrics['mse_events']:.4f}")
    print(f"Test MSE (all): {test_metrics['mse_all']:.4f}")
    print(f"CV C-index: {cv_mean['c_index']:.4f} ± {cv_std['c_index']:.4f}")
    print(f"CV MSE (all): {cv_mean['mse_all']:.4f} ± {cv_std['mse_all']:.4f}")


if __name__ == "__main__":
    main()
