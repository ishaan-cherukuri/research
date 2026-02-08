"""
Train survival models using BSC SLOPES (rates of change) as predictors.

This is the KEY experiment: Testing if rate of BSC degradation predicts 
conversion better than absolute baseline BSC values.

Usage:
    python3 -m code.ml.train_survival_with_slopes \
        --slopes data/index/bsc_longitudinal_slopes.csv \
        --survival data/ml/survival/time_to_conversion.csv \
        --out_dir data/ml/results/slopes \
        --top_k 20
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_merge_data(slopes_path: str, survival_path: str):
    """Merge slopes with survival labels."""
    print(f"Loading slopes: {slopes_path}")
    slopes = pd.read_csv(slopes_path)

    print(f"Loading survival labels: {survival_path}")
    survival = pd.read_csv(survival_path)

    print(f"  Slopes: {len(slopes)} subjects")
    print(f"  Survival: {len(survival)} subjects")

    # Merge on subject
    merged = survival.merge(slopes, on="subject", how="inner")
    print(f"  Merged: {len(merged)} subjects")

    return merged


def select_slope_features(df: pd.DataFrame, top_k: int = 20):
    """Select top slope features by variance."""
    # Get slope columns
    slope_cols = [c for c in df.columns if c.endswith("_slope")]

    # Filter out columns with too many NaNs
    slope_cols = [c for c in slope_cols if df[c].notna().sum() > len(df) * 0.8]

    # Compute variance
    variances = df[slope_cols].var()
    top_features = variances.nlargest(top_k).index.tolist()

    print(f"\nSelected top {top_k} slope features by variance:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i}. {feat}: var={variances[feat]:.6f}")
    if len(top_features) > 10:
        print(f"  ... and {len(top_features) - 10} more")

    return top_features


def train_aft_model(X_train, X_test, y_train, y_test, model_type="weibull"):
    """Train parametric AFT model."""
    # Prepare training data
    train_df = X_train.copy()
    train_df["time_years"] = y_train["time_years"].values
    train_df["event"] = y_train["event"].values

    # Prepare test data
    test_df = X_test.copy()
    test_df["time_years"] = y_test["time_years"].values
    test_df["event"] = y_test["event"].values

    # Select model
    if model_type == "weibull":
        model = WeibullAFTFitter()
    elif model_type == "lognormal":
        model = LogNormalAFTFitter()
    elif model_type == "loglogistic":
        model = LogLogisticAFTFitter()
    else:
        raise ValueError(f"Unknown model: {model_type}")

    # Fit model
    print(f"\nFitting {model_type.upper()} AFT model...")
    model.fit(train_df, duration_col="time_years", event_col="event")

    # Predictions
    train_pred = model.predict_median(train_df)
    test_pred = model.predict_median(test_df)

    # Concordance index (C-index)
    train_c = concordance_index(y_train["time_years"], -train_pred, y_train["event"])
    test_c = concordance_index(y_test["time_years"], -test_pred, y_test["event"])

    # MSE/RMSE on events only
    train_events = y_train["event"] == 1
    test_events = y_test["event"] == 1

    if train_events.sum() > 0:
        train_mse = np.mean(
            (y_train.loc[train_events, "time_years"] - train_pred[train_events]) ** 2
        )
        train_rmse = np.sqrt(train_mse)
    else:
        train_mse = train_rmse = np.nan

    if test_events.sum() > 0:
        test_mse = np.mean(
            (y_test.loc[test_events, "time_years"] - test_pred[test_events]) ** 2
        )
        test_rmse = np.sqrt(test_mse)
    else:
        test_mse = test_rmse = np.nan

    metrics = {
        "train_c_index": float(train_c),
        "test_c_index": float(test_c),
        "train_mse_events": float(train_mse),
        "train_rmse_events": float(train_rmse),
        "test_mse_events": float(test_mse),
        "test_rmse_events": float(test_rmse),
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {model_type.upper()} AFT MODEL WITH BSC SLOPES")
    print(f"{'='*70}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Test C-index:  {test_c:.4f}")
    print(f"Test RMSE (events): {test_rmse:.4f} years")
    print(f"{'='*70}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--slopes", required=True, help="Path to bsc_longitudinal_slopes.csv"
    )
    parser.add_argument(
        "--survival", required=True, help="Path to time_to_conversion.csv"
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for results")
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of top slope features to use"
    )
    parser.add_argument(
        "--model",
        default="weibull",
        choices=["weibull", "lognormal", "loglogistic"],
        help="AFT model type",
    )

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge data
    df = load_and_merge_data(args.slopes, args.survival)

    # Select slope features
    slope_features = select_slope_features(df, args.top_k)

    # Prepare features and target
    X = df[slope_features].copy()
    y = df[["time_years", "event"]].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"\nTrain: {len(X_train)} subjects ({y_train['event'].sum()} events)")
    print(f"Test:  {len(X_test)} subjects ({y_test['event'].sum()} events)")

    # Train model
    model, metrics = train_aft_model(X_train, X_test, y_train, y_test, args.model)

    # Save results
    metrics_path = out_dir / f"slopes_{args.model}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics: {metrics_path}")

    # Save feature list
    features_path = out_dir / f"slopes_{args.model}_features.txt"
    with open(features_path, "w") as f:
        for feat in slope_features:
            f.write(f"{feat}\n")
    print(f"✅ Saved features: {features_path}")

    # Save predictions
    test_df = X_test.copy()
    test_df["subject"] = df.loc[X_test.index, "subject"].values
    test_df["true_time"] = y_test["time_years"].values
    test_df["event"] = y_test["event"].values
    test_df["predicted_median_time"] = model.predict_median(X_test).values

    pred_path = out_dir / f"slopes_{args.model}_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"✅ Saved predictions: {pred_path}")

    # Summary
    summary_path = out_dir / f"slopes_{args.model}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"BSC SLOPES SURVIVAL MODEL - {args.model.upper()}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data: {len(df)} subjects, {y['event'].sum()} events\n")
        f.write(f"Features: {len(slope_features)} slope features\n")
        f.write(f"Train/Test: {len(X_train)}/{len(X_test)}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Train C-index: {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test C-index:  {metrics['test_c_index']:.4f}\n")
        f.write(f"  Test RMSE:     {metrics['test_rmse_events']:.4f} years\n\n")
        f.write("KEY COMPARISON:\n")
        f.write("  Baseline BSC (previous): C-index ~0.24 (FAILED)\n")
        f.write(f"  BSC Slopes (this run):   C-index {metrics['test_c_index']:.4f}\n")

    print(f"✅ Saved summary: {summary_path}")
    print("\n✅ DONE! Check results in:", out_dir)


if __name__ == "__main__":
    main()
