"""
Train Random Survival Forest using BSC SLOPES as predictors.

Based on papers achieving C-index 0.84-0.85 with Random Forest for survival analysis.
This non-parametric approach should handle complex feature interactions better than
parametric AFT models.

Usage:
    python3 -m code.ml.train_rsf_with_slopes \
        --slopes data/index/bsc_longitudinal_slopes.csv \
        --survival data/ml/survival/time_to_conversion.csv \
        --out_dir data/ml/results/rsf \
        --top_k 20

Key Differences from AFT:
    - Non-parametric (no distribution assumptions)
    - Handles complex interactions
    - More robust to feature scaling issues
    - Based on papers: Zawawi 2024 (C-index 0.85), Abuhantash 2025 (C-index 0.84)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
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


def train_rsf_model(X_train, X_test, y_train, y_test, n_estimators=1000):
    """
    Train Random Survival Forest.

    Args:
        X_train, X_test: Feature dataframes
        y_train, y_test: Target dataframes with 'time_years' and 'event'
        n_estimators: Number of trees (default 1000, as per papers)
    """
    # Convert to structured array format required by scikit-survival
    y_train_surv = Surv.from_dataframe("event", "time_years", y_train)
    y_test_surv = Surv.from_dataframe("event", "time_years", y_test)

    # Initialize Random Survival Forest
    # Hyperparameters from Zawawi et al. 2024 paper
    print(f"\nFitting Random Survival Forest...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  min_samples_split: 10")
    print(f"  min_samples_leaf: 15")

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    # Fit model
    rsf.fit(X_train, y_train_surv)

    # Get concordance index (C-index)
    train_c = rsf.score(X_train, y_train_surv)
    test_c = rsf.score(X_test, y_test_surv)

    # Get risk scores for both sets
    train_risk = rsf.predict(X_train)
    test_risk = rsf.predict(X_test)

    # Compute RMSE on events only (using median predicted survival time)
    # For RSF, higher risk score means shorter survival
    train_events = y_train["event"] == 1
    test_events = y_test["event"] == 1

    if train_events.sum() > 0:
        # Approximate predicted time from risk scores (inverse relationship)
        # This is a simplification since RSF doesn't directly predict times
        train_pred_time = -train_risk  # Negative risk as proxy for time
        train_mse = np.mean(
            (
                y_train.loc[train_events, "time_years"].values
                - train_pred_time[train_events.values]
            )
            ** 2
        )
        train_rmse = np.sqrt(train_mse)
    else:
        train_mse = train_rmse = np.nan

    if test_events.sum() > 0:
        test_pred_time = -test_risk
        test_mse = np.mean(
            (
                y_test.loc[test_events, "time_years"].values
                - test_pred_time[test_events.values]
            )
            ** 2
        )
        test_rmse = np.sqrt(test_mse)
    else:
        test_mse = test_rmse = np.nan

    metrics = {
        "train_c_index": float(train_c),
        "test_c_index": float(test_c),
        "train_mse_events": float(train_mse) if not np.isnan(train_mse) else None,
        "train_rmse_events": float(train_rmse) if not np.isnan(train_rmse) else None,
        "test_mse_events": float(test_mse) if not np.isnan(test_mse) else None,
        "test_rmse_events": float(test_rmse) if not np.isnan(test_rmse) else None,
        "n_estimators": n_estimators,
        "n_features": len(X_train.columns),
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: RANDOM SURVIVAL FOREST WITH BSC SLOPES")
    print(f"{'='*70}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Test C-index:  {test_c:.4f}")
    if not np.isnan(test_rmse):
        print(f"Test RMSE (events, approximate): {test_rmse:.4f} years")
    print(f"{'='*70}")

    return rsf, metrics


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
        "--n_estimators", type=int, default=1000, help="Number of trees in forest"
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

    # Standardize features (RSF doesn't require this but may help with feature importance)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y["event"]
    )

    print(f"\nTrain: {len(X_train)} subjects ({y_train['event'].sum()} events)")
    print(f"Test:  {len(X_test)} subjects ({y_test['event'].sum()} events)")

    # Train model
    model, metrics = train_rsf_model(
        X_train, X_test, y_train, y_test, args.n_estimators
    )

    # Save results
    metrics_path = out_dir / "rsf_slopes_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics: {metrics_path}")

    # Save feature list
    features_path = out_dir / "rsf_features.txt"
    with open(features_path, "w") as f:
        for feat in slope_features:
            f.write(f"{feat}\n")
    print(f"✅ Saved features: {features_path}")

    # Save predictions
    test_df = X_test.copy()
    test_df["subject"] = df.loc[X_test.index, "subject"].values
    test_df["true_time"] = y_test["time_years"].values
    test_df["event"] = y_test["event"].values
    test_df["predicted_risk"] = model.predict(X_test)

    pred_path = out_dir / "rsf_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"✅ Saved predictions: {pred_path}")

    # Summary
    summary_path = out_dir / "rsf_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"BSC SLOPES RANDOM SURVIVAL FOREST\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Data: {len(df)} subjects, {y['event'].sum()} events\n")
        f.write(f"Features: {len(slope_features)} slope features\n")
        f.write(f"Train/Test: {len(X_train)}/{len(X_test)}\n")
        f.write(f"N Estimators: {args.n_estimators}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Train C-index: {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test C-index:  {metrics['test_c_index']:.4f}\n")
        if metrics["test_rmse_events"]:
            f.write(f"  Test RMSE:     {metrics['test_rmse_events']:.4f} years\n")
        f.write("\nKEY COMPARISON:\n")
        f.write("  Baseline BSC (AFT):      C-index ~0.24 (FAILED)\n")
        f.write("  Slopes (Weibull AFT):    C-index ~0.61\n")
        f.write(f"  Slopes (Random Forest):  C-index {metrics['test_c_index']:.4f}\n\n")
        f.write("REFERENCE (from papers):\n")
        f.write("  Zawawi 2024:     C-index 0.85 (CN to MCI)\n")
        f.write("  Abuhantash 2025: C-index 0.84 (CN to MCI)\n")

    print(f"✅ Saved summary: {summary_path}")
    print("\n" + "=" * 70)
    print("✅ DONE! Random Survival Forest training complete")
    print("=" * 70)
    print(f"\nResults in: {out_dir}")
    print(f"\nTest C-index: {metrics['test_c_index']:.4f}")
    print("\nNext steps:")
    print("  1. Check if C-index improved vs Weibull (0.61)")
    print("  2. If C-index still low, try adding demographics/cognitive scores")
    print("  3. Consider permutation-based feature importance analysis")
    print("=" * 70)


if __name__ == "__main__":
    main()
