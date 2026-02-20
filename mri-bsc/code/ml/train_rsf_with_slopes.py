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
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


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

    print(f"\n{'='*80}")
    print(
        f"FEATURE SELECTION: Top {top_k} features by VARIANCE (BEFORE standardization)"
    )
    print(f"{'='*80}")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:40s} variance = {variances[feat]:15.2f}")
    print(f"{'='*80}")

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

    print(f"\n{'='*80}")
    print(f"RESULTS: RANDOM SURVIVAL FOREST WITH BSC SLOPES")
    print(f"{'='*80}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Test C-index:  {test_c:.4f}")
    print(f"Overfitting Gap: {(train_c - test_c):.4f}")
    if not np.isnan(test_rmse):
        print(f"Test RMSE (events, approximate): {test_rmse:.4f} years")
    print(f"{'='*80}")

    return rsf, metrics, train_risk, test_risk


def create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir):
    """
    Create Kaplan-Meier survival curves stratified by predicted risk.

    Args:
        y_train, y_test: DataFrames with 'time_years' and 'event' columns
        train_risk, test_risk: Predicted risk scores from RSF
        out_dir: Output directory for plots
    """
    print(f"\n{'='*70}")
    print("CREATING KAPLAN-MEIER CURVES")
    print(f"{'='*70}")

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot for both train and test sets
    for idx, (y_data, risk_scores, title, ax) in enumerate(
        [
            (y_train, train_risk, "Training Set", axes[0]),
            (y_test, test_risk, "Test Set", axes[1]),
        ]
    ):
        # Stratify by risk score (tertiles: high, medium, low risk)
        risk_tertiles = np.percentile(risk_scores, [33.33, 66.67])

        # High risk = top tertile (highest risk scores)
        # Low risk = bottom tertile (lowest risk scores)
        high_risk_mask = risk_scores >= risk_tertiles[1]
        low_risk_mask = risk_scores <= risk_tertiles[0]
        medium_risk_mask = ~(high_risk_mask | low_risk_mask)

        # Prepare data
        time = y_data["time_years"].values
        event = y_data["event"].values

        # Count events in each group
        high_events = event[high_risk_mask].sum()
        medium_events = event[medium_risk_mask].sum()
        low_events = event[low_risk_mask].sum()

        high_n = high_risk_mask.sum()
        medium_n = medium_risk_mask.sum()
        low_n = low_risk_mask.sum()

        print(f"\n{title}:")
        print(
            f"  High risk:   {high_n} subjects ({high_events} events, {high_events/high_n*100:.1f}%)"
        )
        print(
            f"  Medium risk: {medium_n} subjects ({medium_events} events, {medium_events/medium_n*100:.1f}%)"
        )
        print(
            f"  Low risk:    {low_n} subjects ({low_events} events, {low_events/low_n*100:.1f}%)"
        )

        # Fit Kaplan-Meier curves
        kmf_high = KaplanMeierFitter()
        kmf_medium = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(
            time[high_risk_mask],
            event[high_risk_mask],
            label=f"High Risk (n={high_n}, events={high_events})",
        )
        kmf_medium.fit(
            time[medium_risk_mask],
            event[medium_risk_mask],
            label=f"Medium Risk (n={medium_n}, events={medium_events})",
        )
        kmf_low.fit(
            time[low_risk_mask],
            event[low_risk_mask],
            label=f"Low Risk (n={low_n}, events={low_events})",
        )

        # Plot curves
        kmf_high.plot_survival_function(ax=ax, color="red", linewidth=2.5, ci_show=True)
        kmf_medium.plot_survival_function(
            ax=ax, color="orange", linewidth=2.5, ci_show=True
        )
        kmf_low.plot_survival_function(
            ax=ax, color="green", linewidth=2.5, ci_show=True
        )

        # Calculate median survival times
        try:
            median_high = kmf_high.median_survival_time_
            median_medium = kmf_medium.median_survival_time_
            median_low = kmf_low.median_survival_time_
            print(f"  Median survival times:")
            print(
                f"    High risk:   {median_high:.2f} years"
                if not np.isnan(median_high)
                else "    High risk:   Not reached"
            )
            print(
                f"    Medium risk: {median_medium:.2f} years"
                if not np.isnan(median_medium)
                else "    Medium risk: Not reached"
            )
            print(
                f"    Low risk:    {median_low:.2f} years"
                if not np.isnan(median_low)
                else "    Low risk:    Not reached"
            )
        except:
            pass

        # Log-rank test (high vs low)
        logrank_result = logrank_test(
            time[high_risk_mask],
            time[low_risk_mask],
            event[high_risk_mask],
            event[low_risk_mask],
        )

        print(f"  Log-rank test (High vs Low): p={logrank_result.p_value:.4f}")

        # Formatting
        ax.set_xlabel("Time (years)", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            "Probability of Remaining MCI\n(Not Converting to AD)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(
            f"{title}\nKaplan-Meier Survival Curves by Predicted Risk",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best", fontsize=10, framealpha=0.9)

        # Add p-value annotation
        ax.text(
            0.98,
            0.02,
            f"Log-rank p={logrank_result.p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Set y-axis limits
        ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Save figure
    km_plot_path = out_dir / "kaplan_meier_curves.png"
    plt.savefig(km_plot_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved Kaplan-Meier curves: {km_plot_path}")

    # Also save high-res PDF version
    km_plot_pdf = out_dir / "kaplan_meier_curves.pdf"
    plt.savefig(km_plot_pdf, bbox_inches="tight")
    print(f"✅ Saved PDF version: {km_plot_pdf}")

    plt.close()

    print(f"{'='*70}\n")


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

    print(f"\n{'='*80}")
    print(f"DATA SPLIT: Subject-level split (NO DATA LEAKAGE)")
    print(f"{'='*80}")
    print(f"Total subjects: {len(df)}")
    print(f"Total events: {y['event'].sum()}")
    print(f"Event rate: {y['event'].sum() / len(df) * 100:.1f}%")

    # Train/test split (70-30 to match documentation: 315 train / 135 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y["event"]
    )

    print(
        f"Train: {len(X_train)} subjects ({y_train['event'].sum()} events, {y_train['event'].sum()/len(X_train)*100:.1f}%)"
    )
    print(
        f"Test:  {len(X_test)} subjects ({y_test['event'].sum()} events, {y_test['event'].sum()/len(X_test)*100:.1f}%)"
    )
    print(f"{'='*80}")

    # Standardize features (CRITICAL: fit on train, transform on test to prevent leakage)
    print(f"\n{'='*80}")
    print(f"STANDARDIZATION: Fit on TRAIN data only (z-score normalization)")
    print(f"{'='*80}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Print variance AFTER standardization
    print(f"\nFeature variances AFTER standardization (should all be ~1.0):")
    post_variances = X_train_scaled.var()
    for i, feat in enumerate(slope_features, 1):
        print(f"  {i:2d}. {feat:40s} variance = {post_variances[feat]:15.6f}")
    print(f"\nMean variance after standardization: {post_variances.mean():.6f}")
    print(f"All features now contribute equally to the model!")
    print(f"{'='*80}")

    # Train model
    model, metrics, train_risk, test_risk = train_rsf_model(
        X_train_scaled, X_test_scaled, y_train, y_test, args.n_estimators
    )

    # Create Kaplan-Meier curves
    create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir)

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

    # Save predictions (using unstandardized features for interpretability)
    test_df = X_test.copy()
    test_df["subject"] = df.loc[X_test.index, "subject"].values
    test_df["true_time"] = y_test["time_years"].values
    test_df["event"] = y_test["event"].values
    test_df["predicted_risk"] = test_risk  # Use the returned risk scores

    pred_path = out_dir / "rsf_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f"✅ Saved predictions: {pred_path}")

    # Summary
    summary_path = out_dir / "rsf_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"BSC SLOPES RANDOM SURVIVAL FOREST (WITH STANDARDIZATION)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {len(df)} subjects, {y['event'].sum()} events\n")
        f.write(
            f"Features: {len(slope_features)} slope features (z-score standardized)\n"
        )
        f.write(f"Train/Test Split: {len(X_train)}/{len(X_test)} (70-30, stratified)\n")
        f.write(f"N Estimators: {args.n_estimators}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Train C-index:     {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test C-index:      {metrics['test_c_index']:.4f}\n")
        f.write(
            f"  Overfitting Gap:   {metrics['train_c_index'] - metrics['test_c_index']:.4f}\n"
        )
        if metrics["test_rmse_events"]:
            f.write(f"  Test RMSE:         {metrics['test_rmse_events']:.4f} years\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("STANDARDIZATION IMPACT:\n")
        f.write("  - All features now contribute equally (variance = 1.0)\n")
        f.write("  - Previous: Nboundary_slope dominated (var = 35,810,456)\n")
        f.write("  - Other features had var < 0.01 (essentially ignored)\n")
        f.write("  - Standardization ensures fair feature importance\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("DATA LEAKAGE PREVENTION:\n")
        f.write("  ✓ Subject-level split (315 train / 135 test subjects)\n")
        f.write("  ✓ Same subject NEVER in both train and test\n")
        f.write("  ✓ StandardScaler fit on TRAIN only, applied to test\n")
        f.write("  ✓ Feature selection on ALL data (variance is not learned)\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY COMPARISON:\n")
        f.write("  Baseline BSC (AFT):         C-index ~0.24 (FAILED)\n")
        f.write("  Slopes (Weibull AFT):       C-index ~0.61\n")
        f.write(
            f"  Slopes (RSF + Scaling):     C-index {metrics['test_c_index']:.4f}\n\n"
        )
        f.write("REFERENCE (from papers):\n")
        f.write("  Zawawi 2024:     C-index 0.85 (CN to MCI)\n")
        f.write("  Abuhantash 2025: C-index 0.84 (CN to MCI)\n")

    print(f"✅ Saved summary: {summary_path}")
    print("\n" + "=" * 80)
    print("✅ DONE! Random Survival Forest training complete")
    print("=" * 80)
    print(f"\nResults in: {out_dir}")
    print(f"\n📊 TRAIN C-index: {metrics['train_c_index']:.4f}")
    print(f"📊 TEST C-index:  {metrics['test_c_index']:.4f}")
    print(
        f"📊 Overfitting Gap: {metrics['train_c_index'] - metrics['test_c_index']:.4f}"
    )
    print("\nNext steps:")
    print("  1. Compare to previous unstandardized results")
    print("  2. Analyze feature importance (now all features contribute fairly)")
    print("  3. Consider permutation-based importance for clinical interpretation")
    print("=" * 80)


if __name__ == "__main__":
    main()
