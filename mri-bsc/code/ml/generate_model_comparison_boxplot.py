#!/usr/bin/env python3
"""
Generate box plot comparing different model variants.

Similar to the Kaplan-Meier visualization style, creates box plots showing
performance distributions across different model configurations.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold


def run_cross_validation(slopes_df, survival_df, model_type, top_k, n_folds=5):
    """
    Run k-fold cross-validation for a model variant.

    Returns:
        list of dicts with metrics for each fold
    """
    # Merge data
    df = survival_df.merge(slopes_df, on="subject", how="inner")

    # Select features
    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().sum() > len(df) * 0.8]
    variances = df[slope_cols].var()

    if top_k == -1:
        top_features = slope_cols
    else:
        top_features = variances.nlargest(top_k).index.tolist()

    # Prepare features
    X = df[top_features].copy().fillna(df[top_features].median())
    y = df[["time_years", "event"]].copy()

    # K-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        # Train model
        try:
            if model_type == "rsf":
                y_train_surv = Surv.from_dataframe("event", "time_years", y_train)
                y_test_surv = Surv.from_dataframe("event", "time_years", y_test)

                model = RandomSurvivalForest(
                    n_estimators=500,  # Reduced for speed in cross-validation
                    min_samples_split=10,
                    min_samples_leaf=15,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=42,
                )
                model.fit(X_train_scaled, y_train_surv)

                train_c = model.score(X_train_scaled, y_train_surv)
                test_c = model.score(X_test_scaled, y_test_surv)

            else:
                if model_type == "weibull":
                    model = WeibullAFTFitter(penalizer=0.1)
                elif model_type == "lognormal":
                    model = LogNormalAFTFitter(penalizer=0.1)
                elif model_type == "loglogistic":
                    model = LogLogisticAFTFitter(penalizer=0.1)

                train_df = X_train_scaled.copy()
                train_df["time_years"] = y_train["time_years"].values
                train_df["event"] = y_train["event"].values

                test_df = X_test_scaled.copy()
                test_df["time_years"] = y_test["time_years"].values
                test_df["event"] = y_test["event"].values

                model.fit(
                    train_df,
                    duration_col="time_years",
                    event_col="event",
                    show_progress=False,
                )

                train_pred = model.predict_median(train_df)
                test_pred = model.predict_median(test_df)

                train_c = concordance_index(
                    y_train["time_years"], -train_pred, y_train["event"]
                )
                test_c = concordance_index(
                    y_test["time_years"], -test_pred, y_test["event"]
                )

            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_c": train_c,
                    "test_c": test_c,
                    "overfit_gap": train_c - test_c,
                }
            )

        except Exception as e:
            print(f"  Fold {fold_idx} failed: {e}")
            continue

    return fold_results


def generate_comparison_data(slopes_path, survival_path):
    """Generate performance data for all model variants via cross-validation."""
    print("=" * 80)
    print("GENERATING MODEL COMPARISON DATA (5-Fold Cross-Validation)")
    print("=" * 80)

    slopes = pd.read_csv(slopes_path)
    survival = pd.read_csv(survival_path)

    # Model configurations
    configs = [
        ("rsf", 10, "RSF-10"),
        ("rsf", 15, "RSF-15"),
        ("rsf", 20, "RSF-20"),
        ("rsf", 25, "RSF-25"),
        ("rsf", 30, "RSF-30"),
        ("weibull", 20, "Weibull-20"),
        ("lognormal", 20, "LogNormal-20"),
        ("loglogistic", 20, "LogLogistic-20"),
    ]

    all_results = []

    for model_type, top_k, name in configs:
        print(f"\nRunning {name} ({model_type}, top_k={top_k})...")
        fold_results = run_cross_validation(
            slopes, survival, model_type, top_k, n_folds=5
        )

        for result in fold_results:
            result["model"] = name
            result["model_type"] = model_type
            result["top_k"] = top_k
            all_results.append(result)

        avg_test_c = np.mean([r["test_c"] for r in fold_results])
        print(f"  Average test C-index: {avg_test_c:.4f}")

    return pd.DataFrame(all_results)


def create_comparison_boxplots(df, out_dir):
    """Create box plots comparing model performance."""
    print("\n" + "=" * 80)
    print("CREATING MODEL COMPARISON BOX PLOTS")
    print("=" * 80)

    # Set style similar to KM curves
    plt.style.use("seaborn-v0_8-darkgrid")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prepare data - separate by model type
    rsf_models = [m for m in df["model"].unique() if "RSF" in m]
    aft_models = [m for m in df["model"].unique() if "RSF" not in m]
    all_models = sorted(df["model"].unique())

    metrics = [
        ("train_c", "Train C-Index", axes[0]),
        ("test_c", "Test C-Index", axes[1]),
        ("overfit_gap", "Overfitting Gap\n(Train - Test)", axes[2]),
    ]

    for metric, title, ax in metrics:
        # Prepare data for box plot
        data_to_plot = []
        labels = []
        colors = []

        for model in all_models:
            model_data = df[df["model"] == model][metric].values
            data_to_plot.append(model_data)
            labels.append(model)

            # Color by model type
            if "RSF" in model:
                colors.append("#2E86AB")  # Blue for RSF
            elif "Weibull" in model:
                colors.append("#A23B72")  # Purple for Weibull
            elif "LogNormal" in model:
                colors.append("#F18F01")  # Orange for LogNormal
            elif "LogLogistic" in model:
                colors.append("#C73E1D")  # Red for LogLogistic
            else:
                colors.append("#6A994E")  # Green for others

        # Create box plot
        bp = ax.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            widths=0.6,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Formatting
        ax.set_ylabel(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Model Configuration", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        # Add horizontal line at optimal values
        if metric == "test_c":
            ax.axhline(
                y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)"
            )
            ax.legend(loc="upper left", fontsize=9)
        elif metric == "train_c":
            ax.axhline(
                y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)"
            )
            ax.legend(loc="upper left", fontsize=9)
        elif metric == "overfit_gap":
            ax.axhline(
                y=0, color="gray", linestyle="--", alpha=0.5, label="No overfitting"
            )
            ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "Model Performance Comparison (5-Fold Cross-Validation)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    # Save
    plot_path = out_dir / "model_comparison_boxplot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved box plot: {plot_path}")

    plot_pdf = out_dir / "model_comparison_boxplot.pdf"
    plt.savefig(plot_pdf, bbox_inches="tight")
    print(f"✅ Saved PDF: {plot_pdf}")

    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    summary = df.groupby("model")[["test_c", "train_c", "overfit_gap"]].agg(
        ["mean", "std", "min", "max"]
    )
    print(summary.round(4))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slopes", required=True, help="Path to slopes CSV")
    parser.add_argument("--survival", required=True, help="Path to survival CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison data
    df = generate_comparison_data(args.slopes, args.survival)

    # Save raw results
    csv_path = out_dir / "model_comparison_cv_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved cross-validation results: {csv_path}")

    # Create box plots
    create_comparison_boxplots(df, out_dir)

    print("\n" + "=" * 80)
    print("✅ Model comparison box plots generated!")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
