
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def load_and_merge_data(slopes_path: str, survival_path: str):
    print(f"Loading slopes: {slopes_path}")
    slopes = pd.read_csv(slopes_path)

    print(f"Loading survival labels: {survival_path}")
    survival = pd.read_csv(survival_path)

    print(f"  Slopes: {len(slopes)} subjects")
    print(f"  Survival: {len(survival)} subjects")

    merged = survival.merge(slopes, on="subject", how="inner")
    print(f"  Merged: {len(merged)} subjects")

    return merged

def penalize_nboundary_and_select_features(df: pd.DataFrame, top_k: int = 20):
    slope_cols = [c for c in df.columns if c.endswith("_slope")]

    slope_cols = [c for c in slope_cols if df[c].notna().sum() > len(df) * 0.8]

    print(f"\n{'='*80}")
    print(f"STEP 1: RAW VARIANCES (before log-transform)")
    print(f"{'='*80}")

    raw_variances = df[slope_cols].var()
    top_10_raw = raw_variances.nlargest(10)
    for i, (feat, var) in enumerate(top_10_raw.items(), 1):
        print(f"  {i:2d}. {feat:40s} variance = {var:15.2f}")

    print(f"\n{'='*80}")
    print(f"STEP 2: LOG-TRANSFORM Nboundary_slope to prevent domination")
    print(f"{'='*80}")

    transformed_df = df[slope_cols].copy()

    nboundary_col = [c for c in slope_cols if "Nboundary_slope" in c]

    if nboundary_col:
        col = nboundary_col[0]
        original_var = transformed_df[col].var()

        values = transformed_df[col].values
        transformed_values = np.sign(values) * np.log(np.abs(values) + 1)
        transformed_df[col] = transformed_values

        new_var = transformed_df[col].var()

        print(f"  {col}:")
        print(f"    Original variance:    {original_var:15.2f}")
        print(f"    After log-transform:  {new_var:15.2f}")
        print(f"    Reduction factor:     {original_var/new_var:15.2f}x")

    print(f"\n{'='*80}")
    print(
        f"STEP 3: FEATURE SELECTION on log-transformed data (Top {top_k} by variance)"
    )
    print(f"{'='*80}")

    variances = transformed_df.var()
    top_features = variances.nlargest(top_k).index.tolist()

    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:40s} variance = {variances[feat]:15.6f}")

    print(
        f"\n  Variance range: {variances[top_features].min():.6f} to {variances[top_features].max():.6f}"
    )
    print(
        f"  Ratio (max/min): {variances[top_features].max() / variances[top_features].min():.2f}x"
    )
    print(f"{'='*80}")

    return top_features, transformed_df[top_features]

def train_rsf_model(X_train, X_test, y_train, y_test, n_estimators=1000):
    y_train_surv = Surv.from_dataframe("event", "time_years", y_train)
    y_test_surv = Surv.from_dataframe("event", "time_years", y_test)

    print(f"\nFitting Random Survival Forest (NO STANDARDIZATION)...")
    print(f"  n_estimators: {n_estimators}")
    print(f"  min_samples_split: 10")
    print(f"  min_samples_leaf: 15")
    print(f"  max_features: sqrt")

    rsf = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    rsf.fit(X_train, y_train_surv)

    train_c = rsf.score(X_train, y_train_surv)
    test_c = rsf.score(X_test, y_test_surv)

    train_risk = rsf.predict(X_train)
    test_risk = rsf.predict(X_test)

    train_events = y_train["event"] == 1
    test_events = y_test["event"] == 1

    if train_events.sum() > 0:
        train_pred_time = -train_risk
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
    print(f"RESULTS: RANDOM SURVIVAL FOREST (NO STANDARDIZATION)")
    print(f"{'='*80}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Test C-index:  {test_c:.4f}")
    print(f"Overfitting Gap: {(train_c - test_c):.4f}")
    if not np.isnan(test_rmse):
        print(f"Test RMSE (events): {test_rmse:.4f} years")
    print(f"{'='*80}")

    return rsf, metrics, train_risk, test_risk

def create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir):
    print(f"\n{'='*70}")
    print("CREATING KAPLAN-MEIER CURVES")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (y_data, risk_scores, title, ax) in enumerate(
        [
            (y_train, train_risk, "Training Set", axes[0]),
            (y_test, test_risk, "Test Set", axes[1]),
        ]
    ):
        risk_tertiles = np.percentile(risk_scores, [33.33, 66.67])

        high_risk_mask = risk_scores >= risk_tertiles[1]
        low_risk_mask = risk_scores <= risk_tertiles[0]
        medium_risk_mask = ~(high_risk_mask | low_risk_mask)

        time = y_data["time_years"].values
        event = y_data["event"].values

        high_events = event[high_risk_mask].sum()
        medium_events = event[medium_risk_mask].sum()
        low_events = event[low_risk_mask].sum()

        high_n = high_risk_mask.sum()
        medium_n = medium_risk_mask.sum()
        low_n = low_risk_mask.sum()

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

        kmf_high.plot_survival_function(ax=ax, ci_show=True, color="red", linewidth=2)
        kmf_medium.plot_survival_function(
            ax=ax, ci_show=True, color="orange", linewidth=2
        )
        kmf_low.plot_survival_function(ax=ax, ci_show=True, color="green", linewidth=2)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (years)", fontsize=12)
        ax.set_ylabel("Probability of Remaining MCI", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=10)
        ax.set_ylim([0, 1.05])

        results = logrank_test(
            time[high_risk_mask],
            time[low_risk_mask],
            event[high_risk_mask],
            event[low_risk_mask],
        )
        ax.text(
            0.98,
            0.02,
            f"Log-rank p-value (high vs low): {results.p_value:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=9,
        )

    plt.tight_layout()
    plot_path = out_dir / "kaplan_meier_curves_noscale.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f" Saved plot: {plot_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Train Random Survival Forest WITHOUT standardization"
    )
    parser.add_argument(
        "--slopes",
        required=True,
        help="Path to BSC longitudinal slopes CSV",
    )
    parser.add_argument(
        "--survival",
        required=True,
        help="Path to survival labels CSV",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top features to select by variance (default: 20)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1000,
        help="Number of trees in Random Forest (default: 1000)",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_data(args.slopes, args.survival)

    slope_features, X_transformed = penalize_nboundary_and_select_features(
        df, args.top_k
    )

    y = df[["time_years", "event"]].copy()

    print(f"\n{'='*80}")
    print(f"DATA SPLIT: Subject-level split (NO DATA LEAKAGE)")
    print(f"{'='*80}")
    print(f"Total subjects: {len(df)}")
    print(f"Total events: {y['event'].sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.3, random_state=42, stratify=y["event"]
    )

    print(f"Train: {len(X_train)} subjects ({y_train['event'].sum()} events)")
    print(f"Test:  {len(X_test)} subjects ({y_test['event'].sum()} events)")

    print(f"\n{'='*80}")
    print(f"FEATURE VARIANCES IN TRAINING SET (log-transformed, NO standardization)")
    print(f"{'='*80}")
    train_variances = X_train.var()
    for i, feat in enumerate(slope_features, 1):
        print(f"  {i:2d}. {feat:40s} variance = {train_variances[feat]:15.6f}")
    print(
        f"\nVariance range: {train_variances.min():.6f} to {train_variances.max():.6f}"
    )
    print(f"Ratio (max/min): {train_variances.max() / train_variances.min():.2f}x")
    print(f"{'='*80}")

    model, metrics, train_risk, test_risk = train_rsf_model(
        X_train, X_test, y_train, y_test, args.n_estimators
    )

    create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir)

    metrics_path = out_dir / "rsf_noscale_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Saved metrics: {metrics_path}")

    features_path = out_dir / "rsf_noscale_features.txt"
    train_variances = X_train.var()
    with open(features_path, "w") as f:
        f.write(
            "Feature ranking by variance (log-transformed, no standardization):\n\n"
        )
        for i, feat in enumerate(slope_features, 1):
            f.write(f"{i:2d}. {feat:40s} variance = {train_variances[feat]:15.6f}\n")
    print(f" Saved features: {features_path}")

    summary_path = out_dir / "rsf_noscale_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"BSC SLOPES RANDOM SURVIVAL FOREST (NO STANDARDIZATION)\n")
        f.write("=" * 80 + "\n\n")
        f.write("APPROACH:\n")
        f.write("  1. Log-transform Nboundary_slope to prevent domination\n")
        f.write("  2. Select top 20 features by variance on transformed data\n")
        f.write("  3. Train Random Forest WITHOUT StandardScaler\n")
        f.write("  4. Features keep interpretable raw values (range: -inf to +inf)\n\n")
        f.write(f"Data: {len(df)} subjects, {y['event'].sum()} events\n")
        f.write(
            f"Features: {len(slope_features)} slope features (log-transformed only)\n"
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

    print(f" Saved summary: {summary_path}")
    print("\n" + "=" * 80)
    print(" DONE! Random Survival Forest (no standardization) complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
