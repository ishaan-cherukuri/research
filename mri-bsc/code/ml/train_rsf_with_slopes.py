
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

def signed_log1p_df(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return np.sign(X) * np.log1p(np.abs(X))

def fit_winsor_limits(X: pd.DataFrame, lower_q: float, upper_q: float) -> dict:
    lower = X.quantile(lower_q)
    upper = X.quantile(upper_q)
    return {"lower": lower, "upper": upper}

def apply_winsor_limits(X: pd.DataFrame, limits: dict) -> pd.DataFrame:
    return X.clip(lower=limits["lower"], upper=limits["upper"], axis=1)

def minmax_scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler

def select_slope_features_train_only(
    X_train: pd.DataFrame,
    top_k: int = 20,
    penalize_regex: str = "nboundary",
    penalty_factor: float = 0.10,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
):
    X_log = signed_log1p_df(X_train)
    limits = fit_winsor_limits(X_log, winsor_q_low, winsor_q_high)
    X_robust = apply_winsor_limits(X_log, limits)

    raw_var = X_robust.var()
    score = raw_var.copy()

    mask = pd.Series(
        score.index.str.contains(penalize_regex, case=False, regex=True),
        index=score.index,
    )
    score.loc[mask] = score.loc[mask] * penalty_factor

    top_features = score.nlargest(top_k).index.tolist()

    print(f"\n{'='*80}")
    print("FEATURE SELECTION (TRAIN ONLY)")
    print(
        f"  Robust variance ranking: signed_log1p + winsorize [{winsor_q_low}, {winsor_q_high}]"
    )
    print(f"  Penalize regex: '{penalize_regex}' with factor {penalty_factor}")
    print(f"  Selecting Top {top_k} features by SCORE")
    print(f"{'='*80}")
    for i, feat in enumerate(top_features, 1):
        pen = " (PENALIZED)" if mask.loc[feat] else ""
        print(
            f"  {i:2d}. {feat:40s} "
            f"robust_var={raw_var[feat]:12.6f}  score={score[feat]:12.6f}{pen}"
        )
    print(f"{'='*80}")

    return top_features

def train_rsf_model(X_train, X_test, y_train, y_test, n_estimators=1000):
    y_train_surv = Surv.from_dataframe("event", "time_years", y_train)
    y_test_surv = Surv.from_dataframe("event", "time_years", y_test)

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

    rsf.fit(X_train, y_train_surv)

    train_c = rsf.score(X_train, y_train_surv)
    test_c = rsf.score(X_test, y_test_surv)

    train_risk = rsf.predict(X_train)
    test_risk = rsf.predict(X_test)

    train_events = y_train["event"] == 1
    test_events = y_test["event"] == 1

    def rmse_events(y_df, risk, events_mask):
        if events_mask.sum() == 0:
            return np.nan, np.nan
        pred_time = -risk
        mse = np.mean(
            (y_df.loc[events_mask, "time_years"].values - pred_time[events_mask.values])
            ** 2
        )
        return mse, np.sqrt(mse)

    train_mse, train_rmse = rmse_events(y_train, train_risk, train_events)
    test_mse, test_rmse = rmse_events(y_test, test_risk, test_events)

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
    print(f"RESULTS: RANDOM SURVIVAL FOREST (robust selection + minmax scaling)")
    print(f"{'='*80}")
    print(f"Train C-index: {train_c:.4f}")
    print(f"Test C-index:  {test_c:.4f}")
    print(f"Overfitting Gap: {(train_c - test_c):.4f}")
    if not np.isnan(test_rmse):
        print(f"Test RMSE (events, approximate): {test_rmse:.4f} years")
    print(f"{'='*80}")

    return rsf, metrics, train_risk, test_risk

def create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir: Path):
    print(f"\n{'='*70}")
    print("CREATING KAPLAN-MEIER CURVES")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for y_data, risk_scores, title, ax in [
        (y_train, train_risk, "Training Set", axes[0]),
        (y_test, test_risk, "Test Set", axes[1]),
    ]:
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

        kmf_high.plot_survival_function(ax=ax, color="red", linewidth=2.5, ci_show=True)
        kmf_medium.plot_survival_function(
            ax=ax, color="orange", linewidth=2.5, ci_show=True
        )
        kmf_low.plot_survival_function(
            ax=ax, color="green", linewidth=2.5, ci_show=True
        )

        logrank_result = logrank_test(
            time[high_risk_mask],
            time[low_risk_mask],
            event[high_risk_mask],
            event[low_risk_mask],
        )
        print(f"  Log-rank test (High vs Low): p={logrank_result.p_value:.4f}")

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
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    km_plot_path = out_dir / "kaplan_meier_curves.png"
    plt.savefig(km_plot_path, dpi=300, bbox_inches="tight")
    print(f"\n Saved Kaplan-Meier curves: {km_plot_path}")

    km_plot_pdf = out_dir / "kaplan_meier_curves.pdf"
    plt.savefig(km_plot_pdf, bbox_inches="tight")
    print(f" Saved PDF version: {km_plot_pdf}")

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

    parser.add_argument(
        "--penalize_regex",
        type=str,
        default="nboundary",
        help="Regex (case-insensitive) for features to penalize during selection.",
    )
    parser.add_argument(
        "--penalty_factor",
        type=float,
        default=0.10,
        help="Multiply selection SCORE by this for penalized features (e.g. 0.1).",
    )
    parser.add_argument(
        "--winsor_low",
        type=float,
        default=0.01,
        help="Lower quantile for winsorization.",
    )
    parser.add_argument(
        "--winsor_high",
        type=float,
        default=0.99,
        help="Upper quantile for winsorization.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_data(args.slopes, args.survival)

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().sum() > len(df) * 0.8]

    X_all = df[slope_cols].copy()
    y_all = df[["time_years", "event"]].copy()

    X_all = X_all.fillna(X_all.median(numeric_only=True))

    print(f"\n{'='*80}")
    print(f"DATA SPLIT: Subject-level split (NO DATA LEAKAGE)")
    print(f"{'='*80}")
    print(f"Total subjects: {len(df)}")
    print(f"Total events: {y_all['event'].sum()}")
    print(f"Event rate: {y_all['event'].sum() / len(df) * 100:.1f}%")

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all["event"]
    )

    print(
        f"Train: {len(X_train_all)} subjects ({y_train['event'].sum()} events, {y_train['event'].sum()/len(X_train_all)*100:.1f}%)"
    )
    print(
        f"Test:  {len(X_test_all)} subjects ({y_test['event'].sum()} events, {y_test['event'].sum()/len(X_test_all)*100:.1f}%)"
    )
    print(f"{'='*80}")

    slope_features = select_slope_features_train_only(
        X_train_all,
        top_k=args.top_k,
        penalize_regex=args.penalize_regex,
        penalty_factor=args.penalty_factor,
        winsor_q_low=args.winsor_low,
        winsor_q_high=args.winsor_high,
    )

    X_train = X_train_all[slope_features].copy()
    X_test = X_test_all[slope_features].copy()

    X_train_log = signed_log1p_df(X_train)
    X_test_log = signed_log1p_df(X_test)

    limits = fit_winsor_limits(X_train_log, args.winsor_low, args.winsor_high)
    X_train_robust = apply_winsor_limits(X_train_log, limits)
    X_test_robust = apply_winsor_limits(X_test_log, limits)

    print(f"\n{'='*80}")
    print("SCALING: Robust transform + MinMax to (0,1), fit on TRAIN only")
    print(f"{'='*80}")
    X_train_scaled, X_test_scaled, scaler = minmax_scale_train_test(
        X_train_robust, X_test_robust
    )

    post_var = X_train_scaled.var()
    print("Feature ranges AFTER scaling (TRAIN):")
    mins = X_train_scaled.min()
    maxs = X_train_scaled.max()
    for i, feat in enumerate(slope_features, 1):
        print(
            f"  {i:2d}. {feat:40s} min={mins[feat]:.4f} max={maxs[feat]:.4f} var={post_var[feat]:.6f}"
        )
    print(f"\nMean variance after minmax: {post_var.mean():.6f}")
    print(f"{'='*80}")

    model, metrics, train_risk, test_risk = train_rsf_model(
        X_train_scaled, X_test_scaled, y_train, y_test, args.n_estimators
    )

    create_kaplan_meier_curves(y_train, y_test, train_risk, test_risk, out_dir)

    metrics_path = out_dir / "rsf_slopes_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n Saved metrics: {metrics_path}")

    features_path = out_dir / "rsf_features.txt"
    with open(features_path, "w") as f:
        for feat in slope_features:
            f.write(f"{feat}\n")
    print(f" Saved features: {features_path}")

    test_df = X_test.copy()
    test_df["subject"] = df.loc[X_test.index, "subject"].values
    test_df["true_time"] = y_test["time_years"].values
    test_df["event"] = y_test["event"].values
    test_df["predicted_risk"] = test_risk

    pred_path = out_dir / "rsf_predictions.csv"
    test_df.to_csv(pred_path, index=False)
    print(f" Saved predictions: {pred_path}")

    summary_path = out_dir / "rsf_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BSC SLOPES RANDOM SURVIVAL FOREST (ROBUST SELECTION + MINMAX)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {len(df)} subjects, {y_all['event'].sum()} events\n")
        f.write(f"Features: {len(slope_features)} slope features\n")
        f.write("Preprocessing:\n")
        f.write("  - signed_log1p transform\n")
        f.write(
            f"  - winsorization quantiles [{args.winsor_low}, {args.winsor_high}] (fit on train)\n"
        )
        f.write("  - MinMax scaling to (0,1) (fit on train)\n")
        f.write("Selection:\n")
        f.write("  - Robust variance on TRAIN only\n")
        f.write(
            f"  - Penalize regex '{args.penalize_regex}' by factor {args.penalty_factor}\n\n"
        )
        f.write("METRICS:\n")
        f.write(f"  Train C-index:     {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test C-index:      {metrics['test_c_index']:.4f}\n")
        f.write(
            f"  Overfitting Gap:   {metrics['train_c_index'] - metrics['test_c_index']:.4f}\n"
        )
        if metrics["test_rmse_events"]:
            f.write(f"  Test RMSE:         {metrics['test_rmse_events']:.4f} years\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f" Saved summary: {summary_path}")

    print("\n" + "=" * 80)
    print(" DONE! Random Survival Forest training complete")
    print("=" * 80)
    print(f"\nResults in: {out_dir}")
    print(f"\n TRAIN C-index: {metrics['train_c_index']:.4f}")
    print(f" TEST C-index:  {metrics['test_c_index']:.4f}")
    print(
        f" Overfitting Gap: {metrics['train_c_index'] - metrics['test_c_index']:.4f}"
    )
    print("=" * 80)

if __name__ == "__main__":
    main()
