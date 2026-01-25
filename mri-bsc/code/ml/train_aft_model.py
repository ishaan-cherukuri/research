"""code.ml.train_aft_model

Train Accelerated Failure Time (AFT) models for AD conversion prediction.

AFT models directly predict survival TIME (not hazard like Cox).
Interpretation: coefficients show how features accelerate/decelerate time-to-event.

Example: coefficient = -0.5 for dir_mean
  → Higher dir_mean accelerates conversion (reduces time to AD)
  → exp(-0.5) = 0.606, so 1-unit increase shortens time by ~40%

Supported distributions:
  - Weibull (most common, flexible)
  - Log-Normal (symmetric on log scale)
  - Log-Logistic (similar to Cox but parametric)

Usage:
  python3 -m code.ml.train_aft_model \
    --data data/ml/survival/time_to_conversion.csv \
    --out_dir data/ml/results \
    --model weibull \
    --top_k 20

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_survival_data(csv_path: str) -> pd.DataFrame:
    """Load and validate survival data."""
    df = pd.read_csv(csv_path)

    required = {"subject", "event", "time_years"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove subjects with invalid times
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
    """Select feature columns, optionally filtering to top K by variance."""
    feat_cols = [c for c in df.columns if c not in exclude_cols]

    if top_k and len(feat_cols) > top_k:
        print(f"[INFO] Selecting top {top_k} features by variance...")
        # Remove constant/nan features
        variances = df[feat_cols].var()
        valid = variances[variances > 0].sort_values(ascending=False)
        feat_cols = valid.head(top_k).index.tolist()
        print(f"  Selected: {len(feat_cols)} features")

    return feat_cols


def prepare_data(
    df: pd.DataFrame, feat_cols: list[str], standardize: bool = True
) -> tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Prepare features for training."""
    # Handle missing values
    df_clean = df.copy()
    for col in feat_cols:
        if df_clean[col].isna().any():
            median = df_clean[col].median()
            df_clean[col].fillna(median, inplace=True)

    # Remove infinite values
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    for col in feat_cols:
        if df_clean[col].isna().any():
            median = df_clean[col].median()
            df_clean[col].fillna(median, inplace=True)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        df_clean[feat_cols] = scaler.fit_transform(df_clean[feat_cols])
        print("[INFO] Features standardized (mean=0, std=1)")

    return df_clean, scaler


def train_aft_model(
    df: pd.DataFrame,
    model_type: str,
    duration_col: str = "time_years",
    event_col: str = "event",
    penalizer: float = 0.1,
):
    """Train AFT model."""

    if model_type == "weibull":
        model = WeibullAFTFitter(penalizer=penalizer)
    elif model_type == "lognormal":
        model = LogNormalAFTFitter(penalizer=penalizer)
    elif model_type == "loglogistic":
        model = LogLogisticAFTFitter(penalizer=penalizer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Drop non-numeric identifier columns before fitting
    fit_df = df.drop(columns=["subject"], errors="ignore")

    print(f"\n[INFO] Training {model_type.upper()} AFT model...")
    model.fit(fit_df, duration_col=duration_col, event_col=event_col)

    # Compute metrics
    predictions = model.predict_median(fit_df)

    # C-index (higher is better)
    c_index = concordance_index(
        df[duration_col],
        -predictions,  # negative because higher prediction = longer survival
        df[event_col],
    )

    # MSE and MAE on events only
    event_mask = fit_df[event_col] == 1
    if event_mask.sum() > 0:
        mse_events = mean_squared_error(
            fit_df.loc[event_mask, duration_col], predictions[event_mask]
        )
        mae_events = mean_absolute_error(
            fit_df.loc[event_mask, duration_col], predictions[event_mask]
        )
        rmse_events = np.sqrt(mse_events)
    else:
        mse_events = mae_events = rmse_events = float("nan")

    # MSE on all subjects
    mse_all = mean_squared_error(fit_df[duration_col], predictions)
    mae_all = mean_absolute_error(fit_df[duration_col], predictions)
    rmse_all = np.sqrt(mse_all)

    print(f"[INFO] Training complete")
    print(f"  C-index: {c_index:.4f}")
    print(f"  MSE (events only): {mse_events:.4f}")
    print(f"  RMSE (events only): {rmse_events:.4f}")
    print(f"  MAE (events only): {mae_events:.4f}")
    print(f"  MSE (all): {mse_all:.4f}")
    print(f"  RMSE (all): {rmse_all:.4f}")
    print(f"  MAE (all): {mae_all:.4f}")
    print(f"  Log-likelihood: {model.log_likelihood_:.2f}")
    print(f"  AIC: {model.AIC_:.2f}")

    metrics = {
        "c_index": c_index,
        "mse_events": mse_events,
        "rmse_events": rmse_events,
        "mae_events": mae_events,
        "mse_all": mse_all,
        "rmse_all": rmse_all,
        "mae_all": mae_all,
        "log_likelihood": model.log_likelihood_,
        "aic": model.AIC_,
    }

    return model, metrics


def interpret_coefficients(model, top_n: int = 20):
    """Print interpretable coefficient summary."""
    print(f"\n{'='*80}")
    print(f"COEFFICIENT INTERPRETATION (AFT Model)")
    print(f"{'='*80}")
    print("\nHow to read:")
    print("  - NEGATIVE coef: higher feature value → SHORTER time to AD (accelerates)")
    print("  - POSITIVE coef: higher feature value → LONGER time to AD (delays)")
    print("  - Acceleration Factor = exp(coef)")
    print("    - AF < 1: accelerates conversion")
    print("    - AF > 1: delays conversion")

    summary = model.summary

    # Get coefficients - handle MultiIndex (param, covariate)
    # Filter to lambda_ parameters only (exclude rho_)
    if isinstance(summary.index, pd.MultiIndex):
        coefs = summary.xs("lambda_", level="param").copy()
        # Exclude intercept
        coefs = coefs[coefs.index != "Intercept"]
    else:
        # Fallback for single index
        coefs = summary[summary.index.str.contains("_bl", na=False)].copy()

    coefs = coefs.sort_values("coef", key=abs, ascending=False)

    print(f"\nTop {top_n} Most Important Features:\n")
    print(f"{'Feature':<40} {'Coef':>8} {'AF':>8} {'p-value':>10} {'Effect'}")
    print("-" * 80)

    for idx, row in coefs.head(top_n).iterrows():
        feat = idx.replace("_bl", "") if "_bl" in str(idx) else str(idx)
        coef = row["coef"]
        af = np.exp(coef)
        p = row["p"]

        # Effect interpretation
        if coef < 0:
            if af < 0.9:
                effect = "Strong accelerator"
            elif af < 0.95:
                effect = "Moderate accelerator"
            else:
                effect = "Weak accelerator"
        else:
            if af > 1.1:
                effect = "Strong delayer"
            elif af > 1.05:
                effect = "Moderate delayer"
            else:
                effect = "Weak delayer"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"{feat:<40} {coef:>8.4f} {af:>8.4f} {p:>10.4f}{sig:>3} {effect}")

    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")


def make_predictions(model, df: pd.DataFrame, out_path: Path):
    """Generate survival predictions for each subject."""
    predictions = []

    for idx, row in df.iterrows():
        subj = row["subject"]
        actual_time = row["time_years"]
        event = row["event"]

        # Predict median survival time
        pred_median = model.predict_median(row.to_frame().T).iloc[0]

        # Predict survival at specific timepoints
        times = [0.5, 1, 2, 3, 5]
        surv_probs = model.predict_survival_function(row.to_frame().T, times=times)

        pred_row = {
            "subject": subj,
            "actual_time": actual_time,
            "event": event,
            "predicted_median_time": pred_median,
        }

        for t in times:
            pred_row[f"survival_prob_{t}yr"] = surv_probs.loc[t].iloc[0]

        predictions.append(pred_row)

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(out_path, index=False)
    print(f"\n[OK] Predictions saved: {out_path}")

    return pred_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to time_to_conversion.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--model",
        default="weibull",
        choices=["weibull", "lognormal", "loglogistic"],
        help="AFT distribution",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Use only top K features by variance (default: all)",
    )
    ap.add_argument(
        "--penalizer", type=float, default=0.1, help="L2 regularization strength"
    )
    ap.add_argument(
        "--no_standardize", action="store_true", help="Skip feature standardization"
    )
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
    df_clean, scaler = prepare_data(df, feat_cols, not args.no_standardize)

    # Keep only necessary columns
    keep_cols = ["subject", "time_years", "event"] + feat_cols
    df_train = df_clean[keep_cols].copy()

    # Train model
    model, metrics = train_aft_model(df_train, args.model, penalizer=args.penalizer)

    # Print summary
    print("\n" + "=" * 80)
    print(model.summary.to_string())
    print("=" * 80)

    # Interpret coefficients
    interpret_coefficients(model, top_n=20)

    # Save model summary
    summary_path = out_dir / f"aft_{args.model}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(model.summary.to_string())
        f.write(f"\n\n{'='*80}\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write(f"{'='*80}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"\n[OK] Summary saved: {summary_path}")

    # Save metrics as JSON
    import json

    metrics_path = out_dir / f"aft_{args.model}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Metrics saved: {metrics_path}")

    # Make predictions
    pred_path = out_dir / f"aft_{args.model}_predictions.csv"
    make_predictions(model, df_train, pred_path)

    # Save feature list
    feat_path = out_dir / f"aft_{args.model}_features.txt"
    with open(feat_path, "w") as f:
        for feat in feat_cols:
            f.write(f"{feat}\n")
    print(f"[OK] Features saved: {feat_path}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model: {args.model.upper()} AFT")
    print(f"C-index: {metrics['c_index']:.4f}")
    print(f"MSE (events): {metrics['mse_events']:.4f}")
    print(f"MSE (all): {metrics['mse_all']:.4f}")
    print(f"RMSE (all): {metrics['rmse_all']:.4f}")
    print(f"Features: {len(feat_cols)}")
    print(f"Subjects: {len(df_train)}")
    print(f"Events: {df_train['event'].sum()}")
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
