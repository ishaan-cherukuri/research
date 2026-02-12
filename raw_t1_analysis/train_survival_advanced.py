#!/usr/bin/env python3
"""
Advanced XGBoost Survival Analysis Training Pipeline
Optimized for small sample sizes with multiple evaluation strategies
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


class AdvancedSurvivalPipeline:
    """Advanced XGBoost AFT Pipeline with multiple evaluation strategies"""

    def __init__(self, data_path="features_all_456.csv", output_dir="survival_models"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.config = {}
        self.cv_results = []

    def load_data(self):
        """Load and prepare data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} subjects")

        # Event statistics
        n_events = (df["event_observed"] == 1).sum()
        n_censored = (df["event_observed"] == 0).sum()

        print(f"\n{'=' * 60}")
        print(f"Dataset Statistics:")
        print(f"  Total subjects: {len(df)}")
        print(f"  Events (MCI→AD): {n_events} ({n_events / len(df) * 100:.1f}%)")
        print(f"  Censored: {n_censored} ({n_censored / len(df) * 100:.1f}%)")
        print(f"  Mean follow-up: {df['event_time_years'].mean():.2f} years")
        print(f"  Event rate: {n_events}/{len(df)} = {n_events / len(df):.3f}")
        print(f"{'=' * 60}\n")

        return df

    def prepare_features(self, df):
        """Feature engineering"""
        exclude_cols = [
            "subject_id",
            "mci_bl_datetime",
            "baseline_diagnosis",
            "event_observed",
            "event_datetime",
            "censor_datetime",
            "event_time_years",
            "aft_y_lower",
            "aft_y_upper",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            print(f"Encoding {len(categorical_cols)} categorical columns")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Handle missing values
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"Filling {missing_count} missing values with median")
            X = X.fillna(X.median())

        print(f"Feature matrix: {X.shape}")

        # Prepare targets
        y_lower = df["aft_y_lower"].values
        y_upper = df["aft_y_upper"].values
        y_upper = np.where(y_upper == "inf", np.inf, y_upper.astype(float))
        event_observed = df["event_observed"].values

        return X, y_lower, y_upper, event_observed

    def get_default_params(self):
        """Default AFT parameters"""
        return {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }

    def evaluate_fold(self, y_pred, y_true, event_observed):
        """Compute evaluation metrics for a fold"""
        metrics = {}

        # Correlation
        metrics["correlation"] = np.corrcoef(y_pred, y_true)[0, 1]

        # MAE
        metrics["mae"] = np.mean(np.abs(y_pred - y_true))

        # RMSE
        metrics["rmse"] = np.sqrt(np.mean((y_pred - y_true) ** 2))

        # C-index (using -pred_time as risk score)
        metrics["c_index"] = self._concordance_index(y_true, -y_pred, event_observed)

        # Concordance for events only
        event_mask = event_observed == 1
        if event_mask.sum() > 1:
            metrics["event_correlation"] = np.corrcoef(
                y_pred[event_mask], y_true[event_mask]
            )[0, 1]
        else:
            metrics["event_correlation"] = np.nan

        return metrics

    @staticmethod
    def _concordance_index(event_times, risk_scores, event_observed):
        """Compute Harrell's C-index for right-censored data."""
        n_total = 0
        n_concordant = 0
        n_tied = 0

        event_times = np.asarray(event_times)
        risk_scores = np.asarray(risk_scores)
        event_observed = np.asarray(event_observed)

        for i in range(len(event_times)):
            if event_observed[i] != 1:
                continue
            for j in range(len(event_times)):
                if event_times[i] >= event_times[j]:
                    continue
                n_total += 1
                if risk_scores[i] > risk_scores[j]:
                    n_concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    n_tied += 1

        if n_total == 0:
            return np.nan

        return (n_concordant + 0.5 * n_tied) / n_total

    def stratified_cv(
        self, X, y_lower, y_upper, event_observed, n_folds=5, n_boost=200
    ):
        """
        Stratified K-Fold Cross-Validation
        Ensures balanced event/censored ratio in each fold
        """
        print(f"\n{'=' * 60}")
        print(f"Strategy 1: STRATIFIED {n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'=' * 60}")
        print("✓ Uses ALL data for training and validation")
        print("✓ Balanced event distribution in each fold")
        print("✓ Best for unbiased performance estimation\n")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, event_observed), 1):
            print(f"Fold {fold}/{n_folds}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_lower_train, y_lower_val = y_lower[train_idx], y_lower[val_idx]
            y_upper_train, y_upper_val = y_upper[train_idx], y_upper[val_idx]
            event_train, event_val = event_observed[train_idx], event_observed[val_idx]

            print(f"  Train: {len(train_idx)} ({event_train.sum()} events)")
            print(f"  Val: {len(val_idx)} ({event_val.sum()} events)")

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train
            dtrain = xgb.DMatrix(X_train_scaled)
            dtrain.set_float_info("label_lower_bound", y_lower_train)
            dtrain.set_float_info("label_upper_bound", y_upper_train)

            dval = xgb.DMatrix(X_val_scaled)
            dval.set_float_info("label_lower_bound", y_lower_val)
            dval.set_float_info("label_upper_bound", y_upper_val)

            params = self.get_default_params()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_boost,
                evals=[(dtrain, "train"), (dval, "val")],
                verbose_eval=False,
            )

            # Predict and evaluate
            y_pred_val = model.predict(dval)
            metrics = self.evaluate_fold(y_pred_val, y_lower_val, event_val)

            fold_results.append(metrics)

            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  C-index: {metrics['c_index']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.2f} years\n")

        # Summary
        print(f"\n{'=' * 60}")
        print("STRATIFIED CV RESULTS:")
        print(f"{'=' * 60}")
        for metric in ["correlation", "c_index", "rmse", "mae"]:
            values = [r[metric] for r in fold_results if not np.isnan(r[metric])]
            if values:
                print(
                    f"{metric.upper():20s}: {np.mean(values):.4f} ± {np.std(values):.4f}"
                )
        print(f"{'=' * 60}\n")

        self.cv_results = fold_results
        return fold_results

    def holdout_validation(
        self, X, y_lower, y_upper, event_observed, test_size=0.2, n_boost=200
    ):
        """
        Traditional train/test split
        """
        print(f"\n{'=' * 60}")
        print(f"Strategy 2: HOLD-OUT VALIDATION ({int(test_size * 100)}% test)")
        print(f"{'=' * 60}")
        print("✓ Simple, fast")
        print("✗ Wastes data (only ~365 for training)")
        print("✗ Performance varies by random split\n")

        # Split
        train_idx, test_idx = train_test_split(
            range(len(X)), test_size=test_size, random_state=42, stratify=event_observed
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_lower_train, y_lower_test = y_lower[train_idx], y_lower[test_idx]
        y_upper_train, y_upper_test = y_upper[train_idx], y_upper[test_idx]
        event_train, event_test = event_observed[train_idx], event_observed[test_idx]

        print(f"Train: {len(train_idx)} samples ({event_train.sum()} events)")
        print(f"Test: {len(test_idx)} samples ({event_test.sum()} events)\n")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        dtrain = xgb.DMatrix(X_train_scaled)
        dtrain.set_float_info("label_lower_bound", y_lower_train)
        dtrain.set_float_info("label_upper_bound", y_upper_train)

        dtest = xgb.DMatrix(X_test_scaled)
        dtest.set_float_info("label_lower_bound", y_lower_test)
        dtest.set_float_info("label_upper_bound", y_upper_test)

        params = self.get_default_params()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_boost,
            evals=[(dtrain, "train"), (dtest, "test")],
            verbose_eval=25,
        )

        # Evaluate
        y_pred_test = model.predict(dtest)
        metrics = self.evaluate_fold(y_pred_test, y_lower_test, event_test)

        print(f"\n{'=' * 60}")
        print("HOLD-OUT TEST RESULTS:")
        print(f"{'=' * 60}")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric.upper():20s}: {value:.4f}")
        print(f"{'=' * 60}\n")

        return (
            model,
            scaler,
            metrics,
            (X_test_scaled, y_pred_test, y_lower_test, event_test),
        )

    def train_final_model(self, X, y_lower, y_upper, n_boost=200):
        """
        Train on ALL data for final deployment model
        """
        print(f"\n{'=' * 60}")
        print("Strategy 3: TRAIN ON ALL DATA (Final Deployment Model)")
        print(f"{'=' * 60}")
        print("✓ Uses maximum information (456 subjects)")
        print("✓ Best predictions for deployment")
        print("✗ No held-out performance estimate")
        print("  (Use CV results for performance reporting)\n")

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Train
        dtrain = xgb.DMatrix(X_scaled)
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        params = self.get_default_params()
        self.config = params.copy()

        print("Training final model on all data...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_boost,
            evals=[(dtrain, "train")],
            verbose_eval=25,
        )

        print("\n✓ Final model trained on all 456 subjects!")
        print(f"{'=' * 60}\n")

        return self.model

    def plot_cv_results(self):
        """Visualize cross-validation results"""
        if not self.cv_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics = ["correlation", "c_index", "rmse", "mae"]
        titles = [
            "Correlation",
            "C-index (Risk Discrimination)",
            "RMSE (years)",
            "MAE (years)",
        ]

        for ax, metric, title in zip(axes.flat, metrics, titles):
            values = [r[metric] for r in self.cv_results if not np.isnan(r[metric])]

            ax.bar(range(1, len(values) + 1), values, alpha=0.7, color="steelblue")
            ax.axhline(
                np.mean(values),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(values):.4f}",
            )
            ax.set_xlabel("Fold")
            ax.set_ylabel(title)
            ax.set_title(f"{title} Across Folds")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "cv_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved CV results to {output_path}")
        plt.close()

    def plot_predictions(self, test_data):
        """Visualize test predictions"""
        X_test, y_pred, y_true, events = test_data

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Predicted vs Actual
        ax = axes[0, 0]
        scatter = ax.scatter(y_true, y_pred, c=events, cmap="RdYlBu_r", alpha=0.6, s=50)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=2, alpha=0.5, label="Perfect prediction")
        ax.set_xlabel("Actual Time (years)")
        ax.set_ylabel("Predicted Time (years)")
        ax.set_title("Predicted vs Actual Survival Time")
        ax.legend()
        plt.colorbar(scatter, ax=ax, label="Event")

        # 2. Distribution by event status
        ax = axes[0, 1]
        event_mask = events == 1
        ax.hist(
            y_pred[event_mask],
            bins=15,
            alpha=0.7,
            label=f"Events (n={event_mask.sum()})",
            color="red",
            edgecolor="black",
        )
        ax.hist(
            y_pred[~event_mask],
            bins=15,
            alpha=0.7,
            label=f"Censored (n={(~event_mask).sum()})",
            color="blue",
            edgecolor="black",
        )
        ax.set_xlabel("Predicted Survival Time (years)")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Time by Event Status")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Residuals
        ax = axes[1, 0]
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, c=events, cmap="RdYlBu_r", alpha=0.6, s=50)
        ax.axhline(0, color="k", linestyle="--", lw=2, alpha=0.5)
        ax.set_xlabel("Predicted Time (years)")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"Residual Plot (Mean: {residuals.mean():.2f})")
        ax.grid(True, alpha=0.3)

        # 4. Box plot by event
        ax = axes[1, 1]
        data_to_plot = [y_pred[~event_mask], y_pred[event_mask]]
        bp = ax.boxplot(data_to_plot, labels=["Censored", "Events"], patch_artist=True)
        bp["boxes"][0].set_facecolor("blue")
        bp["boxes"][1].set_facecolor("red")
        ax.set_ylabel("Predicted Survival Time (years)")
        ax.set_title("Predicted Time Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        output_path = self.output_dir / "test_predictions.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved predictions plot to {output_path}")
        plt.close()

    def save_model(self, model_name="final_model"):
        """Save model artifacts"""
        print(f"\nSaving model to {self.output_dir}/...")

        # Model
        model_path = self.output_dir / f"{model_name}.json"
        self.model.save_model(str(model_path))
        print(f"  ✓ Model: {model_path.name}")

        # Scaler
        import pickle

        scaler_path = self.output_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Scaler: {scaler_path.name}")

        # Config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"  ✓ Config: {config_path.name}")

        # CV Results
        if self.cv_results:
            cv_path = self.output_dir / "cv_results.json"
            cv_summary = {
                "n_folds": len(self.cv_results),
                "metrics": {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    r[metric]
                                    for r in self.cv_results
                                    if not np.isnan(r[metric])
                                ]
                            )
                        ),
                        "std": float(
                            np.std(
                                [
                                    r[metric]
                                    for r in self.cv_results
                                    if not np.isnan(r[metric])
                                ]
                            )
                        ),
                        "folds": [
                            float(r[metric]) if not np.isnan(r[metric]) else None
                            for r in self.cv_results
                        ],
                    }
                    for metric in ["correlation", "c_index", "rmse", "mae"]
                },
            }
            with open(cv_path, "w") as f:
                json.dump(cv_summary, f, indent=2)
            print(f"  ✓ CV Results: {cv_path.name}")

        print()


def main():
    """Main pipeline with evaluation strategies"""
    print("=" * 60)
    print("ADVANCED SURVIVAL ANALYSIS PIPELINE")
    print("Multiple Evaluation Strategies for Small Datasets")
    print("=" * 60)

    pipeline = AdvancedSurvivalPipeline()

    # Load data
    df = pipeline.load_data()
    X, y_lower, y_upper, event_observed = pipeline.prepare_features(df)

    print("\n" + "=" * 60)
    print("EVALUATION OPTIONS:")
    print("=" * 60)
    print("1. Stratified CV     - Best for performance estimation")
    print("2. Hold-out split    - Simple train/test")
    print("3. Train on all data - Best model for deployment")
    print("=" * 60)

    choice = input("\nSelect strategy (1/2/3/all): ").strip().lower()

    if choice in ["1", "all"]:
        # Stratified CV
        cv_results = pipeline.stratified_cv(
            X, y_lower, y_upper, event_observed, n_folds=5
        )
        pipeline.plot_cv_results()

    if choice in ["2", "all"]:
        # Hold-out
        model_holdout, scaler_holdout, metrics_holdout, test_data = (
            pipeline.holdout_validation(
                X, y_lower, y_upper, event_observed, test_size=0.2
            )
        )
        pipeline.plot_predictions(test_data)

    if choice in ["3", "all"]:
        # Final model on all data
        final_model = pipeline.train_final_model(X, y_lower, y_upper)
        pipeline.save_model("final_model_all_data")

    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("1. Report CV performance (unbiased estimate)")
    print("2. Train final model on ALL 456 subjects")
    print("3. Use final model for predictions")
    print("=" * 60)
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
