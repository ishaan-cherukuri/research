
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

class SurvivalXGBPipeline:

    def __init__(self, data_path="features_all_456.csv", output_dir="survival_models"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.config = {}

    def load_and_prepare_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} subjects")

        print(f"\nEvent distribution:")
        print(df["event_observed"].value_counts())
        print(f"\nMean follow-up time: {df['event_time_years'].mean():.2f} years")

        return df

    def engineer_features(self, df):
        print("\n=== Feature Engineering ===")

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
        print(f"Total features: {len(feature_cols)}")

        X = df[feature_cols].copy()

        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            print(
                f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}"
            )
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            print(f"Filling {missing_before} missing values with median")
            X = X.fillna(X.median())

        print(f"Final feature matrix: {X.shape}")

        return X, feature_cols

    def prepare_aft_targets(self, df):

        y_lower = df["aft_y_lower"].values
        y_upper = df["aft_y_upper"].values

        y_upper = np.where(y_upper == "inf", np.inf, y_upper.astype(float))

        print(f"\n=== AFT Targets ===")
        print(f"Events: {(df['event_observed'] == 1).sum()}")
        print(f"Censored: {(df['event_observed'] == 0).sum()}")
        print(
            f"Mean event time: {df.loc[df['event_observed'] == 1, 'event_time_years'].mean():.2f} years"
        )
        print(
            f"Mean censor time: {df.loc[df['event_observed'] == 0, 'event_time_years'].mean():.2f} years"
        )

        return y_lower, y_upper

    def split_data(self, X, y_lower, y_upper, test_size=0.2, random_state=42):
        print(f"\n=== Data Split ===")

        X_train, X_test, y_lower_train, y_lower_test, y_upper_train, y_upper_test = (
            train_test_split(
                X,
                y_lower,
                y_upper,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        )

        print(f"Train: {len(X_train)} samples")
        print(f"Test: {len(X_test)} samples")

        return X_train, X_test, y_lower_train, y_lower_test, y_upper_train, y_upper_test

    def scale_features(self, X_train, X_test):
        print("\n=== Feature Scaling ===")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Scaled features to mean=0, std=1")

        return X_train_scaled, X_test_scaled

    def train_aft_model(self, X_train, y_lower_train, y_upper_train, params=None):
        print("\n=== Training XGBoost AFT Model ===")

        default_params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        }

        if params:
            default_params.update(params)

        self.config = default_params.copy()

        dtrain = xgb.DMatrix(X_train)
        dtrain.set_float_info("label_lower_bound", y_lower_train)
        dtrain.set_float_info("label_upper_bound", y_upper_train)

        print(f"Training with parameters:")
        for k, v in default_params.items():
            print(f"  {k}: {v}")

        self.model = xgb.train(
            default_params,
            dtrain,
            num_boost_round=default_params.pop("n_estimators", 200),
            verbose_eval=25,
        )

        print("\nTraining complete!")
        return self.model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)

        return predictions

    def evaluate_model(self, X_test, y_lower_test, y_upper_test, event_observed_test):
        print("\n=== Model Evaluation ===")

        y_pred = self.predict(X_test)

        print(f"\nPredicted survival time statistics:")
        print(f"  Mean: {np.mean(y_pred):.3f}")
        print(f"  Std: {np.std(y_pred):.3f}")
        print(f"  Min: {np.min(y_pred):.3f}")
        print(f"  Max: {np.max(y_pred):.3f}")

        c_index = self._concordance_index(y_lower_test, -y_pred, event_observed_test)
        if not np.isnan(c_index):
            print(f"\nC-index: {c_index:.4f}")

        actual_times = y_lower_test
        corr = np.corrcoef(y_pred, actual_times)[0, 1]
        print(f"Correlation with actual time: {corr:.4f}")

        return {
            "predictions": y_pred,
            "actual_lower": y_lower_test,
            "actual_upper": y_upper_test,
            "event_observed": event_observed_test,
        }

    @staticmethod
    def _concordance_index(event_times, risk_scores, event_observed):
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

    def plot_feature_importance(self, top_n=30):
        print("\n=== Feature Importance ===")

        importance = self.model.get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            [{"feature": k, "importance": v} for k, v in importance.items()]
        ).sort_values("importance", ascending=False)

        print(f"\nTop {top_n} features:")
        print(importance_df.head(top_n).to_string(index=False))

        plt.figure(figsize=(10, 12))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features["importance"])
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Importance (Gain)")
        plt.title(f"Top {top_n} Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()

        return importance_df

    def plot_survival_predictions(self, results):
        print("\n=== Plotting Results ===")

        y_pred = results["predictions"]
        y_actual = results["actual_lower"]
        events = results["event_observed"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax = axes[0, 0]
        scatter = ax.scatter(y_actual, y_pred, c=events, cmap="RdYlBu_r", alpha=0.6)
        ax.plot(
            [y_actual.min(), y_actual.max()],
            [y_actual.min(), y_actual.max()],
            "k--",
            lw=2,
            alpha=0.5,
        )
        ax.set_xlabel("Actual Time (years)")
        ax.set_ylabel("Predicted Time (years)")
        ax.set_title("Predicted vs Actual Survival Time")
        plt.colorbar(scatter, ax=ax, label="Event (1=AD, 0=Censored)")

        ax = axes[0, 1]
        event_mask = events == 1
        ax.hist(
            y_pred[event_mask],
            bins=20,
            alpha=0.7,
            label="Events (converted)",
            color="red",
        )
        ax.hist(y_pred[~event_mask], bins=20, alpha=0.7, label="Censored", color="blue")
        ax.set_xlabel("Predicted Survival Time (years)")
        ax.set_ylabel("Count")
        ax.set_title("Predicted Time Distribution by Event Status")
        ax.legend()

        ax = axes[1, 0]
        residuals = y_actual - y_pred
        ax.scatter(y_pred, residuals, c=events, cmap="RdYlBu_r", alpha=0.6)
        ax.axhline(0, color="k", linestyle="--", lw=2, alpha=0.5)
        ax.set_xlabel("Predicted Time (years)")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title("Residual Plot")

        ax = axes[1, 1]
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals")

        plt.tight_layout()
        output_path = self.output_dir / "survival_predictions.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
        plt.close()

    def save_model(self, model_name="xgb_aft_model"):
        print("\n=== Saving Model ===")

        model_path = self.output_dir / f"{model_name}.json"
        self.model.save_model(str(model_path))
        print(f"Saved model to {model_path}")

        import pickle

        scaler_path = self.output_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved config to {config_path}")

        metadata = {
            "train_date": datetime.now().isoformat(),
            "data_path": str(self.data_path),
            "n_features": len(self.scaler.mean_)
            if hasattr(self.scaler, "mean_")
            else None,
            "config": self.config,
        }
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def cross_validate(self, X, y_lower, y_upper, n_folds=5):
        print(f"\n=== {n_folds}-Fold Cross-Validation ===")

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\nFold {fold}/{n_folds}")

            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_lower_train_fold = y_lower[train_idx]
            y_lower_val_fold = y_lower[val_idx]
            y_upper_train_fold = y_upper[train_idx]
            y_upper_val_fold = y_upper[val_idx]

            scaler_fold = StandardScaler()
            X_train_fold = scaler_fold.fit_transform(X_train_fold)
            X_val_fold = scaler_fold.transform(X_val_fold)

            dtrain = xgb.DMatrix(X_train_fold)
            dtrain.set_float_info("label_lower_bound", y_lower_train_fold)
            dtrain.set_float_info("label_upper_bound", y_upper_train_fold)

            dval = xgb.DMatrix(X_val_fold)
            dval.set_float_info("label_lower_bound", y_lower_val_fold)
            dval.set_float_info("label_upper_bound", y_upper_val_fold)

            model_fold = xgb.train(
                self.config,
                dtrain,
                num_boost_round=200,
                evals=[(dtrain, "train"), (dval, "val")],
                verbose_eval=False,
            )

            y_pred_val = model_fold.predict(dval)
            corr = np.corrcoef(y_pred_val, y_lower_val_fold)[0, 1]

            print(f"  Validation correlation: {corr:.4f}")
            fold_scores.append(corr)

        print(f"\n=== CV Results ===")
        print(
            f"Mean correlation: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}"
        )

        return fold_scores

def main():
    print("=" * 60)
    print("XGBoost AFT Survival Analysis Pipeline")
    print("MCI to AD Conversion Prediction")
    print("=" * 60)

    pipeline = SurvivalXGBPipeline(data_path="features_all_456.csv")

    df = pipeline.load_and_prepare_data()

    X, feature_cols = pipeline.engineer_features(df)

    y_lower, y_upper = pipeline.prepare_aft_targets(df)

    X_train, X_test, y_lower_train, y_lower_test, y_upper_train, y_upper_test = (
        pipeline.split_data(X, y_lower, y_upper, test_size=0.2)
    )

    test_indices = X_test.index
    event_observed_test = df.loc[test_indices, "event_observed"].values

    X_train_scaled, X_test_scaled = pipeline.scale_features(X_train, X_test)

    pipeline.train_aft_model(X_train_scaled, y_lower_train, y_upper_train)

    results = pipeline.evaluate_model(
        X_test_scaled, y_lower_test, y_upper_test, event_observed_test
    )

    importance_df = pipeline.plot_feature_importance(top_n=30)

    pipeline.plot_survival_predictions(results)

    pipeline.save_model()

    print("\n" + "=" * 60)
    cv_option = input("Run cross-validation? (y/n): ").strip().lower()
    if cv_option == "y":
        X_full_scaled = pipeline.scaler.fit_transform(X)
        cv_scores = pipeline.cross_validate(X_full_scaled, y_lower, y_upper, n_folds=5)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Model saved to: {pipeline.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
