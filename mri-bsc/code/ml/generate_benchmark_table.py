
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def run_model_variant(
    slopes_df, survival_df, model_type, top_k, test_size=0.3, random_state=42
):
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
    from lifelines.utils import concordance_index
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import time

    df = survival_df.merge(slopes_df, on="subject", how="inner")

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().sum() > len(df) * 0.8]
    variances = df[slope_cols].var()

    if top_k == -1:
        top_features = slope_cols
    else:
        top_features = variances.nlargest(top_k).index.tolist()

    X = df[top_features].copy().fillna(df[top_features].median())
    y = df[["time_years", "event"]].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y["event"]
    )

    start_time = time.time()

    if model_type == "rsf":
        y_train_surv = Surv.from_dataframe("event", "time_years", y_train)
        y_test_surv = Surv.from_dataframe("event", "time_years", y_test)

        model = RandomSurvivalForest(
            n_estimators=1000,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train_surv)

        train_c = model.score(X_train, y_train_surv)
        test_c = model.score(X_test, y_test_surv)

        train_risk = model.predict(X_train)
        test_risk = model.predict(X_test)

        train_events = y_train["event"] == 1
        test_events = y_test["event"] == 1
        train_rmse = np.nan
        test_rmse = np.nan
        log_likelihood = np.nan

    else:
        if model_type == "weibull":
            model = WeibullAFTFitter(penalizer=0.1)
        elif model_type == "lognormal":
            model = LogNormalAFTFitter(penalizer=0.1)
        elif model_type == "loglogistic":
            model = LogLogisticAFTFitter(penalizer=0.1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        train_df = X_train.copy()
        train_df["time_years"] = y_train["time_years"].values
        train_df["event"] = y_train["event"].values

        test_df = X_test.copy()
        test_df["time_years"] = y_test["time_years"].values
        test_df["event"] = y_test["event"].values

        model.fit(
            train_df, duration_col="time_years", event_col="event", show_progress=False
        )

        train_pred = model.predict_median(train_df)
        test_pred = model.predict_median(test_df)

        train_c = concordance_index(
            y_train["time_years"], -train_pred, y_train["event"]
        )
        test_c = concordance_index(y_test["time_years"], -test_pred, y_test["event"])

        train_events = y_train["event"] == 1
        test_events = y_test["event"] == 1

        if train_events.sum() > 0:
            train_rmse = np.sqrt(
                np.mean(
                    (
                        y_train.loc[train_events, "time_years"].values
                        - train_pred[train_events.values]
                    )
                    ** 2
                )
            )
        else:
            train_rmse = np.nan

        if test_events.sum() > 0:
            test_rmse = np.sqrt(
                np.mean(
                    (
                        y_test.loc[test_events, "time_years"].values
                        - test_pred[test_events.values]
                    )
                    ** 2
                )
            )
        else:
            test_rmse = np.nan

        log_likelihood = model.log_likelihood_

    train_time = time.time() - start_time

    overfit_gap = train_c - test_c

    n_train_events = y_train["event"].sum()
    n_test_events = y_test["event"].sum()

    return {
        "model": model_type,
        "top_k": top_k if top_k != -1 else len(top_features),
        "n_features": len(top_features),
        "train_c": train_c,
        "test_c": test_c,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "overfit_gap": overfit_gap,
        "log_likelihood": log_likelihood,
        "train_time_sec": train_time,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_events": n_train_events,
        "n_test_events": n_test_events,
    }

def generate_benchmark_table(slopes_path, survival_path, out_dir):
    print("=" * 80)
    print("GENERATING COMPREHENSIVE BENCHMARK TABLE")
    print("=" * 80)

    slopes = pd.read_csv(slopes_path)
    survival = pd.read_csv(survival_path)

    configs = [
        ("rsf", 5, "RSF-5"),
        ("rsf", 10, "RSF-10"),
        ("rsf", 15, "RSF-15"),
        ("rsf", 20, "RSF-20"),
        ("rsf", 25, "RSF-25"),
        ("rsf", 30, "RSF-30"),
        ("rsf", -1, "RSF-All"),
        ("weibull", 20, "Weibull-20"),
        ("lognormal", 20, "LogNorm-20"),
        ("loglogistic", 20, "LogLog-20"),
        ("weibull", 10, "Weibull-10"),
        ("weibull", 30, "Weibull-30"),
    ]

    results = []

    for model_type, top_k, name in configs:
        print(f"\nRunning: {name} ({model_type}, top_k={top_k})")
        try:
            metrics = run_model_variant(
                slopes, survival, model_type, top_k, test_size=0.3, random_state=42
            )
            metrics["config_name"] = name
            results.append(metrics)
            print(f"   Test C-index: {metrics['test_c']:.4f}")
        except Exception as e:
            print(f"   Failed: {e}")

    df = pd.DataFrame(results)

    col_order = [
        "config_name",
        "model",
        "n_features",
        "train_c",
        "test_c",
        "overfit_gap",
        "test_rmse",
        "log_likelihood",
        "train_time_sec",
        "n_train",
        "n_test",
        "n_train_events",
        "n_test_events",
    ]
    df = df[col_order]

    df = df.sort_values("test_c", ascending=False)

    csv_path = out_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n Saved raw results: {csv_path}")

    create_latex_table(df, out_dir)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (Top 5 by Test C-index)")
    print("=" * 80)
    print(
        df[["config_name", "test_c", "train_c", "overfit_gap", "n_features"]]
        .head()
        .to_string(index=False)
    )
    print("=" * 80)

    return df

def create_latex_table(df, out_dir):

    best_test_c = df["test_c"].max()
    best_train_c = df["train_c"].max()
    min_overfit = df["overfit_gap"].min()

    latex_rows = []

    for _, row in df.iterrows():
        test_c_str = f"{row['test_c']:.3f}"
        train_c_str = f"{row['train_c']:.3f}"
        overfit_str = f"{row['overfit_gap']:.3f}"

        if row["test_c"] == best_test_c:
            test_c_str = f"\\textbf{{{test_c_str}}}"

        latex_row = (
            f"{row['config_name']} & "
            f"{row['n_features']} & "
            f"{train_c_str} & "
            f"{test_c_str} & "
            f"{overfit_str} & "
            f"{row['test_rmse']:.2f} & "
            f"{row['train_time_sec']:.1f} \\\\"
        )

        latex_rows.append(latex_row)

    latex_table = (
        r"""
\begin{table*}[t!]
\centering
\caption{Benchmark comparison of model variants and feature selection strategies. 
Best test C-index values are shown in bold. Lower overfitting gap indicates better generalization.}
\label{tab:benchmark}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Features} & \textbf{Train C} & \textbf{Test C} & \textbf{Overfit Gap} & \textbf{Test RMSE} & \textbf{Time (s)} \\
\hline
"""
        + "\n".join(latex_rows)
        + r"""
\hline
\end{tabular}
\end{table*}
"""
    )

    latex_path = out_dir / "benchmark_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)

    print(f" Saved LaTeX table: {latex_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slopes", required=True, help="Path to slopes CSV")
    parser.add_argument("--survival", required=True, help="Path to survival CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_benchmark_table(args.slopes, args.survival, out_dir)

    print("\n Benchmark table generation complete!")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
