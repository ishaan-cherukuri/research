
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_data(features_csv: str, manifest_csv: str) -> pd.DataFrame:

    features = pd.read_csv(features_csv)
    print(f"[INFO] Loaded {len(features)} scans with features")

    manifest = pd.read_csv(manifest_csv)

    df = features.merge(
        manifest[["subject", "visit_code", "acq_date", "diagnosis"]],
        left_on=["subject", "visit_code", "acq_date"],
        right_on=["subject", "visit_code", "acq_date"],
        how="left",
    )

    df = df[df["diagnosis"].notna()].copy()

    print(f"\n[INFO] After merge: {len(df)} scans with diagnosis")
    print(f"  Diagnosis distribution:")
    for dx, count in df["diagnosis"].value_counts().sort_index().items():
        print(f"    {dx}: {count}")

    return df

def map_diagnosis_to_classes(df: pd.DataFrame, task: str = "3class") -> pd.DataFrame:

    if task == "3class":
        df["class"] = df["diagnosis"].map({0: "CN", 1: "MCI", 2: "MCI", 3: "AD"})
        print(f"\n[INFO] 3-class problem: CN vs MCI vs AD")

    elif task == "2class":
        df["class"] = df["diagnosis"].map(
            {0: "CN/MCI", 1: "CN/MCI", 2: "CN/MCI", 3: "AD"}
        )
        print(f"\n[INFO] 2-class problem: CN/MCI vs AD")

    elif task == "binary_cn_ad":
        df = df[df["diagnosis"].isin([0, 3])].copy()
        df["class"] = df["diagnosis"].map({0: "CN", 3: "AD"})
        print(f"\n[INFO] Binary problem: CN vs AD only")

    df = df[df["class"].notna()].copy()

    print(f"  Final class distribution:")
    for cls, count in df["class"].value_counts().sort_index().items():
        print(f"    {cls}: {count}")

    return df

def select_features(
    df: pd.DataFrame, exclude_cols: set[str], top_k: Optional[int] = None
) -> list[str]:

    feat_cols = [c for c in df.columns if c not in exclude_cols]

    if top_k and len(feat_cols) > top_k:
        print(f"\n[INFO] Selecting top {top_k} features by variance...")
        variances = df[feat_cols].var()
        valid = variances[variances > 0].sort_values(ascending=False)
        feat_cols = valid.head(top_k).index.tolist()
        print(f"  Selected: {len(feat_cols)} features")

    return feat_cols

def prepare_data(
    df: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:

    df_clean = df.copy()

    for col in feat_cols:
        if df_clean[col].isna().any() or np.isinf(df_clean[col]).any():
            median = df_clean[col].replace([np.inf, -np.inf], np.nan).median()
            df_clean[col] = (
                df_clean[col].replace([np.inf, -np.inf], np.nan).fillna(median)
            )

    scaler = StandardScaler()
    df_clean[feat_cols] = scaler.fit_transform(df_clean[feat_cols])
    print("[INFO] Features standardized")

    return df_clean, scaler

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:

    print("\n[INFO] Training Logistic Regression...")

    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)

    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Train Balanced Accuracy: {train_bal_acc:.4f}")
    print(f"  Test Balanced Accuracy: {test_bal_acc:.4f}")

    print("\n  Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_test)

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_bal_acc": train_bal_acc,
        "test_bal_acc": test_bal_acc,
        "confusion_matrix": cm,
        "y_pred_test": y_pred_test,
    }

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:

    print("\n[INFO] Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)

    print(f"\n  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Train Balanced Accuracy: {train_bal_acc:.4f}")
    print(f"  Test Balanced Accuracy: {test_bal_acc:.4f}")

    print("\n  Test Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred_test)

    feat_imp = model.feature_importances_

    return {
        "model": model,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_bal_acc": train_bal_acc,
        "test_bal_acc": test_bal_acc,
        "confusion_matrix": cm,
        "y_pred_test": y_pred_test,
        "feature_importance": feat_imp,
    }

def cross_validate_model(
    X: np.ndarray, y: np.ndarray, model_type: str = "logistic"
) -> dict:

    print(f"\n[INFO] Running 5-fold cross-validation ({model_type})...")

    if model_type == "logistic":
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        model, X, y, cv=skf, scoring="balanced_accuracy", n_jobs=-1
    )

    print(f"  CV Balanced Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path):

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Confusion matrix saved: {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to bsc_simple_features.csv")
    ap.add_argument("--manifest", required=True, help="Path to manifest CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument(
        "--task",
        default="3class",
        choices=["3class", "2class", "binary_cn_ad"],
        help="Classification task",
    )
    ap.add_argument(
        "--top_k", type=int, default=None, help="Use only top K features by variance"
    )
    ap.add_argument(
        "--test_size", type=float, default=0.2, help="Test set size (default 0.2)"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_data(args.features, args.manifest)

    df = map_diagnosis_to_classes(df, args.task)

    exclude = {"subject", "visit_code", "acq_date", "image_id", "diagnosis", "class"}
    feat_cols = select_features(df, exclude, args.top_k)
    print(f"\n[INFO] Using {len(feat_cols)} features")

    df_clean, scaler = prepare_data(df, feat_cols)

    le = LabelEncoder()
    y = le.fit_transform(df_clean["class"])
    class_names = le.classes_.tolist()

    X = df_clean[feat_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    print(f"\n[INFO] Train: {len(X_train)}, Test: {len(X_test)}")

    lr_results = train_logistic_regression(
        X_train, y_train, X_test, y_test, class_names
    )

    rf_results = train_random_forest(X_train, y_train, X_test, y_test, class_names)

    lr_cv = cross_validate_model(X, y, "logistic")
    rf_cv = cross_validate_model(X, y, "random_forest")

    results = {
        "task": args.task,
        "n_features": len(feat_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_names": class_names,
        "logistic_regression": {
            "train_acc": lr_results["train_acc"],
            "test_acc": lr_results["test_acc"],
            "train_bal_acc": lr_results["train_bal_acc"],
            "test_bal_acc": lr_results["test_bal_acc"],
            "cv_mean": lr_cv["cv_mean"],
            "cv_std": lr_cv["cv_std"],
        },
        "random_forest": {
            "train_acc": rf_results["train_acc"],
            "test_acc": rf_results["test_acc"],
            "train_bal_acc": rf_results["train_bal_acc"],
            "test_bal_acc": rf_results["test_bal_acc"],
            "cv_mean": rf_cv["cv_mean"],
            "cv_std": rf_cv["cv_std"],
        },
    }

    results_path = out_dir / "classification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved: {results_path}")

    plot_confusion_matrix(
        lr_results["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix_logistic.png",
    )
    plot_confusion_matrix(
        rf_results["confusion_matrix"],
        class_names,
        out_dir / "confusion_matrix_rf.png",
    )

    if "feature_importance" in rf_results:
        feat_imp_df = pd.DataFrame(
            {"feature": feat_cols, "importance": rf_results["feature_importance"]}
        ).sort_values("importance", ascending=False)

        imp_path = out_dir / "rf_feature_importance.csv"
        feat_imp_df.to_csv(imp_path, index=False)
        print(f"[OK] Feature importance saved: {imp_path}")

        print(f"\n{'='*80}")
        print("TOP 20 FEATURES (Random Forest)")
        print(f"{'='*80}")
        print(feat_imp_df.head(20).to_string(index=False))

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Task: {args.task}")
    print(f"Classes: {class_names}")
    print()
    print(f"Logistic Regression:")
    print(f"  Test Balanced Accuracy: {lr_results['test_bal_acc']:.4f}")
    print(f"  CV Balanced Accuracy: {lr_cv['cv_mean']:.4f} ± {lr_cv['cv_std']:.4f}")
    print()
    print(f"Random Forest:")
    print(f"  Test Balanced Accuracy: {rf_results['test_bal_acc']:.4f}")
    print(f"  CV Balanced Accuracy: {rf_cv['cv_mean']:.4f} ± {rf_cv['cv_std']:.4f}")

if __name__ == "__main__":
    main()
