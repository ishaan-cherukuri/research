"""
Diagnose why the survival models are performing poorly.

Key questions:
1. Are slope features actually predictive?
2. Is there a data quality issue?
3. Do we need different features or preprocessing?
4. Should we use different train/test split or cross-validation?
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading data...")
survival = pd.read_csv("data/ml/survival/time_to_conversion.csv")
slopes = pd.read_csv("data/index/bsc_longitudinal_slopes.csv")

# Merge
df = survival[["subject", "event", "time_years"]].merge(
    slopes, on="subject", how="inner"
)

print(f"\nDataset: {len(df)} subjects")
print(f"  Events (converters): {df['event'].sum()}")
print(f"  Censored (stable): {(df['event'] == 0).sum()}")

# Check slope features
slope_cols = [c for c in df.columns if c.endswith("_slope")]
print(f"\nSlope features: {len(slope_cols)}")

# Missing data check
missing = df[slope_cols].isna().sum()
print(f"\nMissing data:")
print(f"  Features with >20% missing: {(missing > len(df) * 0.2).sum()}")
print(f"  Features with >50% missing: {(missing > len(df) * 0.5).sum()}")

# Check if slopes differ between converters and stable
print(f"\n{'='*80}")
print("UNIVARIATE TESTS: Do slopes differ between converters vs stable?")
print(f"{'='*80}")

converters = df[df["event"] == 1]
stable = df[df["event"] == 0]

# Test top features
results = []
for col in slope_cols:
    if df[col].notna().sum() < len(df) * 0.5:
        continue

    conv_vals = converters[col].dropna()
    stable_vals = stable[col].dropna()

    if len(conv_vals) < 10 or len(stable_vals) < 10:
        continue

    # T-test
    t_stat, p_val = stats.ttest_ind(conv_vals, stable_vals, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((conv_vals.std() ** 2 + stable_vals.std() ** 2) / 2)
    cohens_d = (
        (conv_vals.mean() - stable_vals.mean()) / pooled_std if pooled_std > 0 else 0
    )

    results.append(
        {
            "feature": col,
            "conv_mean": conv_vals.mean(),
            "stable_mean": stable_vals.mean(),
            "p_value": p_val,
            "cohens_d": cohens_d,
        }
    )

results_df = pd.DataFrame(results).sort_values("p_value")

print("\nTop 10 most discriminative slope features (by p-value):")
print(results_df.head(10).to_string(index=False))

# Check if ANY features are significant after Bonferroni correction
bonferroni_alpha = 0.05 / len(results)
n_significant = (results_df["p_value"] < bonferroni_alpha).sum()
print(
    f"\nFeatures significant after Bonferroni correction (α={bonferroni_alpha:.2e}): {n_significant}"
)

# Check correlations between top features
print(f"\n{'='*80}")
print("MULTICOLLINEARITY CHECK")
print(f"{'='*80}")

top_10_features = results_df.head(10)["feature"].tolist()
corr_matrix = df[top_10_features].corr()

# Find highly correlated pairs (|r| > 0.8)
high_corr = []
for i in range(len(top_10_features)):
    for j in range(i + 1, len(top_10_features)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.8:
            high_corr.append((top_10_features[i], top_10_features[j], r))

if high_corr:
    print(f"Found {len(high_corr)} highly correlated pairs (|r| > 0.8):")
    for f1, f2, r in high_corr[:5]:
        print(f"  {f1} <-> {f2}: r={r:.3f}")
else:
    print("No severe multicollinearity detected among top 10 features")

# Recommend next steps
print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

if n_significant == 0:
    print("⚠️  NO features are significant after correction!")
    print("   → Consider:")
    print("     1. Use more scans per subject (currently ≥4)")
    print("     2. Different slope computation (quadratic/spline instead of linear)")
    print("     3. Alternative features (acceleration, curvature)")
    print("     4. Combine with clinical variables (age, APOE, MMSE)")
elif n_significant < 5:
    print(f"⚠️  Only {n_significant} significant features")
    print("   → Model may be underpowered")
    print("   → Consider feature engineering or multimodal data")
else:
    print(f"✓ {n_significant} significant features - model should work")

# Check baseline C-index issue
print(f"\n{'='*80}")
print("BASELINE MODEL INVESTIGATION")
print(f"{'='*80}")

baseline_cols = [c for c in df.columns if c.endswith("_bl")]
if baseline_cols:
    print(f"Baseline features available: {len(baseline_cols)}")

    # Test baseline features
    baseline_results = []
    for col in baseline_cols[:20]:  # Test first 20
        if df[col].notna().sum() < len(df) * 0.5:
            continue

        conv_vals = converters[col].dropna()
        stable_vals = stable[col].dropna()

        if len(conv_vals) < 10 or len(stable_vals) < 10:
            continue

        t_stat, p_val = stats.ttest_ind(conv_vals, stable_vals, equal_var=False)

        baseline_results.append(
            {
                "feature": col,
                "p_value": p_val,
                "conv_mean": conv_vals.mean(),
                "stable_mean": stable_vals.mean(),
            }
        )

    baseline_df = pd.DataFrame(baseline_results).sort_values("p_value")
    print("\nTop 5 baseline features:")
    print(baseline_df.head(5).to_string(index=False))

    n_sig_baseline = (baseline_df["p_value"] < 0.05).sum()
    print(f"\nBaseline features with p < 0.05: {n_sig_baseline}")

    if n_sig_baseline == 0:
        print(
            "⚠️  Baseline features are NOT predictive - explains low baseline C-index!"
        )

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. If slopes ARE predictive: Retrain with better hyperparameters/regularization")
print(
    "2. If slopes NOT predictive: Need to re-examine BSC computation or use different features"
)
print("3. Check if you want to update paper with HONEST results or improve model first")
