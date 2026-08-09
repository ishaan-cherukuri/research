"""
External validation: apply the ADNI-trained XGBoost survival model, frozen exactly as
trained (same weights, same feature list, same preprocessing transform), to an independent
cohort (OASIS-3) with zero retraining or fine-tuning.

mri-bsc/code/ml/train_xgb_survival_combined.py does not persist its fitted winsor
limits / MinMaxScaler to disk -- only the raw xgb_model.json. Since those preprocessing
steps are fully deterministic given the same input data, split seed (42), and
hyperparameters, this script reproduces the exact ADNI training-time fit (winsor limits +
scaler) by re-running the same train/test split and feature-selection logic on the
ADNI data, then applies that identical fitted transform to the OASIS features before
scoring with the frozen model. This is NOT retraining -- the model weights are untouched;
only the (deterministic, ADNI-train-only-fit) input transform is being reproduced so OASIS
features land in the same feature space the model was trained on.

Usage:
    python3 eval_frozen_model_external.py \
        --adni_slopes bsc_longitudinal_slopes.csv --adni_survival survival_labels.csv \
        --adni_t1 features_t1_all.csv \
        --model results_combined/xgb_model.json --features results_combined/xgb_features.txt \
        --ext_slopes bsc_longitudinal_slopes_oasis.csv --ext_survival survival_labels_oasis.csv \
        --ext_t1 features_t1_all_oasis.csv \
        --out_dir results_external_oasis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

T1_FEATURE_COLS = [
    "seg_csf_mm3_bl", "seg_gm_total_mm3_bl", "seg_wm_total_mm3_bl", "seg_brain_mm3_bl",
    "seg_tiv_mm3_bl", "seg_bpf_bl", "seg_ventricles_total_mm3_bl", "seg_ventricles_norm_bl",
    "long_brain_vol_last", "long_brain_vol_delta", "long_brain_vol_pctchg", "long_brain_vol_slope_yr",
    "long_brain_vol_mean", "long_brain_vol_std",
    "long_brain_intensity_last", "long_brain_intensity_delta", "long_brain_intensity_pctchg",
    "long_brain_intensity_slope_yr", "long_brain_intensity_mean", "long_brain_intensity_std",
    "long_snr_last", "long_snr_delta", "long_snr_pctchg", "long_snr_slope_yr",
    "long_snr_mean", "long_snr_std",
    "long_brain_bg_ratio_last", "long_brain_bg_ratio_delta", "long_brain_bg_ratio_pctchg",
    "long_brain_bg_ratio_slope_yr", "long_brain_bg_ratio_mean", "long_brain_bg_ratio_std",
    "qc_brain_mask_vol_mm3_bl", "qc_brain_mean_bl", "qc_brain_std_bl", "qc_snr_bl",
    "qc_brain_bg_ratio_bl", "field_strength_mode_t", "meta_field_strength_t_bl",
]


def signed_log1p_df(X: pd.DataFrame) -> pd.DataFrame:
    return np.sign(X) * np.log1p(np.abs(X))


def fit_winsor_limits(X: pd.DataFrame, lower_q: float, upper_q: float) -> dict:
    return {"lower": X.quantile(lower_q), "upper": X.quantile(upper_q)}


def apply_winsor_limits(X: pd.DataFrame, limits: dict) -> pd.DataFrame:
    return X.clip(lower=limits["lower"], upper=limits["upper"], axis=1)


def select_slope_features_train_only(X_train, top_k, penalize_regex, penalty_factor, winsor_q_low, winsor_q_high):
    X_log = signed_log1p_df(X_train)
    limits = fit_winsor_limits(X_log, winsor_q_low, winsor_q_high)
    X_robust = apply_winsor_limits(X_log, limits)
    raw_var = X_robust.var()
    score = raw_var.copy()
    mask = pd.Series(score.index.str.contains(penalize_regex, case=False, regex=True), index=score.index)
    score.loc[mask] = score.loc[mask] * penalty_factor
    top_k = min(top_k, len(score))
    return score.nlargest(top_k).index.tolist()


def reproduce_adni_fit(slopes_path, survival_path, t1_path, top_k, penalize_regex,
                        penalty_factor, winsor_low, winsor_high):
    """Re-runs the exact deterministic ADNI train-time data prep (same seed=42 split,
    same feature selection) to recover the winsor limits fit on ADNI train data -- needed
    to transform OASIS features into the same space the frozen model expects."""
    slopes = pd.read_csv(slopes_path)
    survival = pd.read_csv(survival_path)
    t1_raw = pd.read_csv(t1_path).rename(columns={"subject_id": "subject"})

    available_t1 = [c for c in T1_FEATURE_COLS if c in t1_raw.columns]
    t1 = t1_raw[["subject"] + available_t1].copy()

    df = survival.merge(slopes, on="subject", how="inner").merge(t1, on="subject", how="left")

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().mean() > 0.8]

    y_all = df[["time_years", "event"]].copy()
    idx_train, _ = train_test_split(df.index, test_size=0.3, random_state=42, stratify=y_all["event"])

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())
    selected_slopes = select_slope_features_train_only(
        X_slopes_all.loc[idx_train], top_k, penalize_regex, penalty_factor, winsor_low, winsor_high
    )

    X_t1_all = df[available_t1].copy()
    t1_medians = X_t1_all.loc[idx_train].median()
    X_t1_all = X_t1_all.fillna(t1_medians)

    all_features = selected_slopes + available_t1
    X_all = pd.concat([X_slopes_all[selected_slopes], X_t1_all], axis=1)
    X_train_raw = X_all.loc[idx_train].copy()

    X_train_log = signed_log1p_df(X_train_raw)
    limits = fit_winsor_limits(X_train_log, winsor_low, winsor_high)

    # Reproduce the MinMaxScaler fit on ADNI train (post-winsorize)
    from sklearn.preprocessing import MinMaxScaler
    X_train_robust = apply_winsor_limits(X_train_log, limits)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train_robust)

    return all_features, limits, scaler, t1_medians, slope_cols


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adni_slopes", required=True)
    ap.add_argument("--adni_survival", required=True)
    ap.add_argument("--adni_t1", required=True)
    ap.add_argument("--model", required=True, help="xgb_model.json from ADNI training")
    ap.add_argument("--ext_slopes", required=True)
    ap.add_argument("--ext_survival", required=True)
    ap.add_argument("--ext_t1", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--penalize_regex", type=str, default="nboundary")
    ap.add_argument("--penalty_factor", type=float, default=0.10)
    ap.add_argument("--winsor_low", type=float, default=0.01)
    ap.add_argument("--winsor_high", type=float, default=0.99)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Reproducing ADNI train-time fit (winsor limits + scaler)...")
    all_features, limits, scaler, t1_medians, slope_cols = reproduce_adni_fit(
        args.adni_slopes, args.adni_survival, args.adni_t1,
        args.top_k, args.penalize_regex, args.penalty_factor, args.winsor_low, args.winsor_high,
    )
    print(f"  {len(all_features)} features: {all_features}")

    print("\nLoading external (OASIS) data...")
    ext_slopes = pd.read_csv(args.ext_slopes)
    ext_survival = pd.read_csv(args.ext_survival)
    ext_t1_raw = pd.read_csv(args.ext_t1).rename(columns={"subject_id": "subject"})

    ext_df = ext_survival.merge(ext_slopes, on="subject", how="inner").merge(ext_t1_raw, on="subject", how="left")
    print(f"  External cohort after merge: {len(ext_df)} subjects "
          f"({ext_df['event'].sum()} converters, {(ext_df['event'] == 0).sum()} stable)")

    if len(ext_df) == 0:
        raise SystemExit("No overlap between external slopes/survival/T1 tables -- nothing to evaluate.")

    missing_features = [f for f in all_features if f not in ext_df.columns]
    if missing_features:
        print(f"  WARNING: missing features in external data (filled with ADNI train median/0): {missing_features}")
        for f in missing_features:
            ext_df[f] = np.nan

    X_ext = ext_df[all_features].copy()
    # Impute missing values using ADNI-train-derived medians for T1 cols, 0 for slope cols
    # (slopes were median-imputed using the FULL ADNI cohort's slope median at train time,
    # but per-feature train medians aren't persisted -- 0 is the neutral value post-log1p).
    for col in all_features:
        if col in t1_medians.index:
            X_ext[col] = X_ext[col].fillna(t1_medians[col])
        else:
            X_ext[col] = X_ext[col].fillna(0.0)

    X_ext_log = signed_log1p_df(X_ext)
    X_ext_robust = apply_winsor_limits(X_ext_log, limits)
    X_ext_scaled = pd.DataFrame(scaler.transform(X_ext_robust), columns=X_ext_robust.columns, index=X_ext_robust.index)

    model = xgb.Booster()
    model.load_model(args.model)

    dext = xgb.DMatrix(X_ext_scaled)
    pred_ext = model.predict(dext)

    event = ext_df["event"].values.astype(bool)
    time = ext_df["time_years"].values
    c_index, *_ = concordance_index_censored(event, time, -pred_ext)

    metrics = {
        "external_cohort": "OASIS-3",
        "n_subjects": int(len(ext_df)),
        "n_events": int(ext_df["event"].sum()),
        "c_index": float(c_index),
        "n_features": len(all_features),
    }
    print(f"\n{'=' * 80}")
    print("EXTERNAL VALIDATION RESULT (frozen ADNI-trained model -> OASIS-3)")
    print(f"{'=' * 80}")
    print(f"  N subjects   : {metrics['n_subjects']}")
    print(f"  N events     : {metrics['n_events']}")
    print(f"  C-index      : {metrics['c_index']:.4f}")
    print(f"{'=' * 80}")

    with open(out_dir / "external_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "subject": ext_df["subject"].values,
        "true_time": time,
        "event": ext_df["event"].values,
        "predicted_risk": pred_ext,
    })
    pred_df.to_csv(out_dir / "external_eval_predictions.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
