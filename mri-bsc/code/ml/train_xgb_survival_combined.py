
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sksurv.metrics import concordance_index_censored

T1_FEATURE_COLS = [
    "seg_csf_mm3_bl",
    "seg_gm_total_mm3_bl",
    "seg_wm_total_mm3_bl",
    "seg_brain_mm3_bl",
    "seg_tiv_mm3_bl",
    "seg_bpf_bl",
    "seg_ventricles_total_mm3_bl",
    "seg_ventricles_norm_bl",
    "long_brain_vol_last",
    "long_brain_vol_delta",
    "long_brain_vol_pctchg",
    "long_brain_vol_slope_yr",
    "long_brain_vol_mean",
    "long_brain_vol_std",
    "long_brain_intensity_last",
    "long_brain_intensity_delta",
    "long_brain_intensity_pctchg",
    "long_brain_intensity_slope_yr",
    "long_brain_intensity_mean",
    "long_brain_intensity_std",
    "long_snr_last",
    "long_snr_delta",
    "long_snr_pctchg",
    "long_snr_slope_yr",
    "long_snr_mean",
    "long_snr_std",
    "long_brain_bg_ratio_last",
    "long_brain_bg_ratio_delta",
    "long_brain_bg_ratio_pctchg",
    "long_brain_bg_ratio_slope_yr",
    "long_brain_bg_ratio_mean",
    "long_brain_bg_ratio_std",
    "qc_brain_mask_vol_mm3_bl",
    "qc_brain_mean_bl",
    "qc_brain_std_bl",
    "qc_snr_bl",
    "qc_brain_bg_ratio_bl",
    "field_strength_mode_t",
    "meta_field_strength_t_bl",
]

def load_and_merge(slopes_path: str, survival_path: str, t1_path: str) -> pd.DataFrame:
    slopes = pd.read_csv(slopes_path)
    survival = pd.read_csv(survival_path)
    t1_raw = pd.read_csv(t1_path)

    print(f"  BSC slopes:  {len(slopes)} subjects")
    print(f"  Survival:    {len(survival)} subjects")
    print(f"  T1 features: {len(t1_raw)} subjects")

    t1_raw = t1_raw.rename(columns={"subject_id": "subject"})

    available_t1 = [c for c in T1_FEATURE_COLS if c in t1_raw.columns]
    missing_t1 = [c for c in T1_FEATURE_COLS if c not in t1_raw.columns]
    if missing_t1:
        print(f"  WARNING: T1 columns not found (skipped): {missing_t1}")
    t1 = t1_raw[["subject"] + available_t1].copy()

    merged = survival.merge(slopes, on="subject", how="inner")
    merged = merged.merge(t1, on="subject", how="left")

    print(f"  After merge: {len(merged)} subjects  "
          f"({merged['event'].sum()} converters, "
          f"{(merged['event'] == 0).sum()} stable)")
    return merged, available_t1

def signed_log1p_df(X: pd.DataFrame) -> pd.DataFrame:
    return np.sign(X) * np.log1p(np.abs(X))

def fit_winsor_limits(X: pd.DataFrame, lower_q: float, upper_q: float) -> dict:
    return {"lower": X.quantile(lower_q), "upper": X.quantile(upper_q)}

def apply_winsor_limits(X: pd.DataFrame, limits: dict) -> pd.DataFrame:
    return X.clip(lower=limits["lower"], upper=limits["upper"], axis=1)

def minmax_scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xtr = pd.DataFrame(scaler.fit_transform(X_train),
                       columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test),
                       columns=X_test.columns, index=X_test.index)
    return Xtr, Xte, scaler

def select_slope_features_train_only(
    X_train: pd.DataFrame,
    top_k: int = 20,
    penalize_regex: str = "nboundary",
    penalty_factor: float = 0.10,
    winsor_q_low: float = 0.01,
    winsor_q_high: float = 0.99,
    quiet: bool = False,
) -> list:
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

    if not quiet:
        print(f"\n{'='*80}")
        print("BSC SLOPE FEATURE SELECTION (TRAIN ONLY)")
        print(f"  Robust variance: signed_log1p + winsorize [{winsor_q_low}, {winsor_q_high}]")
        print(f"  Penalise regex: '{penalize_regex}' × {penalty_factor}")
        print(f"  Selecting top {top_k} BSC slope features")
        print(f"{'='*80}")
        for i, feat in enumerate(top_features, 1):
            pen = " (PENALISED)" if mask.loc[feat] else ""
            print(f"  {i:2d}. {feat:45s} var={raw_var[feat]:.6f}  score={score[feat]:.6f}{pen}")
        print(f"{'='*80}")

    return top_features

def make_aft_labels(y: pd.DataFrame):
    y_lower = y["time_years"].values.astype(float)
    y_upper = np.where(y["event"].values == 1, y["time_years"].values, np.inf).astype(float)
    return y_lower, y_upper

def train_xgb_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    params: dict,
    n_estimators: int,
) -> tuple:
    y_lower_tr, y_upper_tr = make_aft_labels(y_train)
    y_lower_te, y_upper_te = make_aft_labels(y_test)

    dtrain = xgb.DMatrix(X_train)
    dtrain.set_float_info("label_lower_bound", y_lower_tr)
    dtrain.set_float_info("label_upper_bound", y_upper_tr)

    dtest = xgb.DMatrix(X_test)
    dtest.set_float_info("label_lower_bound", y_lower_te)
    dtest.set_float_info("label_upper_bound", y_upper_te)

    print(f"\nFitting XGBoost AFT  (n_estimators={n_estimators}, "
          f"max_depth={params.get('max_depth')}, lr={params.get('learning_rate')}) ...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=50,
    )

    pred_train = model.predict(dtrain)
    pred_test = model.predict(dtest)

    def c_index(y_df, pred):
        event = y_df["event"].values.astype(bool)
        time = y_df["time_years"].values
        ci, _, _, _, _ = concordance_index_censored(event, time, -pred)
        return float(ci)

    train_c = c_index(y_train, pred_train)
    test_c = c_index(y_test, pred_test)

    metrics = {
        "train_c_index": train_c,
        "test_c_index": test_c,
        "overfitting_gap": round(train_c - test_c, 4),
        "n_features": X_train.shape[1],
        "n_estimators": n_estimators,
    }

    print(f"\n{'='*80}")
    print("RESULTS: XGBoost AFT Survival (BSC slopes + T1 morphometry)")
    print(f"{'='*80}")
    print(f"  Train C-index : {train_c:.4f}")
    print(f"  Test  C-index : {test_c:.4f}")
    print(f"  Overfitting   : {train_c - test_c:.4f}")
    print(f"{'='*80}")

    return model, metrics, pred_train, pred_test

def run_5fold_cv(df, slope_cols, t1_cols, y_all, params, n_estimators, top_k,
                 penalize_regex, penalty_factor, winsor_low, winsor_high):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_scores, test_scores = [], []

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())
    X_t1_all = df[t1_cols].copy()

    print(f"\n{'='*80}")
    print("5-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")

    for fold_i, (tr_idx, te_idx) in enumerate(
            skf.split(df.index, y_all["event"]), 1):
        tr_idx = df.index[tr_idx]
        te_idx = df.index[te_idx]
        y_tr = y_all.loc[tr_idx]
        y_te = y_all.loc[te_idx]

        sel = select_slope_features_train_only(
            X_slopes_all.loc[tr_idx], top_k, penalize_regex, penalty_factor,
            winsor_low, winsor_high, quiet=True,
        )

        t1_med = X_t1_all.loc[tr_idx].median()
        X_t1_fold = X_t1_all.fillna(t1_med)

        X_tr = pd.concat([X_slopes_all.loc[tr_idx, sel], X_t1_fold.loc[tr_idx]], axis=1)
        X_te = pd.concat([X_slopes_all.loc[te_idx, sel], X_t1_fold.loc[te_idx]], axis=1)

        X_tr_log = signed_log1p_df(X_tr)
        X_te_log = signed_log1p_df(X_te)
        lims = fit_winsor_limits(X_tr_log, winsor_low, winsor_high)
        X_tr_r = apply_winsor_limits(X_tr_log, lims)
        X_te_r = apply_winsor_limits(X_te_log, lims)
        X_tr_s, X_te_s, _ = minmax_scale_train_test(X_tr_r, X_te_r)

        y_lower_tr = y_tr["time_years"].values.astype(float)
        y_upper_tr = np.where(y_tr["event"].values == 1,
                              y_tr["time_years"].values, np.inf).astype(float)
        y_lower_te = y_te["time_years"].values.astype(float)
        y_upper_te = np.where(y_te["event"].values == 1,
                              y_te["time_years"].values, np.inf).astype(float)

        dtr = xgb.DMatrix(X_tr_s)
        dtr.set_float_info("label_lower_bound", y_lower_tr)
        dtr.set_float_info("label_upper_bound", y_upper_tr)
        dte = xgb.DMatrix(X_te_s)
        dte.set_float_info("label_lower_bound", y_lower_te)
        dte.set_float_info("label_upper_bound", y_upper_te)

        m = xgb.train(params, dtr, num_boost_round=n_estimators,
                      verbose_eval=False)

        def ci(y_df, pred):
            ev = y_df["event"].values.astype(bool)
            t = y_df["time_years"].values
            c, *_ = concordance_index_censored(ev, t, -pred)
            return float(c)

        c_tr = ci(y_tr, m.predict(dtr))
        c_te = ci(y_te, m.predict(dte))
        train_scores.append(c_tr)
        test_scores.append(c_te)
        print(f"  Fold {fold_i}  train={c_tr:.4f}  test={c_te:.4f}  "
              f"gap={c_tr - c_te:.4f}")

    gaps = [tr - te for tr, te in zip(train_scores, test_scores)]
    print(f"\n  Mean  train={np.mean(train_scores):.4f}  "
          f"test={np.mean(test_scores):.4f}  gap={np.mean(gaps):.4f}")
    print(f"  Std   train={np.std(train_scores):.4f}  "
          f"test={np.std(test_scores):.4f}  gap={np.std(gaps):.4f}")
    print(f"{'='*80}")

    return {"train": train_scores, "test": test_scores, "gap": gaps}

def plot_cv_boxplots(cv_results: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(
        "XGBoost AFT  —  BSC Slopes + T1 Morphometry\n"
        "5-Fold Cross-Validation C-Index",
        fontsize=14, fontweight="bold",
    )

    panels = [
        ("train", "Training C-Index", "#2ecc71"),
        ("test",  "Test C-Index",     "#3498db"),
        ("gap",   "Overfitting Gap\n(Train − Test)", "#e74c3c"),
    ]

    for ax, (key, title, color) in zip(axes, panels):
        data = cv_results[key]
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="black", linewidth=2.5))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.65)

        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(data))
        ax.scatter([1 + j for j in jitter], data, color=color,
                   edgecolors="black", s=60, zorder=5, linewidths=0.8)

        mean_val = np.mean(data)
        ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_ylabel("C-Index" if key != "gap" else "Gap", fontsize=11)
        ax.text(1.0, mean_val, f" μ={mean_val:.3f}", va="center",
                fontsize=10, color="black")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved boxplot → {out_path}")

def create_km_curves(y_train, y_test, pred_train, pred_test, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for y_data, risk, title, ax in [
        (y_train, pred_train, "Training Set", axes[0]),
        (y_test,  pred_test,  "Test Set",     axes[1]),
    ]:
        p33, p67 = np.percentile(risk, [33.33, 66.67])
        high = risk >= p67
        low  = risk <= p33
        mid  = ~(high | low)

        time  = y_data["time_years"].values
        event = y_data["event"].values

        h_n, h_e = high.sum(), event[high].sum()
        m_n, m_e = mid.sum(),  event[mid].sum()
        l_n, l_e = low.sum(),  event[low].sum()
        print(f"\n{title}:  high={h_n}({h_e} ev)  mid={m_n}({m_e} ev)  low={l_n}({l_e} ev)")

        for mask, label, color in [
            (high, f"High Risk (n={h_n}, events={h_e})",   "red"),
            (mid,  f"Medium Risk (n={m_n}, events={m_e})", "orange"),
            (low,  f"Low Risk (n={l_n}, events={l_e})",    "green"),
        ]:
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask], label=label)
            kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, ci_show=True)

        lr = logrank_test(time[high], time[low], event[high], event[low])
        print(f"  Log-rank p={lr.p_value:.4f}")

        ax.set_xlabel("Time (years)", fontsize=12, fontweight="bold")
        ax.set_ylabel("P(Remain MCI)", fontsize=12, fontweight="bold")
        ax.set_title(f"{title}\nKaplan-Meier by Predicted Risk",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.text(0.98, 0.02, f"Log-rank p={lr.p_value:.4f}",
                transform=ax.transAxes, fontsize=10,
                ha="right", va="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved KM curves → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slopes",    required=True, help="bsc_longitudinal_slopes.csv")
    parser.add_argument("--survival",  required=True, help="time_to_conversion.csv")
    parser.add_argument("--t1_features", required=True, help="raw_t1_analysis/features_all_456.csv")
    parser.add_argument("--out_dir",   required=True, help="Output directory")
    parser.add_argument("--top_k",     type=int,   default=20)
    parser.add_argument("--penalize_regex", type=str, default="nboundary")
    parser.add_argument("--penalty_factor", type=float, default=0.10)
    parser.add_argument("--winsor_low",  type=float, default=0.01)
    parser.add_argument("--winsor_high", type=float, default=0.99)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth",   type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--figs_dir", type=str, default="figs",
                        help="Directory to save figure PNGs (default: figs/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    df, t1_cols_used = load_and_merge(args.slopes, args.survival, args.t1_features)

    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    slope_cols = [c for c in slope_cols if df[c].notna().mean() > 0.8]

    y_all = df[["time_years", "event"]].copy()

    print(f"\n{'='*80}")
    print("DATA SPLIT  (70 / 30, stratified by event)")
    print(f"{'='*80}")
    print(f"  Total: {len(df)}  |  events: {y_all['event'].sum()}  "
          f"({y_all['event'].mean()*100:.1f}%)")

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.3, random_state=42, stratify=y_all["event"]
    )
    y_train = y_all.loc[idx_train]
    y_test  = y_all.loc[idx_test]
    print(f"  Train: {len(idx_train)}  |  events: {y_train['event'].sum()}")
    print(f"  Test:  {len(idx_test)}  |  events: {y_test['event'].sum()}")

    X_slopes_all = df[slope_cols].fillna(df[slope_cols].median())
    selected_slopes = select_slope_features_train_only(
        X_slopes_all.loc[idx_train],
        top_k=args.top_k,
        penalize_regex=args.penalize_regex,
        penalty_factor=args.penalty_factor,
        winsor_q_low=args.winsor_low,
        winsor_q_high=args.winsor_high,
    )

    X_t1_all = df[t1_cols_used].copy()
    t1_medians = X_t1_all.loc[idx_train].median()
    X_t1_all = X_t1_all.fillna(t1_medians)

    all_features = selected_slopes + t1_cols_used
    X_all = pd.concat([X_slopes_all[selected_slopes], X_t1_all], axis=1)

    print(f"\n{'='*80}")
    print(f"COMBINED FEATURES: {len(selected_slopes)} BSC slopes  +  "
          f"{len(t1_cols_used)} T1 features  =  {len(all_features)} total")
    print(f"{'='*80}")

    X_train_raw = X_all.loc[idx_train].copy()
    X_test_raw  = X_all.loc[idx_test].copy()

    X_train_log = signed_log1p_df(X_train_raw)
    X_test_log  = signed_log1p_df(X_test_raw)

    limits = fit_winsor_limits(X_train_log, args.winsor_low, args.winsor_high)
    X_train_robust = apply_winsor_limits(X_train_log, limits)
    X_test_robust  = apply_winsor_limits(X_test_log,  limits)

    X_train_scaled, X_test_scaled, _ = minmax_scale_train_test(
        X_train_robust, X_test_robust
    )

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    model, metrics, pred_train, pred_test = train_xgb_model(
        X_train_scaled, X_test_scaled, y_train, y_test, params, args.n_estimators
    )

    model.save_model(str(out_dir / "xgb_model.json"))

    with open(out_dir / "xgb_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "xgb_features.txt", "w") as f:
        for feat in all_features:
            f.write(f"{feat}\n")

    test_df = pd.DataFrame({
        "subject":        df.loc[idx_test, "subject"].values,
        "true_time":      y_test["time_years"].values,
        "event":          y_test["event"].values,
        "predicted_risk": pred_test,
    })
    test_df.to_csv(out_dir / "xgb_predictions.csv", index=False)

    with open(out_dir / "xgb_summary.txt", "w") as f:
        f.write("XGBoost AFT Survival — BSC Slopes + T1 Morphometry\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Subjects: {len(df)}  |  converters: {y_all['event'].sum()}\n")
        f.write(f"BSC slope features selected: {len(selected_slopes)}\n")
        f.write(f"T1 features included:        {len(t1_cols_used)}\n")
        f.write(f"Total features:              {len(all_features)}\n\n")
        f.write("Preprocessing:\n")
        f.write("  signed_log1p → winsorize [0.01, 0.99] → MinMax (0,1) — fit on train\n\n")
        f.write("XGBoost params:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  n_estimators: {args.n_estimators}\n\n")
        f.write("METRICS:\n")
        f.write(f"  Train C-index : {metrics['train_c_index']:.4f}\n")
        f.write(f"  Test  C-index : {metrics['test_c_index']:.4f}\n")
        f.write(f"  Overfitting   : {metrics['overfitting_gap']:.4f}\n")

    print(f"\nResults saved to: {out_dir}")

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"  Saving to: {figs_dir}")
    print(f"{'='*80}")

    cv_results = run_5fold_cv(
        df, slope_cols, t1_cols_used, y_all, params, args.n_estimators,
        args.top_k, args.penalize_regex, args.penalty_factor,
        args.winsor_low, args.winsor_high,
    )
    plot_cv_boxplots(cv_results, figs_dir / "model_comparison_boxplot.png")

    with open(out_dir / "xgb_cv_results.json", "w") as f:
        json.dump({k: [round(v, 4) for v in vs] for k, vs in cv_results.items()}, f, indent=2)

    create_km_curves(y_train, y_test, pred_train, pred_test,
                     figs_dir / "kaplan_meier_curves.png")

    print(f"\n{'='*80}")
    print(f"  TRAIN C-index : {metrics['train_c_index']:.4f}")
    print(f"  TEST  C-index : {metrics['test_c_index']:.4f}")
    print(f"  5-fold CV mean test C-index: {np.mean(cv_results['test']):.4f} "
          f"± {np.std(cv_results['test']):.4f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
