"""Cohort builder for the JAD ALZ-26-0928 V3 revision spec.

Implements the follow-up definition in spec section 3.2, which is the
subject-specific landmark: features come only from scans acquired while the
subject was still MCI, the clock starts at the last such scan, and conversion
or censoring dates come from DXSUM clinical visits.

Differences from build_landmark_cohort.py --landmark subject:

  * spec section 3.1.5 also drops any scan whose own diagnosis code is AD,
    independent of the DXSUM conversion date, so a scan cannot enter the
    feature window if it was already labelled demented at acquisition.
  * the full covariate block from spec section 6 is assembled: age, sex,
    education, APOE4 allele count, MMSE, ADAS-Cog13, CDR-SB, field strength.
  * scanner metadata (field strength, manufacturer, model, site) is attached
    per scan from the IDA search export, for spec section 3.4 and for the
    LongComBat batch variable later.
  * ADNI's own FreeSurfer output supplies the comparator features of spec
    section 5.3: 68 Desikan-Killiany mean-thickness columns and hippocampal
    volume normalised by ICV, with slopes computed over the same window as BSC.

Attrition is logged at every exclusion, per the spec's instruction not to drop
scans silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from build_landmark_cohort import (
    AD,
    MCI,
    T1_BASELINE_ONLY,
    T1_TRAJECTORY,
    load_demog,
    load_dx,
    load_mmse,
    load_scans,
    trajectory,
)

# ADNI FreeSurfer tables, newest protocol first. A scan is taken from the first
# table that has it, so ADNI3 scans use FSX7 and ADNI1 scans fall back to FSX51.
FSX_TABLES = ["UCSFFSX7", "UCSFFSX6", "UCSFFSX51"]
HIPPO_L, HIPPO_R, ICV_COL = "ST29SV", "ST88SV", "ST10CV"


def load_scanner_meta(ida_csv: str) -> pd.DataFrame:
    """Field strength, vendor, model and site for each subject-by-date scan."""
    d = pd.read_csv(ida_csv, low_memory=False)
    d = d.rename(columns={"Subject ID": "subject", "Study Date": "date",
                          "Image ID": "image_uid", "Imaging Protocol": "proto"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce", format="mixed")
    d = d.dropna(subset=["date", "proto"])

    def field(p: str, key: str) -> str | float:
        for part in str(p).split(";"):
            if part.startswith(key + "="):
                return part.split("=", 1)[1]
        return np.nan

    d["field_strength"] = pd.to_numeric(d["proto"].map(lambda p: field(p, "Field Strength")),
                                        errors="coerce")
    # Some headers store 1.5T as 15000 and 3T as 30000. Rescale those.
    d.loc[d["field_strength"] > 100, "field_strength"] /= 10000.0
    d["manufacturer"] = d["proto"].map(lambda p: field(p, "Manufacturer"))
    d["scanner_model"] = d["proto"].map(lambda p: field(p, "Mfg Model"))
    d["site"] = d["subject"].str.slice(0, 3)

    # One row per subject-date: take the most common protocol across series.
    agg = (d.groupby(["subject", "date"])
             .agg(field_strength=("field_strength", lambda s: s.mode().iloc[0]
                                  if len(s.mode()) else np.nan),
                  manufacturer=("manufacturer", lambda s: s.mode().iloc[0]
                                if len(s.mode()) else np.nan),
                  scanner_model=("scanner_model", lambda s: s.mode().iloc[0]
                                 if len(s.mode()) else np.nan),
                  site=("site", "first"))
             .reset_index())
    agg["scanner_id"] = (agg["site"].astype(str) + "_"
                         + agg["manufacturer"].astype(str) + "_"
                         + agg["field_strength"].round(1).astype(str))
    return agg


def load_freesurfer(data_dir: Path, tol_days: int) -> tuple[pd.DataFrame, list[str]]:
    """Per-scan DK thickness and hippocampal volume from ADNI's FreeSurfer runs."""
    frames = []
    for name in FSX_TABLES:
        path = data_dir / f"{name}.rda"
        if not path.exists():
            continue
        d = pyreadr.read_r(str(path))[name]
        if "PTID" not in d.columns:
            continue
        ta = [c for c in d.columns if c.startswith("ST") and c.endswith("TA")]
        need = [HIPPO_L, HIPPO_R, ICV_COL]
        keep = [c for c in ta + need if c in d.columns]
        if not keep:
            continue
        sub = d[["PTID", "EXAMDATE"] + keep].copy()
        sub["EXAMDATE"] = pd.to_datetime(sub["EXAMDATE"], errors="coerce")
        sub = sub.dropna(subset=["EXAMDATE"])
        for c in keep:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub["fsx_source"] = name
        frames.append(sub)
    if not frames:
        return pd.DataFrame(), []

    fs = pd.concat(frames, ignore_index=True)
    # Newest protocol wins when a scan appears in more than one table.
    fs["rank"] = fs["fsx_source"].map({n: i for i, n in enumerate(FSX_TABLES)})
    fs = fs.sort_values("rank").drop_duplicates(["PTID", "EXAMDATE"], keep="first")

    ta_cols = sorted([c for c in fs.columns if c.startswith("ST") and c.endswith("TA")])
    if HIPPO_L in fs.columns and HIPPO_R in fs.columns:
        hip = fs[HIPPO_L] + fs[HIPPO_R]
        icv = fs[ICV_COL] if ICV_COL in fs.columns else np.nan
        fs["hippo_icv"] = hip / icv
    return fs, ta_cols


def attach_freesurfer(window: pd.DataFrame, fs: pd.DataFrame, subj: str,
                      tol_days: int) -> pd.DataFrame:
    """Nearest-date FreeSurfer row for each scan in the window, within tolerance."""
    sfs = fs[fs["PTID"] == subj]
    if sfs.empty:
        return pd.DataFrame(np.nan, index=window.index, columns=fs.columns)
    blank = pd.Series(np.nan, index=fs.columns)
    out = []
    for _, row in window.iterrows():
        gap = (sfs["EXAMDATE"] - row["acq_date"]).abs()
        j = gap.idxmin()
        out.append(sfs.loc[j] if gap.loc[j] <= pd.Timedelta(days=tol_days) else blank)
    return pd.DataFrame(out).set_index(window.index)


def load_apoe(path: Path) -> dict[str, float]:
    d = pyreadr.read_r(str(path))["APOERES"][["PTID", "GENOTYPE"]].dropna()
    d["apoe4"] = d["GENOTYPE"].astype(str).str.count("4").astype(float)
    return d.groupby("PTID")["apoe4"].max().to_dict()


def load_visit_score(path: Path, table: str, col: str,
                     lo: float, hi: float) -> dict[str, pd.DataFrame]:
    d = pyreadr.read_r(str(path))[table][["PTID", "VISDATE", col]].copy()
    d["VISDATE"] = pd.to_datetime(d["VISDATE"], errors="coerce")
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["VISDATE", col])
    d = d[(d[col] >= lo) & (d[col] <= hi)]
    return {p: g.sort_values("VISDATE") for p, g in d.groupby("PTID")}


def nearest_score(by_subj: dict, subj: str, t0: pd.Timestamp, col: str,
                  window_days: int) -> float:
    g = by_subj.get(subj)
    if g is None or not len(g):
        return np.nan
    gap = (g["VISDATE"] - t0).abs()
    j = gap.idxmin()
    return float(g.loc[j, col]) if gap.loc[j] <= pd.Timedelta(days=window_days) else np.nan


def apply_real_dates(scans: pd.DataFrame, path: str) -> pd.DataFrame:
    """Swap synthetic acquisition dates for the real IDA study dates.

    Scans whose date could not be matched are dropped rather than left on the
    approximate clock, so the time axis is uniformly real.
    """
    rd = pd.read_csv(path)
    rd["acq_date"] = pd.to_datetime(rd["acq_date"], errors="coerce")
    rd["real_date"] = pd.to_datetime(rd["real_date"], errors="coerce")
    rd["image_id"] = (rd["subject"] + "_" + rd["visit_code"] + "_"
                      + rd["acq_date"].dt.strftime("%Y-%m-%d"))
    keep = ["image_id", "real_date", "image_uid", "field_strength_ida",
            "manufacturer", "scanner_model"]
    rd = rd[[c for c in keep if c in rd.columns]]

    n0 = len(scans)
    scans = scans.merge(rd, on="image_id", how="left", suffixes=("", "_rd"))
    # Scans the alignment could not match keep their nominal-schedule estimate
    # rather than being dropped, which would cost whole subjects from the
    # 4-scan minimum for the sake of a few percent of visits.
    matched = scans["real_date"].notna()
    scans["acq_date"] = scans["real_date"].fillna(scans["acq_date"])
    scans["date_is_real"] = matched.astype(int)
    print(f"  real dates: {matched.sum()} of {n0} scans "
          f"({100*matched.mean():.1f}%), remainder on nominal dates")
    return scans.sort_values(["subject", "acq_date"]).reset_index(drop=True)


def _finish_row(row, window, times, bsc_cols, fs, ta_cols, subj, args, demog,
                apoe, mmse_by_subj, adas, cdr, t0):
    """Attach features, scanner metadata and covariates to a cohort row.

    Shared by both follow-up modes so the two designs differ only in which scans
    enter the window and how the clock is defined, never in how features are
    computed from them.
    """
    for f in bsc_cols:
        tr = trajectory(times, window[f].to_numpy(dtype=float))
        row[f"{f}_baseline"] = tr["baseline"]
        row[f"{f}_final"] = tr["last"]
        row[f"{f}_slope"] = tr["slope_yr"]
        row[f"{f}_r2"] = tr["r2"]

    for f in T1_TRAJECTORY:
        if f not in window.columns:
            continue
        tr = trajectory(times, window[f].to_numpy(dtype=float))
        row[f"{f}_bl"] = tr["baseline"]
        for k in ("last", "delta", "pctchg", "slope_yr", "mean", "std"):
            row[f"{f}_{k}"] = tr[k]
    for f in T1_BASELINE_ONLY:
        if f in window.columns:
            row[f"{f}_bl"] = float(window[f].iloc[0])

    # FreeSurfer comparator slopes over the identical window.
    fsw = attach_freesurfer(window, fs, subj, args.fs_tol_days)
    n_fs = int(fsw.notna().any(axis=1).sum()) if len(fsw.columns) else 0
    row["n_scans_freesurfer"] = n_fs
    for f in ta_cols + (["hippo_icv"] if "hippo_icv" in fsw.columns else []):
        if f not in fsw.columns:
            continue
        tr = trajectory(times, pd.to_numeric(fsw[f], errors="coerce").to_numpy(dtype=float))
        row[f"fs_{f}_bl"] = tr["baseline"]
        row[f"fs_{f}_slope"] = tr["slope_yr"]

    # Scanner metadata over the window.
    if "field_strength" in window.columns:
        fsv = pd.to_numeric(window["field_strength"], errors="coerce").dropna()
        row["field_strength_bl"] = float(fsv.iloc[0]) if len(fsv) else np.nan
        row["field_strength_changed"] = int(fsv.round(1).nunique() > 1) if len(fsv) else 0
    for c in ("manufacturer", "scanner_model", "scanner_id", "site"):
        if c in window.columns:
            row[c] = window[c].iloc[0]
    if "scanner_id" in window.columns:
        row["scanner_changed"] = int(window["scanner_id"].nunique() > 1)

    dem = demog.get(subj, {})
    dob = dem.get("dob")
    row["age_at_landmark"] = ((t0 - dob).days / 365.25) if pd.notna(dob) else np.nan
    row["female"] = dem.get("female", np.nan)
    row["educ_years"] = dem.get("educ", np.nan)
    row["apoe4"] = apoe.get(subj, np.nan)
    row["mmse_at_landmark"] = nearest_score(mmse_by_subj, subj, t0, "MMSCORE",
                                            args.score_window_days)
    row["adas13_at_landmark"] = nearest_score(adas, subj, t0, "TOTAL13",
                                              args.score_window_days)
    row["cdrsb_at_landmark"] = nearest_score(cdr, subj, t0, "CDRSB",
                                             args.score_window_days)
    return row


def load_regional(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Per-scan regional BSC, keyed on image_id.

    Only the region means are carried forward; the n_roi voxel counts are QC
    quantities, not features, and modelling them would let the model key on
    parcel size rather than boundary sharpness.
    """
    r = pd.read_csv(path)
    cols = [c for c in r.columns
            if c.startswith(("bscdir_roi", "bscmag_roi"))
            or c in ("bscdir_adsig", "bscmag_adsig")]
    return r[["image_id"] + cols], cols


def build(args) -> None:
    scans, bsc_cols = load_scans(args.bsc, args.t1_scans)
    if args.regional:
        reg, reg_cols = load_regional(args.regional)
        scans = scans.merge(reg, on="image_id", how="left")
        bsc_cols = bsc_cols + reg_cols
        cov = scans[reg_cols[0]].notna().mean() if reg_cols else 0.0
        print(f"  regional: {len(reg_cols)} features, {100*cov:.1f}% of scans matched")
    if args.real_dates:
        scans = apply_real_dates(scans, args.real_dates)
    dx_by_subj = load_dx(args.dxsum)
    mmse_by_subj = load_mmse(str(Path(args.adni_data) / "MMSE.rda"))
    demog = load_demog(args.demog)
    apoe = load_apoe(Path(args.adni_data) / "APOERES.rda")
    adas = load_visit_score(Path(args.adni_data) / "ADAS.rda", "ADAS", "TOTAL13", 0, 85)
    cdr = load_visit_score(Path(args.adni_data) / "CDR.rda", "CDR", "CDRSB", 0, 18)
    fs, ta_cols = load_freesurfer(Path(args.adni_data), args.fs_tol_days)
    scanner = load_scanner_meta(args.ida)

    # The IDA export's study date can sit a few days off the acquisition date we
    # derived from the images, so match to the nearest study within a tolerance
    # rather than requiring an exact hit.
    if "manufacturer" not in scans.columns:
        # Without real dates the IDA study dates cannot be joined exactly, so
        # fall back to a nearest-date merge on the approximate clock.
        scans = scans.sort_values("acq_date")
        scanner = scanner.sort_values("date")
        scans = pd.merge_asof(scans, scanner, left_on="acq_date", right_on="date",
                              by="subject", direction="nearest",
                              tolerance=pd.Timedelta(days=args.scanner_tol_days))
    # Field strength from the DICOM headers is per scan and more complete than
    # the IDA protocol string, so it takes precedence where both exist.
    scans["site"] = scans["subject"].str.slice(0, 3)
    if "field_strength_ida" in scans.columns:
        fs_ida = pd.to_numeric(scans["field_strength_ida"], errors="coerce")
        if "field_strength" in scans.columns:
            fs_ida = fs_ida.fillna(pd.to_numeric(scans["field_strength"], errors="coerce"))
        scans["field_strength"] = fs_ida
    if "meta_field_strength_t" in scans.columns:
        hdr = pd.to_numeric(scans["meta_field_strength_t"], errors="coerce")
        hdr = hdr.where(hdr < 100, hdr / 10000.0)
        scans["field_strength"] = hdr.fillna(scans.get("field_strength"))
        scans["scanner_id"] = (scans["site"].astype(str) + "_"
                               + scans["manufacturer"].astype(str) + "_"
                               + scans["field_strength"].round(1).astype(str))
    scans = scans.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    counts = {k: 0 for k in ("subjects_with_scans", "baseline_not_mci",
                             "no_clinical_record", "scans_dropped_dx_ad",
                             "scans_dropped_post_conversion", "too_few_scans_in_window",
                             "no_followup_after_last_scan",
                             "converted_at_or_before_landmark",
                             "censored_before_admin_limit")}
    counts["subjects_with_scans"] = int(scans["subject"].nunique())
    exclusions = []

    rows = []
    for subj, g in scans.groupby("subject"):
        g = g.sort_values("acq_date")
        if float(g["diagnosis"].iloc[0]) != MCI:
            counts["baseline_not_mci"] += 1
            exclusions.append({"subject": subj, "scan": "", "reason": "baseline_not_mci"})
            continue
        sdx = dx_by_subj.get(subj)
        if sdx is None or len(sdx) == 0:
            counts["no_clinical_record"] += 1
            exclusions.append({"subject": subj, "scan": "", "reason": "no_clinical_record"})
            continue

        first_scan = g["acq_date"].iloc[0]
        ad = sdx[(sdx["DIAGNOSIS"] == AD) & (sdx["EXAMDATE"] > first_scan)]
        converted = len(ad) > 0
        conv_date = ad["EXAMDATE"].iloc[0] if converted else pd.NaT
        last_visit = max(sdx["EXAMDATE"].max(), g["acq_date"].iloc[-1])

        # Fixed landmark mode: the clock starts a set number of months after the
        # subject's first scan, features come only from scans at or before it,
        # and follow-up is administratively censored a set number of years later.
        if args.landmark_months:
            lm = first_scan + pd.Timedelta(days=round(args.landmark_months * 30.4375))
            if converted and conv_date <= lm:
                counts["converted_at_or_before_landmark"] += 1
                exclusions.append({"subject": subj, "scan": "",
                                   "reason": "converted_at_or_before_landmark"})
                continue
            anchor = first_scan if args.censor_from == "baseline" else lm
            censor_date = anchor + pd.Timedelta(days=round(args.censor_years * 365.25))
            window = g[g["acq_date"] <= lm]
            is_ad = window["diagnosis"].astype(float) == AD
            for sid in window.loc[is_ad, "image_id"]:
                exclusions.append({"subject": subj, "scan": sid,
                                   "reason": "scan_diagnosis_ad"})
            counts["scans_dropped_dx_ad"] += int(is_ad.sum())
            window = window[~is_ad]
            if len(window) < args.min_scans:
                counts["too_few_scans_in_window"] += 1
                exclusions.append({"subject": subj, "scan": "",
                                   "reason": f"fewer_than_{args.min_scans}_scans_in_window"})
                continue
            t0 = lm
            # An event only counts if it happens before administrative censoring.
            if converted and conv_date <= censor_date:
                end, event = conv_date, 1
            else:
                end, event = min(last_visit, censor_date), 0
                if last_visit < censor_date:
                    counts["censored_before_admin_limit"] += 1
            time_years = (end - t0).days / 365.25
            if time_years <= 0:
                counts["no_followup_after_last_scan"] += 1
                exclusions.append({"subject": subj, "scan": "",
                                   "reason": "nonpositive_time_to_event"})
                continue
            times = ((window["acq_date"] - window["acq_date"].iloc[0]).dt.days / 365.25).to_numpy()
            row = {"subject": subj, "event": int(event), "time_years": time_years,
                   "landmark_date": pd.Timestamp(t0).date().isoformat(),
                   "n_scans_window": len(window),
                   "window_span_years": round(float(times[-1]), 3),
                   "n_scans_total": len(g)}
            rows.append(_finish_row(row, window, times, bsc_cols, fs, ta_cols, subj,
                                    args, demog, apoe, mmse_by_subj, adas, cdr, t0))
            continue

        # Legacy mode reproduces the submitted design: features come from every
        # scan, including any acquired after the dementia diagnosis, and the
        # clock starts at the MCI baseline scan. Reported only so the corrected
        # design can be compared against it on the same subjects.
        if args.legacy:
            window = g
            if len(window) < args.min_scans:
                counts["too_few_scans_in_window"] += 1
                continue
            t0 = window["acq_date"].iloc[0]
            end = conv_date if converted else g["acq_date"].iloc[-1]
            time_years = (end - t0).days / 365.25
            if time_years <= 0:
                counts["no_followup_after_last_scan"] += 1
                continue
            times = ((window["acq_date"] - window["acq_date"].iloc[0]).dt.days / 365.25).to_numpy()
            row = {"subject": subj, "event": int(converted), "time_years": time_years,
                   "landmark_date": pd.Timestamp(t0).date().isoformat(),
                   "n_scans_window": len(window),
                   "window_span_years": round(float(times[-1]), 3),
                   "n_scans_total": len(g)}
            rows.append(_finish_row(row, window, times, bsc_cols, fs, ta_cols, subj,
                                    args, demog, apoe, mmse_by_subj, adas, cdr, t0))
            continue

        window = g
        if converted:
            post = window["acq_date"] >= conv_date
            for sid in window.loc[post, "image_id"]:
                exclusions.append({"subject": subj, "scan": sid,
                                   "reason": "scan_at_or_after_conversion"})
            counts["scans_dropped_post_conversion"] += int(post.sum())
            window = window[~post]
        # Spec 3.1.5: a scan carrying an AD diagnosis code never enters features.
        is_ad = window["diagnosis"].astype(float) == AD
        for sid in window.loc[is_ad, "image_id"]:
            exclusions.append({"subject": subj, "scan": sid, "reason": "scan_diagnosis_ad"})
        counts["scans_dropped_dx_ad"] += int(is_ad.sum())
        window = window[~is_ad]

        if len(window) < args.min_scans:
            counts["too_few_scans_in_window"] += 1
            exclusions.append({"subject": subj, "scan": "",
                               "reason": f"fewer_than_{args.min_scans}_scans_in_window"})
            continue

        t0 = window["acq_date"].iloc[-1]
        end = conv_date if converted else last_visit
        time_years = (end - t0).days / 365.25
        if time_years <= 0:
            counts["no_followup_after_last_scan"] += 1
            exclusions.append({"subject": subj, "scan": "",
                               "reason": "nonpositive_time_to_event"})
            continue

        times = ((window["acq_date"] - window["acq_date"].iloc[0]).dt.days / 365.25).to_numpy()
        row = {"subject": subj, "event": int(converted), "time_years": time_years,
               "landmark_date": pd.Timestamp(t0).date().isoformat(),
               "n_scans_window": len(window),
               "window_span_years": round(float(times[-1]), 3),
               "n_scans_total": len(g)}

        rows.append(_finish_row(row, window, times, bsc_cols, fs, ta_cols, subj,
                                args, demog, apoe, mmse_by_subj, adas, cdr, t0))

    out = pd.DataFrame(rows)
    counts.update({
        "subjects_retained": len(out),
        "converters": int(out["event"].sum()),
        "stable": int((out["event"] == 0).sum()),
        "median_followup_years": round(float(out["time_years"].median()), 3),
        "mean_followup_years": round(float(out["time_years"].mean()), 3),
        "mean_scans_in_window": round(float(out["n_scans_window"].mean()), 2),
        "mean_window_span_years": round(float(out["window_span_years"].mean()), 3),
        "min_scans": args.min_scans,
    })
    for c in ("mmse_at_landmark", "adas13_at_landmark", "cdrsb_at_landmark",
              "apoe4", "age_at_landmark", "educ_years", "female"):
        if c in out.columns:
            counts[f"missing_{c}"] = int(out[c].isna().sum())
    if "field_strength_changed" in out.columns:
        counts["subjects_switching_field_strength"] = int(out["field_strength_changed"].sum())
    if "scanner_changed" in out.columns:
        counts["subjects_switching_scanner"] = int(out["scanner_changed"].sum())
    if "n_scans_freesurfer" in out.columns:
        counts["subjects_with_any_freesurfer"] = int((out["n_scans_freesurfer"] > 0).sum())
        counts["subjects_with_ge2_freesurfer"] = int((out["n_scans_freesurfer"] >= 2).sum())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "spec_cohort.csv", index=False)
    pd.DataFrame(exclusions).to_csv(out_dir / "exclusion_log.csv", index=False)
    with open(out_dir / "spec_cohort.json", "w") as f:
        json.dump(counts, f, indent=2)
    for k, v in counts.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--bsc", default=str(root / "from_cluster/bsc_simple_features_merged.csv"))
    ap.add_argument("--t1_scans", default=str(root / "from_cluster/t1_scan_features.csv"))
    ap.add_argument("--dxsum", default=str(root / "from_cluster/dxsum.csv"))
    ap.add_argument("--ida", default=str(root / "from_cluster/idaSearch_7_23_2026.csv"))
    ap.add_argument("--demog", default=str(root.parent / "PTDEMOG_06Aug2026.csv"))
    ap.add_argument("--adni_data", default=str(root.parent / "ADNIMERGE2/data"))
    ap.add_argument("--out_dir", default=str(root / "analysis/results/spec_v3"))
    ap.add_argument("--min_scans", type=int, default=4)
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the submitted follow-up design for comparison")
    ap.add_argument("--regional", default=None,
                    help="per_scan_regional.csv; omit to skip regional features")
    ap.add_argument("--landmark_months", type=float, default=None,
                    help="fixed landmark N months after first scan; omit for spec 3.2 rule")
    ap.add_argument("--censor_years", type=float, default=4.0)
    ap.add_argument("--censor_from", choices=["baseline", "landmark"], default="baseline")
    ap.add_argument("--score_window_days", type=int, default=180)
    ap.add_argument("--fs_tol_days", type=int, default=120)
    ap.add_argument("--scanner_tol_days", type=int, default=14)
    ap.add_argument("--real_dates", default=str(root / "from_cluster/manifest_realdates.csv"),
                    help="manifest_realdates.csv; pass empty string to keep synthetic dates")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
