"""Apply LongComBat to the per-scan BSC features and report batch diagnostics."""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import longcombat as lc

ap = argparse.ArgumentParser()
root = Path(__file__).resolve().parents[1]
ap.add_argument("--bsc", default=str(root/"from_cluster/bsc_simple_features_merged.csv"))
ap.add_argument("--realdates", default=str(root/"from_cluster/manifest_realdates.csv"))
ap.add_argument("--out", default=str(root/"from_cluster/bsc_simple_features_harmonized.csv"))
ap.add_argument("--regional", default=None)
ap.add_argument("--regional_out", default=str(root/"from_cluster/per_scan_regional_harmonized.csv"))
ap.add_argument("--diag_dir", default=str(root/"analysis/results/harmonization"))
a = ap.parse_args()

bsc = pd.read_csv(a.bsc)
rd = pd.read_csv(a.realdates)
rd["image_id"] = (rd.subject+"_"+rd.visit_code+"_"
                  + pd.to_datetime(rd.acq_date).dt.strftime("%Y-%m-%d"))
meta = rd[["image_id","field_strength_ida","manufacturer","scanner_model","real_date"]]
d = bsc.merge(meta, on="image_id", how="left")

d["site"] = d.subject.str.slice(0,3)
fs = pd.to_numeric(d.field_strength_ida, errors="coerce")
d["fs_bin"] = np.where(fs < 2.25, "1.5T", "3T")
# Batch = scanner identity. Vendor and field strength drive the step changes;
# site is folded in because coil and sequence tuning vary by centre.
d["scanner_id"] = (d.manufacturer.fillna("unk").astype(str).str.split().str[0]
                   + "_" + d.fs_bin + "_" + d.site.astype(str))
# Batches with too few scans cannot support a variance estimate; pool them by
# vendor and field strength rather than dropping the scans.
vc = d.scanner_id.value_counts()
d["batch"] = np.where(d.scanner_id.map(vc) >= 15, d.scanner_id,
                      d.manufacturer.fillna("unk").astype(str).str.split().str[0]+"_"+d.fs_bin)

feats = [c for c in bsc.columns if c.startswith("bsc_") or c == "Nboundary"]
d["dx"] = pd.to_numeric(d.diagnosis, errors="coerce").fillna(2.0)
d["acq_dt"] = pd.to_datetime(d.real_date.fillna(d.acq_date), errors="coerce")
d["age_proxy"] = (d.acq_dt - d.groupby("subject").acq_dt.transform("min")).dt.days/365.25

pre = lc.batch_diagnostics(d, feats, "batch")
params = lc.fit(d, feats, batch_col="batch", covars=["age_proxy","dx"],
                subject_col="subject")
h = lc.apply(d, params)
post = lc.batch_diagnostics(h, feats, "batch")

Path(a.diag_dir).mkdir(parents=True, exist_ok=True)
cmp = pre[["feature","p"]].rename(columns={"p":"p_before"}).merge(
      post[["feature","p"]].rename(columns={"p":"p_after"}), on="feature")
cmp.to_csv(Path(a.diag_dir)/"kruskal_before_after.csv", index=False)
h[bsc.columns].to_csv(a.out, index=False)

print(f"batches: {d.batch.nunique()}  scans: {len(d)}  features: {len(feats)}")
print(f"Kruskal-Wallis p>0.05 (no residual batch effect):")
print(f"  before harmonization: {(cmp.p_before>0.05).sum()}/{len(cmp)}")
print(f"  after  harmonization: {(cmp.p_after>0.05).sum()}/{len(cmp)}")
print(f"median p: {cmp.p_before.median():.2e} -> {cmp.p_after.median():.2e}")

if a.regional:
    # Regional features get the same batch model. Scanner effects are larger
    # here than globally, since each parcel rests on far fewer voxels.
    reg = pd.read_csv(a.regional)
    rcols = [c for c in reg.columns
             if c.startswith(("bscdir_roi", "bscmag_roi"))
             or c in ("bscdir_adsig", "bscmag_adsig")]
    rd2 = reg.merge(d[["image_id", "batch", "dx", "age_proxy"]], on="image_id", how="inner")
    rp = lc.fit(rd2, rcols, batch_col="batch", covars=["age_proxy", "dx"],
                subject_col="subject")
    rh = lc.apply(rd2, rp)
    keep = [c for c in reg.columns if c not in rcols]
    out = reg[keep].merge(rh[["image_id"] + rcols], on="image_id", how="left")
    out.to_csv(a.regional_out, index=False)
    pre_r = lc.batch_diagnostics(rd2, rcols[:40], "batch")
    post_r = lc.batch_diagnostics(rh, rcols[:40], "batch")
    print(f"\nregional: {len(rcols)} features, {len(rd2)} scans matched to a batch")
    print(f"  median Kruskal p: {pre_r.p.median():.2e} -> {post_r.p.median():.2e}")
    print(f"  wrote {a.regional_out}")
