"""Table 1 demographics for the V3 revision spec cohort."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "results/spec_v3_harmonized/spec_cohort.csv")
fs = pd.to_numeric(d["field_strength_bl"], errors="coerce")
d["fs3t"] = (fs >= 2.25).astype(float)

conv, stab = d[d.event == 1], d[d.event == 0]
rows = []

def cont(label, col, dec=1):
    a, b = conv[col].dropna(), stab[col].dropna()
    p = stats.mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan
    rows.append({"variable": label,
                 "converters": f"{a.mean():.{dec}f} ({a.std():.{dec}f})",
                 "stable": f"{b.mean():.{dec}f} ({b.std():.{dec}f})",
                 "overall": f"{d[col].dropna().mean():.{dec}f} ({d[col].dropna().std():.{dec}f})",
                 "p": p, "n_missing": int(d[col].isna().sum())})

def binary(label, col, pct_of=1.0):
    a, b = conv[col].dropna(), stab[col].dropna()
    tab = [[int(a.sum()), len(a) - int(a.sum())], [int(b.sum()), len(b) - int(b.sum())]]
    p = stats.fisher_exact(tab).pvalue if min(len(a), len(b)) > 0 else np.nan
    rows.append({"variable": label,
                 "converters": f"{int(a.sum())} ({100*a.mean():.1f}%)",
                 "stable": f"{int(b.sum())} ({100*b.mean():.1f}%)",
                 "overall": f"{int(d[col].sum())} ({100*d[col].mean():.1f}%)",
                 "p": p, "n_missing": int(d[col].isna().sum())})

rows.append({"variable": "N", "converters": str(len(conv)), "stable": str(len(stab)),
             "overall": str(len(d)), "p": np.nan, "n_missing": 0})
cont("Age at landmark, y", "age_at_landmark")
binary("Female", "female")
cont("Education, y", "educ_years")
cont("APOE4 alleles", "apoe4", 2)
binary("APOE4 carrier", "apoe4_pos")
cont("MMSE", "mmse_at_landmark")
cont("ADAS-Cog13", "adas13_at_landmark")
cont("CDR-SB", "cdrsb_at_landmark", 2)
binary("3T at landmark", "fs3t")
binary("Switched field strength", "field_strength_changed")
cont("Scans in window", "n_scans_window", 2)
cont("Window span, y", "window_span_years", 2)
cont("Follow-up after last scan, y", "time_years", 2)

out = pd.DataFrame(rows)
out["p"] = out["p"].map(lambda v: "" if pd.isna(v) else (f"{v:.3g}" if v >= 0.001 else "<0.001"))
out.to_csv(root / "results/spec_v3_harmonized/table1_demographics.csv", index=False)
print(out.to_string(index=False))
print()
print("median follow-up (IQR): converters "
      f"{conv.time_years.median():.2f} ({conv.time_years.quantile(.25):.2f}-{conv.time_years.quantile(.75):.2f}); "
      f"stable {stab.time_years.median():.2f} ({stab.time_years.quantile(.25):.2f}-{stab.time_years.quantile(.75):.2f})")
print(f"total scans in feature windows: {int(d.n_scans_window.sum())}")
