"""Build a synthetic fsaverage6-shaped world to exercise the pipeline end to end.

A 2,500-vertex triangulated grid per hemisphere with a 16-parcel annotation
(the 7 AD-signature names and 9 others), 40 subjects at 5 to 8 scans over 2 to
6 years, spatially smooth noise, per-subject noise scales differing threefold,
and an optional slope planted in right middle temporal.

The smoothing is not cosmetic. With independent per-vertex noise the
suprathreshold clusters are single vertices, the permutation null collapses to
1 or 2 mm2 and the cluster test stops being a cluster test; the first
calibration run done that way was invalid and said so by rejecting at the same
rate for every alpha.

Usage:
    python3 make_synthetic_world.py OUT_DIR [PLANTED_SLOPE]
        PLANTED_SLOPE -0.03 for the recovery test, 0 for calibration
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
import nibabel as nib
from nibabel.freesurfer import write_geometry, write_annot
from scipy.ndimage import gaussian_filter

root = Path(sys.argv[1]); planted = float(sys.argv[2]) if len(sys.argv) > 2 else -0.030
rng = np.random.default_rng(0)
G = 50                                   # G x G triangulated grid per hemisphere
SIG = ["entorhinal","parahippocampal","inferiortemporal","middletemporal",
       "precuneus","posteriorcingulate","fusiform"]
OTHER = ["superiorfrontal","precentral","supramarginal","lingual","insula",
         "cuneus","paracentral","parsopercularis","temporalpole"]
NAMES = ["unknown"] + SIG + OTHER

fsd = root / "fsdir" / "fsaverage6"
(fsd / "surf").mkdir(parents=True, exist_ok=True)
(fsd / "label").mkdir(parents=True, exist_ok=True)

SMOOTH = 3.0          # mm; real surface maps carry 10 mm FWHM, and without any
                      # spatial correlation the cluster null is single vertices
                      # and the whole cluster test degenerates


def field(scale):
    """Spatially smooth noise on the grid, scaled to the requested SD."""
    z = gaussian_filter(rng.normal(size=(G, G)), SMOOTH, mode="nearest")
    return (scale * z / z.std()).ravel()


ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
coords = np.column_stack([ii.ravel(), jj.ravel(), np.zeros(G*G)]).astype(np.float32)
faces = []
for i in range(G-1):
    for j in range(G-1):
        a, b, c, d = i*G+j, i*G+j+1, (i+1)*G+j, (i+1)*G+j+1
        faces += [[a, b, c], [b, d, c]]
faces = np.array(faces, dtype=np.int32)

# strips across j: 16 parcels, the 7 signature ones a contiguous block
strip = (jj.ravel() * len(NAMES[1:]) // G) + 1
border = (ii.ravel() == 0) | (ii.ravel() == G-1) | (jj.ravel() == 0) | (jj.ravel() == G-1)
labels = np.where(border, 0, strip).astype(np.int32)
ctab = np.column_stack([rng.integers(0, 255, (len(NAMES), 3)),
                        np.zeros(len(NAMES)), np.arange(len(NAMES))]).astype(np.int32)
cortex_idx = np.flatnonzero(~border)

for hemi in ("lh", "rh"):
    write_geometry(str(fsd / "surf" / f"{hemi}.white"), coords, faces)
    write_annot(str(fsd / "label" / f"{hemi}.aparc.annot"), labels, ctab, NAMES,
                fill_ctab=True)
    with open(fsd / "label" / f"{hemi}.cortex.label", "w") as f:
        f.write(f"#!ascii label\n{len(cortex_idx)}\n")
        for v in cortex_idx:
            f.write(f"{v} {coords[v,0]:.3f} {coords[v,1]:.3f} {coords[v,2]:.3f} 0.0\n")

# the planted effect lives in middletemporal of the right hemisphere only
mt = labels == (NAMES.index("middletemporal"))
print(f"mesh {G*G} vertices/hemi, cortex {len(cortex_idx)}, "
      f"signature {int(np.isin(labels, [NAMES.index(s) for s in SIG]).sum())}, "
      f"middletemporal {int(mt.sum())}")

NSUBJ, NSCAN = 40, 5
subj_rows, scan_rows = [], []
surf = root / "surf"
for s in range(NSUBJ):
    sid = f"TEST_S_{s:04d}"
    event = 1 if s < NSUBJ // 2 else 0
    # unequal scan counts and spans, which is what the weights exist for
    nscan = NSCAN + (s % 4)
    span = 2.0 + 4.0 * (s % 3) / 2
    noise_scale = 0.010 * (1.0 + 2.0 * (s % 5) / 4)
    t = np.linspace(0.0, span, nscan)
    base = {h: 1.0 + field(0.05) for h in ("lh", "rh")}
    slope = {h: field(0.002) for h in ("lh", "rh")}
    if event:
        slope["rh"] = slope["rh"] + planted * mt
    for k in range(nscan):
        iid = f"{sid}_m{k*12:02d}"
        d = surf / iid
        d.mkdir(parents=True, exist_ok=True)
        for h in ("lh", "rh"):
            g = base[h] + slope[h] * t[k] + field(noise_scale)
            for meas, vals in (("gbsc", g),
                               ("thickness", 2.5 + field(0.1)),
                               ("curv", field(0.01))):
                nib.save(nib.MGHImage(vals.astype(np.float32).reshape(-1,1,1), np.eye(4)),
                         str(d / f"{h}.{meas}.fsaverage6.mgh"))
        scan_rows.append({"image_id": iid, "subject": sid,
                          "t1_path": f"/fake/sub-{sid}_acq-{'3' if s%2 else '15'}T_T1w.nii.gz",
                          "bsc_dir": f"/fake/bsc/{iid}", "date": "2010-01-01",
                          "years": float(t[k]),
                          "field_strength_ida": 3.0 if s % 2 else 1.5})
    subj_rows.append({"subject": sid, "event": event, "time_years": span,
                      "n_scans_window": nscan, "age_at_landmark": 70 + rng.normal()*7,
                      "female": float(s % 2), "apoe4": float(s % 3 == 0),
                      "field_strength_bl": 3.0 if s % 2 else 1.5,
                      "field_strength_changed": 1 if s % 9 == 0 else 0})

coh = root / "cohort"; coh.mkdir(exist_ok=True)
pd.DataFrame(scan_rows).to_csv(coh / "vertex_scans.csv", index=False)
pd.DataFrame(subj_rows).to_csv(coh / "vertex_subjects.csv", index=False)
print(f"wrote {len(scan_rows)} scans, {len(subj_rows)} subjects "
      f"({sum(r['event'] for r in subj_rows)} converters), planted slope {planted:+.3f} "
      f"in rh middletemporal")
