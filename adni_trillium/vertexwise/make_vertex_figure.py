"""Figure 8: why the band cannot be sampled, and how far the null was from passing.

Four panels, in the paper's house style (Okabe-Ito, 7.2 pt, 300 dpi):

  A  Distribution across all scans of the fraction of cortex vertices that
     receive a non-zero gBSC sample when the stored boundary-band map is
     projected onto the white surface. A quarter of scans sit at zero, which
     is the reason the dense field had to be recomputed.
  B  The same single scan twice on the inflated surface: the band's footprint,
     then the dense field evaluated at the surface. Same colour scale.
  C  Permutation null of the largest cluster area under Model A, with the
     observed largest cluster and the critical area marked.
  D  The same for Model B.

Panels C and D are the ones that turn "no cluster survived" into "the largest
cluster found was a third of the size it needed to be", which is the stronger
statement and the one a reader cannot get from a map of noise.

Inputs:
  --results      directory written by run_vertexwise.sh fetch
  --coverage     CSV from band_surface_coverage.py (columns image_id, coverage)
  --surfaces     NPZ from band_surface_coverage.py holding the two example maps
  --fs_subjects_dir  local copy of fsaverage6

Usage:
    python3 make_vertex_figure.py --results results --coverage results/band_coverage.csv \
        --surfaces results/band_example.npz --fs_subjects_dir fs_subjects \
        --out ../../mri-bsc/paper/aperture_neuro/fig8.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from nibabel.freesurfer import read_geometry, read_label  # noqa: E402

TRG = "fsaverage6"
BLUE, VERM, GREEN, VIOLET = "#0072B2", "#D55E00", "#009E73", "#8B5FA8"
INK, MUTED, RULE, NULLC = "#1a1a1a", "#6b6b6b", "#cfcfcf", "#8a8a8a"


def clean_rc():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 7.8,
        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })


def panel_letter(ax, letter, x=-0.14, y=1.06):
    # Axes3D.text needs a z; text2D takes the same 2-D axes coordinates.
    fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    fn(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
       va="bottom", ha="left", color=INK)


# ---------------------------------------------------------------- panel A
def draw_coverage(ax, cov: np.ndarray):
    """Fraction of cortex vertices receiving a non-zero band sample, per scan."""
    pct = 100.0 * cov
    bins = np.linspace(0, 100, 51)
    ax.hist(pct, bins=bins, color=MUTED, edgecolor="none", zorder=3)
    n_zero = int((cov <= 0.005).sum())
    ax.hist(pct[cov <= 0.005], bins=bins, color=VERM, edgecolor="none", zorder=4)

    ax.set_xlabel("cortex vertices with a non-zero band sample (%)")
    ax.set_ylabel("scans")
    ax.set_xlim(-2, 100)
    med = float(np.median(pct))
    ax.axvline(med, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=5)
    ax.annotate(f"median {med:.0f}%", xy=(med, ax.get_ylim()[1] * 0.93),
                xytext=(4, 0), textcoords="offset points", fontsize=6.4,
                color=INK, va="top", ha="left")
    top = ax.get_ylim()[1]
    ax.annotate(f"{n_zero} scans ({100*n_zero/len(cov):.0f}%)\nsample nothing at all",
                xy=(1.2, n_zero * 0.55), xycoords="data",
                xytext=(38, top * 0.62), textcoords="data",
                fontsize=6.4, color=VERM, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=VERM,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=-0.18"))
    ax.annotate(f"never above {100*cov.max():.0f}% on any scan",
                xy=(100 * float(cov.max()), 0), xycoords="data",
                xytext=(38, top * 0.30), textcoords="data",
                fontsize=6.4, color=MUTED, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=0.18"))
    ax.set_title("The stored band is not a surface", loc="left", color=INK)


# ---------------------------------------------------------------- panel B
def draw_hemi(ax, coords, faces, rgba, elev=0, azim=180):
    """Back-face-culled, depth-sorted render; matplotlib has no depth buffer."""
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tri = coords[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = normals @ view > 0
    keep = np.flatnonzero(front)[np.argsort((tri.mean(axis=1) @ view)[front])]
    ax.add_collection3d(Poly3DCollection(tri[keep], facecolors=rgba[keep],
                                         edgecolors="none", linewidths=0))
    c = coords.mean(axis=0)
    r = np.abs(coords - c).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.75)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def face_rgba(vals, faces, cmap, vmax):
    """Grey where the map is zero, colour-mapped where it is not."""
    fv = vals[faces].mean(axis=1)
    on = (vals[faces] != 0).all(axis=1)
    rgba = np.tile(np.array([0.82, 0.82, 0.82, 1.0]), (len(faces), 1))
    rgba[on] = cmap(np.clip(fv[on] / vmax, 0, 1))
    return rgba


def draw_example(ax, coords, faces, vals, cmap, vmax, title, subtitle, color):
    draw_hemi(ax, coords, faces, face_rgba(vals, faces, cmap, vmax))
    ax.set_title(title, loc="center", color=INK, pad=-2)
    ax.text2D(0.5, -0.02, subtitle, transform=ax.transAxes, fontsize=6.4,
              color=color, ha="center", va="top")


# ---------------------------------------------------------------- panels C, D
def draw_null(ax, null, observed, obs_label, cwp, letter_title, alpha=0.05):
    crit = float(np.percentile(null, 100 * (1 - alpha)))
    hi = max(crit, observed) * 1.28
    bins = np.linspace(0, hi, 46)
    ax.hist(null, bins=bins, color=NULLC, edgecolor="none", zorder=3)

    ax.axvline(crit, color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.axvline(observed, color=VERM, lw=1.6, zorder=6)

    top = ax.get_ylim()[1]
    ax.annotate(f"critical\n{crit:.0f} mm$^2$", xy=(crit, top * 0.88),
                xytext=(4, 0), textcoords="offset points", fontsize=6.4,
                color=INK, va="top", ha="left")
    ax.annotate(f"largest observed\n{observed:.0f} mm$^2$ · {obs_label}\ncorrected $p$ = {cwp:.2f}",
                xy=(observed, top * 0.55), xytext=(6, 0), textcoords="offset points",
                fontsize=6.4, color=VERM, va="center", ha="left")

    ax.set_xlabel("largest cluster area under permutation (mm$^2$)")
    ax.set_ylabel("permutations")
    ax.set_xlim(0, hi)
    ax.set_title(letter_title, loc="left", color=INK)


def load_null(res: Path, model: str) -> np.ndarray:
    """Per-iteration maximum across hemispheres, as combine_clusters.py uses."""
    return np.maximum(np.load(res / f"py{model}_lh" / "perm_max_area.npy"),
                      np.load(res / f"py{model}_rh" / "perm_max_area.npy"))


def observed_largest(res: Path, model: str):
    df = pd.read_csv(res / f"model{model}_cluster_summary.csv")
    r = df.loc[df["area_mm2"].idxmax()]
    label = str(r["peak_aparc"]).replace("superiorfrontal", "sup. frontal") \
                                .replace("middletemporal", "mid. temporal")
    return float(r["area_mm2"]), f"{r['hemi']} {label}", float(r["cwp"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--coverage", default=None)
    ap.add_argument("--surfaces", default=None)
    ap.add_argument("--fs_subjects_dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    clean_rc()

    have_band = (args.coverage and Path(args.coverage).exists()
                 and args.surfaces and Path(args.surfaces).exists())

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 0.92],
                  hspace=0.62, wspace=0.55,
                  left=0.075, right=0.985, top=0.93, bottom=0.10)

    # ---- A: coverage
    axA = fig.add_subplot(gs[0, 0:2])
    if have_band:
        cov = pd.read_csv(args.coverage)["coverage"].to_numpy(dtype=float)
        draw_coverage(axA, cov)
    else:
        axA.text(0.5, 0.5, "band coverage pending", ha="center", va="center",
                 transform=axA.transAxes, color=MUTED)
        axA.set_xticks([]); axA.set_yticks([])
    panel_letter(axA, "A")

    # ---- B: the same scan, band footprint then dense field
    if have_band and args.fs_subjects_dir:
        fsd = Path(args.fs_subjects_dir)
        coords, faces = read_geometry(str(fsd / TRG / "surf" / "lh.inflated"))
        ctx = np.zeros(coords.shape[0], dtype=bool)
        ctx[read_label(str(fsd / TRG / "label" / "lh.cortex.label"))] = True
        z = np.load(args.surfaces)
        band, field = z["band"].astype(float), z["field"].astype(float)
        band[~ctx] = 0.0
        field[~ctx] = 0.0
        vmax = float(np.percentile(field[field != 0], 98))
        cmap = plt.get_cmap("magma")
        axB1 = fig.add_subplot(gs[0, 2], projection="3d")
        draw_example(axB1, coords, faces, band, cmap, vmax, "stored band map",
                     f"{100*(band != 0)[ctx].mean():.0f}% of cortex", VERM)
        panel_letter(axB1, "B", x=-0.06, y=0.94)
        axB2 = fig.add_subplot(gs[0, 3], projection="3d")
        draw_example(axB2, coords, faces, field, cmap, vmax, "recomputed field",
                     f"{100*(field != 0)[ctx].mean():.0f}% of cortex", GREEN)
    else:
        axB = fig.add_subplot(gs[0, 2:4])
        axB.text(0.5, 0.5, "example surfaces pending", ha="center", va="center",
                 transform=axB.transAxes, color=MUTED)
        axB.set_xticks([]); axB.set_yticks([])
        panel_letter(axB, "B")

    # ---- C, D: permutation nulls
    for col, model, title in ((0, "A", "Model A · unadjusted"),
                              (2, "B", "Model B · adjusted for thickness and curvature")):
        ax = fig.add_subplot(gs[1, col:col + 2])
        obs, label, cwp = observed_largest(res, model)
        draw_null(ax, load_null(res, model), obs, label, cwp, title)
        panel_letter(ax, "C" if model == "A" else "D")

    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}"
          + ("" if have_band else "  [panels A and B are placeholders]"))


if __name__ == "__main__":
    main()
