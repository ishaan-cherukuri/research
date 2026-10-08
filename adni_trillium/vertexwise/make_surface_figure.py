"""Figure 8: Model B clusters on the inflated fsaverage6 surface.

Lateral and medial views of both hemispheres, sulcal depth in grey, and the
converter-minus-non-converter t painted inside clusters that survive
cluster-wise correction. Rendered with matplotlib alone so that it runs in the
same environment as the other figure scripts; the fsaverage6 surfaces come from
--fs_subjects_dir (copy fsaverage6/surf and label from Trillium, see README).

--mode clusters (default) paints t only inside clusters that survive
correction. --mode tmap paints the unthresholded t everywhere in cortex and
outlines the clusters formed at the cluster-forming threshold, which is the
right picture when nothing survives. With --fallback_model_a, vertices in
Model A's surviving clusters are outlined underneath.

Usage:
    python3 make_surface_figure.py --results results --fs_subjects_dir ~/fs_subjects \
        --out ../../mri-bsc/paper/aperture_neuro/fig8.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from nibabel.freesurfer import read_geometry, read_morph_data  # noqa: E402

TRG = "fsaverage6"
VIEWS = {"lh": {"lateral": (0, 180), "medial": (0, 0)},
         "rh": {"lateral": (0, 0), "medial": (0, 180)}}


def load(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float64).reshape(-1)


def face_colors(vals_v: np.ndarray, faces: np.ndarray, sulc: np.ndarray, norm, cmap, mask_v):
    """Per-face RGBA: grey sulcal shading, overlaid where all three vertices are in a cluster."""
    sulc_f = sulc[faces].mean(axis=1)
    g = 0.72 - 0.22 * (sulc_f > 0)                      # sulci darker than gyri
    rgba = np.stack([g, g, g, np.ones_like(g)], axis=1)
    on = mask_v[faces].all(axis=1)
    rgba[on] = cmap(norm(vals_v[faces][on].mean(axis=1)))
    return rgba


def draw_hemi(ax, coords, faces, rgba, elev, azim, outline_faces=None):
    """matplotlib has no depth buffer, so keep only faces that face the camera
    and draw them back to front."""
    el, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    tri = coords[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = normals @ view > 0
    depth = tri.mean(axis=1) @ view
    order = np.argsort(depth[front])
    keep = np.flatnonzero(front)[order]
    pc = Poly3DCollection(tri[keep], facecolors=rgba[keep], edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    if outline_faces is not None:
        ok = keep[outline_faces[keep]]
        if ok.size:
            oc = Poly3DCollection(tri[ok], facecolors=(0, 0, 0, 0),
                                  edgecolors=(0.0, 0.0, 0.0, 0.9), linewidths=0.6)
            ax.add_collection3d(oc)
    c = coords.mean(axis=0)
    r = np.abs(coords - c).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.9)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="directory from run_vertexwise.sh fetch")
    ap.add_argument("--fs_subjects_dir", required=True)
    ap.add_argument("--model", default="B", choices=["A", "B"])
    ap.add_argument("--fallback_model_a", action="store_true")
    ap.add_argument("--mode", choices=["clusters", "tmap"], default="clusters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmax", type=float, default=None)
    args = ap.parse_args()
    res, fsd = Path(args.results), Path(args.fs_subjects_dir)

    data = {}
    for hemi in ("lh", "rh"):
        coords, faces = read_geometry(str(fsd / TRG / "surf" / f"{hemi}.inflated"))
        sulc = read_morph_data(str(fsd / TRG / "surf" / f"{hemi}.sulc"))
        sig = load(res / f"{hemi}.model{args.model}.sig_clusters.mgh")
        outline = None
        if args.mode == "tmap":
            d = res / f"py{args.model}_{hemi}"
            sig = load(d / "t.mgh")
            cid = load(d / "cluster_id.mgh") > 0
            outline = cid[faces].all(axis=1)
        elif args.fallback_model_a:
            a = load(res / f"{hemi}.modelA.sig_clusters.mgh") != 0
            outline = a[faces].all(axis=1)
        data[hemi] = (coords, faces, sulc, sig, outline)

    allsig = np.concatenate([d[3] for d in data.values()])
    tmax = args.tmax or (4.0 if args.mode == "tmap" else
                         (float(np.abs(allsig[allsig != 0]).max()) if (allsig != 0).any() else 4.0))
    info = res / f"py{args.model}_lh" / "info.json"
    tmin = json.load(open(info))["tcrit"] if info.exists() else float("nan")
    norm = Normalize(vmin=-tmax, vmax=tmax)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(8.0, 5.2), dpi=300)
    panels = [("lh", "lateral"), ("rh", "lateral"), ("lh", "medial"), ("rh", "medial")]
    for i, (hemi, view) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        coords, faces, sulc, sig, outline = data[hemi]
        rgba = face_colors(sig, faces, sulc, norm, cmap, sig != 0)
        elev, azim = VIEWS[hemi][view]
        draw_hemi(ax, coords, faces, rgba, elev, azim, outline)
        ax.set_title(f"{'Left' if hemi == 'lh' else 'Right'} {view}", fontsize=9, pad=0)
    fig.subplots_adjust(left=0.01, right=0.88, top=0.95, bottom=0.03, wspace=0, hspace=0)
    cax = fig.add_axes([0.905, 0.25, 0.018, 0.5])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(rf"$t$, converter $-$ non-converter (Model {args.model})", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    n_sig = int((allsig != 0).sum())
    if args.mode == "tmap":
        note = (f"unthresholded $t$; outlined clusters formed at |t| > {tmin:.2f} ($p<0.01$), "
                f"none survives cluster-wise correction at $p<0.05$")
    else:
        note = f"{n_sig} vertices in clusters with corrected $p<0.05$; cluster-forming |t| > {tmin:.2f}"
    fig.text(0.5, 0.005, note, ha="center", fontsize=7, color="0.3")
    fig.savefig(args.out, dpi=300)
    print(f"wrote {args.out} ({n_sig} painted vertices, tmax {tmax:.2f})")


if __name__ == "__main__":
    main()
