"""Figure 8: vertex-wise t maps rendered on the inflated fsaverage6 surface.

Rows are Model A (age, sex, APOE4, field strength) and Model B (plus the
thickness and curvature slopes at the same vertex). Columns are left lateral,
left medial, right medial, right lateral. Colour is the unthresholded
converter-minus-non-converter t; clusters formed at the cluster-forming
threshold are outlined in black and the largest in each model is labelled.
The medial wall carries no data and is shown as sulcal shading only.

Rendered with PyVista (VTK, offscreen) for proper depth and lighting, then
composed with matplotlib.

Usage:
    python render_surface_figure.py --results results --fs_subjects_dir fs_subjects \
        --out ../../mri-bsc/paper/aperture_neuro/fig8.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyvista as pv  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from nibabel.freesurfer import read_geometry, read_label, read_morph_data  # noqa: E402

pv.OFF_SCREEN = True
VMAX = 4.0
CMAP = plt.get_cmap("RdBu_r")
# camera direction (from the brain toward the camera), RAS
VIEWS = [("lh", "lateral", (-1, 0, 0)), ("lh", "medial", (1, 0, 0)),
         ("rh", "medial", (-1, 0, 0)), ("rh", "lateral", (1, 0, 0))]
MODEL_TITLE = {"A": "Model A: age, sex, APOE4, field strength",
               "B": "Model B: Model A + thickness and curvature slopes at the vertex"}


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float64).ravel()


def vertex_rgb(t, cortex, sulc, cluster):
    """Sulcal grey everywhere, t overlay in cortex, black on cluster boundaries."""
    g = np.where(sulc > 0, 0.55, 0.78)
    g[~cortex] = 0.86                      # medial wall: no data, flat grey
    rgb = np.stack([g, g, g], axis=1)
    norm = Normalize(-VMAX, VMAX)
    over = CMAP(norm(t[cortex]))[:, :3]
    # blend over the shading so the folding still reads through
    rgb[cortex] = 0.85 * over + 0.15 * rgb[cortex]
    return rgb


def boundary_lines(coords, faces, member):
    """Edges of the mesh that separate cluster from non-cluster faces."""
    fin = member[faces].all(axis=1)
    if not fin.any():
        return None
    e = np.concatenate([faces[fin][:, [0, 1]], faces[fin][:, [1, 2]], faces[fin][:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    edges = uniq[cnt == 1]  # used by exactly one in-cluster face
    lines = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()
    return pv.PolyData(coords, lines=lines)


def render(coords, faces, rgb, outline, direction, size=(1400, 1000)):
    mesh = pv.PolyData(coords, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    mesh.point_data["rgb"] = (rgb * 255).astype(np.uint8)
    mesh = mesh.compute_normals(cell_normals=False, split_vertices=False)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True,
                ambient=0.35, diffuse=0.7, specular=0.08)
    if outline is not None:
        # nudge lines toward the camera so they are not z-fought into the surface
        outline.points = outline.points + 0.6 * np.asarray(direction, float)
        pl.add_mesh(outline, color="black", line_width=2.2)
    centre = coords.mean(axis=0)
    pl.camera.focal_point = centre
    pl.camera.position = centre + 500 * np.asarray(direction, float)
    pl.camera.up = (0, 0, 1)
    pl.enable_parallel_projection()
    pl.reset_camera()
    pl.camera.zoom(1.35)
    img = pl.screenshot(return_img=True)
    pl.close()
    # trim white margin
    nonwhite = np.where((img < 250).any(axis=2))
    r0, r1, c0, c1 = nonwhite[0].min(), nonwhite[0].max(), nonwhite[1].min(), nonwhite[1].max()
    return img[max(r0 - 10, 0):r1 + 10, max(c0 - 10, 0):c1 + 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--fs_subjects_dir", type=Path, default=Path("fs_subjects"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sdir = a.fs_subjects_dir / "fsaverage6"
    surf = {}
    for h in ("lh", "rh"):
        coords, faces = read_geometry(str(sdir / "surf" / f"{h}.inflated"))
        sulc = read_morph_data(str(sdir / "surf" / f"{h}.sulc"))
        cortex = np.zeros(len(coords), bool)
        cortex[read_label(str(sdir / "label" / f"{h}.cortex.label"))] = True
        surf[h] = (coords, faces.astype(np.int64), sulc, cortex)

    summ = {m: pd.read_csv(a.results / f"model{m}_cluster_summary.csv") for m in "AB"}

    fig = plt.figure(figsize=(12, 5.6), dpi=300)
    # header row above each image row, so titles never collide with the brains
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.06],
                          height_ratios=[0.28, 1, 0.22, 1], wspace=0.02, hspace=0.0,
                          left=0.01, right=0.93, top=0.98, bottom=0.02)
    for r, m in enumerate("AB"):
        top = summ[m].sort_values("area_mm2", ascending=False).iloc[0]
        hax = fig.add_subplot(gs[2 * r, :4])
        hax.set_axis_off()
        hax.text(0.0, 0.95, f"{'AB'[r]}   {MODEL_TITLE[m]}", fontsize=11, weight="bold",
                 ha="left", va="top", transform=hax.transAxes)
        hax.text(0.0, 0.42 if r == 0 else 0.25,
                 f"largest cluster: {top.hemi} {top.peak_aparc}, {top.area_mm2:.0f} mm²,"
                 f" corrected p = {top.cwp:.2f}  ·  no cluster survives correction",
                 fontsize=8.5, color="0.3", ha="left", va="top", transform=hax.transAxes)
        for c, (h, view, d) in enumerate(VIEWS):
            coords, faces, sulc, cortex = surf[h]
            t = load(a.results / f"py{m}_{h}" / "t.mgh")
            cid = load(a.results / f"py{m}_{h}" / "cluster_id.mgh")
            img = render(coords, faces, vertex_rgb(t, cortex, sulc, cid),
                         boundary_lines(coords, faces, cid > 0), d)
            ax = fig.add_subplot(gs[2 * r + 1, c])
            ax.imshow(img)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"{'Left' if h == 'lh' else 'Right'} {view}", fontsize=10, pad=2)

    cax = fig.add_subplot(gs[1:, 4])
    sm = plt.cm.ScalarMappable(norm=Normalize(-VMAX, VMAX), cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("t, converter minus non-converter", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=300, facecolor="white")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
