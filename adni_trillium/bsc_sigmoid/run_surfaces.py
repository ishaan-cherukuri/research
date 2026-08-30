"""Generate the ten sampling surfaces for one scan and sample T1w intensity on them.

Port of generate_gray_surfaces.sh, generate_white_surfaces.sh and
sample_surfaces.sh from CoBrALab/BSC (vendored under methods/). The reference
scripts emit qbatch joblists across the whole cohort; this does the same work
one scan at a time so it can be fanned out with GNU Parallel the way the
existing slurm/run_bsc_pipeline.sh does, and so a rerun skips finished scans.

Surface construction follows the reference exactly. Gray surfaces are built by
repeated midpoint averaging rather than by displacing along a normal:

    GM_25    = midpoint(WM_0,     GM_50)     <- GM_50 is CIVET's mid surface
    GM_12_5  = midpoint(WM_0,     GM_25)
    GM_6_25  = midpoint(WM_0,     GM_12_5)
    GM_18_75 = midpoint(GM_12_5,  GM_25)

White surfaces are the gray ones reflected through the boundary, which is what
move_surface_along_flipped_vector does: it displaces each WM_0 vertex by the
negation of the vector to the matching gray vertex. So the white samples sit at
the same percentages of cortical thickness on the other side, measured
per vertex, and inherit the gray surfaces' vertex correspondence.

Usage:
    python3 run_surfaces.py --civet_root $SCRATCH/civet_out --work_root $SCRATCH/bsc_sigmoid \
        --civet_id scan000123 --prefix adni
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# (output fraction, first parent, second parent); order matters, each step
# consumes the previous one's output.
GRAY_STEPS = [
    ("GM_25", "WM_0", "GM_50"),
    ("GM_12_5", "WM_0", "GM_25"),
    ("GM_6_25", "WM_0", "GM_12_5"),
    ("GM_18_75", "GM_12_5", "GM_25"),
]

WHITE_FRACTIONS = ["WM_6_25", "WM_12_5", "WM_18_75", "WM_25"]

ALL_FRACTIONS = [
    "WM_25", "WM_18_75", "WM_12_5", "WM_6_25", "WM_0",
    "GM_6_25", "GM_12_5", "GM_18_75", "GM_25", "GM_50",
]

HEMIS = ["left", "right"]

MOVE_BIN = Path(__file__).resolve().parent / "methods" / "move_surface_along_flipped_vector"


def run(cmd: list[str]) -> None:
    subprocess.run([str(c) for c in cmd], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def surf(work: Path, frac: str, cid: str, hemi: str) -> Path:
    return work / "surfaces" / f"{frac}_surfaces" / f"{cid}_{frac}_surface_{hemi}.obj"


def stage_civet_surfaces(civet_root: Path, work: Path, cid: str,
                         prefix: str, hemi: str) -> None:
    """Point WM_0 and GM_50 at CIVET's white and mid surfaces under the names the
    rest of the pipeline expects. Symlinked rather than copied: CIVET output for
    the full cohort is already several terabytes."""
    pairs = [
        ("WM_0", civet_root / cid / "surfaces"
         / f"{prefix}_{cid}_white_surface_{hemi}_81920.obj"),
        ("GM_50", civet_root / cid / "surfaces"
         / f"{prefix}_{cid}_mid_surface_{hemi}_81920.obj"),
    ]
    for frac, src in pairs:
        if not src.exists():
            raise FileNotFoundError(f"CIVET output missing: {src}")
        dst = surf(work, frac, cid, hemi)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src)


def build_surfaces(work: Path, cid: str, hemi: str) -> None:
    for frac, p1, p2 in GRAY_STEPS:
        out = surf(work, frac, cid, hemi)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        # average_surfaces <out> <none: avg surface> <none: rms> <n> <inputs...>
        run(["average_surfaces", out, "none", "none", "1",
             surf(work, p1, cid, hemi), surf(work, p2, cid, hemi)])

    for frac in WHITE_FRACTIONS:
        out = surf(work, frac, cid, hemi)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            continue
        gray = frac.replace("WM_", "GM_")
        run([MOVE_BIN, surf(work, "WM_0", cid, hemi),
             surf(work, gray, cid, hemi), out])


def sample_surfaces(work: Path, civet_root: Path, cid: str, prefix: str,
                    hemi: str) -> None:
    """Sample the stereotaxic T1 at every vertex of all ten surfaces."""
    t1 = civet_root / cid / "final" / f"{prefix}_{cid}_t1_tal.mnc"
    if not t1.exists():
        raise FileNotFoundError(f"CIVET stereotaxic T1 missing: {t1}")

    out_dir = work / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for frac in ALL_FRACTIONS:
        out = out_dir / f"{cid}_{frac}_{hemi}.txt"
        if out.exists():
            continue
        run(["volume_object_evaluate", t1, surf(work, frac, cid, hemi), out])


def is_done(work: Path, cid: str) -> bool:
    return all((work / "samples" / f"{cid}_{frac}_{hemi}.txt").exists()
               for frac in ALL_FRACTIONS for hemi in HEMIS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--civet_root", required=True)
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--civet_id", required=True)
    ap.add_argument("--prefix", default="adni")
    args = ap.parse_args()

    work = Path(args.work_root)
    civet_root = Path(args.civet_root)
    cid = args.civet_id

    if is_done(work, cid):
        print(f"[SKIP] {cid} already sampled")
        return

    for hemi in HEMIS:
        stage_civet_surfaces(civet_root, work, cid, args.prefix, hemi)
        build_surfaces(work, cid, hemi)
        sample_surfaces(work, civet_root, cid, args.prefix, hemi)

    print(f"[OK] {cid} sampled")


if __name__ == "__main__":
    main()
