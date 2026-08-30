"""Per-vertex sigmoid fit to the cortical intensity profile (Olafson et al. 2021).

BSC is the growth-rate parameter of a sigmoid fit to ten T1w intensity samples
taken along the axis crossing the gray/white boundary:

    profile(x) = a + exp(k) - exp(k) / (1 + exp(-c * (x - d)))

where x is displacement expressed as a percentage of cortical thickness, signed
so that it increases toward the pial surface, and c is the BSC.

This is a Python port of sigmoid_unsmoothed_{left,right}.R from CoBrALab/BSC
(vendored under methods/). It reproduces the reference deliberately, including
the choices that are not stated in the paper:

  - x is in percent of cortical thickness, not millimetres, and the white side
    reuses the gray side's percentages rather than converting to a distance.
  - The fit is box-constrained, matching the reference's nls algorithm="port"
    bounds. c is capped at 1, which is a real ceiling in the data, not a
    formality: a profile that saturates within a couple of percent of thickness
    pins against it.
  - The log transform is log(c + 0.1). The offset matters because c is bounded
    below by 0 and a boundary-pinned fit would otherwise be -Inf.
  - exp(k) rather than a bare amplitude keeps the sigmoid's height positive, so
    the curve can only run bright-to-dark moving from white matter to gray.

Two departures from the reference, both deliberate:

  - The reference loops vertices 1:40963 over a table whose 40963rd column is
    the x vector itself, so its last "vertex" is a fit of x against x. Only the
    40962 real vertices are fit here.
  - The reference fits every vertex a second time on min-max rescaled
    intensities and then discards the result. That fit is not reproduced.

A second coefficient, model_c_free, is fit alongside the faithful one because
the reference's lower bound a >= min(y) is not innocuous. That bound asserts
the sigmoid reaches its gray-side asymptote within the sampled window. A sharp
boundary does; a blurred one does not, and for it the bound is binding and
wrong. The consequence is a floor: on noiseless simulated profiles, true growth
rates of 0.01, 0.02 and 0.05 all return c between 0.0587 and 0.0652, while
above roughly 0.1 the estimator is exact to five decimals. So the published BSC
cannot resolve degrees of blurring, which is the direction pathology is
hypothesised to move. model_c_free repeats the fit with a unbounded below and
recovers the true rate across the whole range; it is a sensitivity analysis,
not the primary measure.

Usage:
    python3 fit_sigmoid.py --samples_dir samples/ --image_id ADNI_002_S_0295_bl_2006-04-18 \
        --hemi left --out_dir sigmoid_fit/unsmoothed/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

# Sampling positions in percent of cortical thickness, ordered white -> pial.
# The four negative entries are the mirrored white matter surfaces; 0 is the
# gray/white boundary itself; the five positive entries are the gray surfaces.
X_PERCENT = np.array(
    [-25.0, -18.75, -12.5, -6.25, 0.0, 6.25, 12.5, 18.75, 25.0, 50.0],
    dtype=np.float64,
)

# Sample file suffixes in the same order as X_PERCENT.
SAMPLE_ORDER = [
    "WM_25", "WM_18_75", "WM_12_5", "WM_6_25",
    "WM_0",
    "GM_6_25", "GM_12_5", "GM_18_75", "GM_25", "GM_50",
]

# Index into SAMPLE_ORDER of the two surfaces forming the tissue intensity
# ratio. The reference computes yvalues[1] / yvalues[9] in R's 1-based indexing,
# which is WM_25 / GM_25 -- the inverse of the ratio described in the paper's
# Figure 1 caption. The reference behaviour is kept; invert downstream if the
# published orientation is wanted.
RATIO_NUM, RATIO_DEN = 0, 8

# Box constraints from the reference nls call, in (a, k, c, d) order. The lower
# bound on a is per-vertex min(y) and is filled in at fit time.
LOWER = np.array([np.nan, 0.0, 0.0, -50.0])
UPPER = np.array([2000.0, 100.0, 1.0, 50.0])
# Lower bound on a for the sensitivity variant. Finite rather than -inf so trf
# keeps a bounded problem; far below any plausible T1 intensity.
A_LOWER_FREE = -1e6
START_K, START_C, START_D = 5.0, 0.1, 0.0

LOG_OFFSET = 0.1
N_VERTICES_DEFAULT = 40962
MAX_NFEV = 50


def sigmoid(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the profile model at x for parameters (a, k, c, d)."""
    a, k, c, d = params
    amp = np.exp(k)
    s = 1.0 / (1.0 + np.exp(-c * (x - d)))
    return a + amp - amp * s


def _residual(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return sigmoid(params, x) - y


def _jacobian(params: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytic Jacobian. Supplying it roughly halves the per-vertex cost, which
    matters at ~82k fits per scan."""
    a, k, c, d = params
    amp = np.exp(k)
    s = 1.0 / (1.0 + np.exp(-c * (x - d)))
    ds = s * (1.0 - s)

    j = np.empty((x.size, 4), dtype=np.float64)
    j[:, 0] = 1.0
    j[:, 1] = amp * (1.0 - s)
    j[:, 2] = -amp * ds * (x - d)
    j[:, 3] = amp * c * ds
    return j


def _run_fit(y: np.ndarray, x: np.ndarray, a_lower: float) -> tuple[float, float, bool]:
    lower = LOWER.copy()
    lower[0] = a_lower
    x0 = np.array([float(np.min(y)), START_K, START_C, START_D], dtype=np.float64)
    x0 = np.clip(x0, lower, UPPER)

    try:
        res = least_squares(
            _residual, x0, jac=_jacobian, bounds=(lower, UPPER),
            method="trf", xtol=1e-5, ftol=1e-5, max_nfev=MAX_NFEV,
            args=(x, y),
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, False

    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    # warnOnly=TRUE in the reference keeps non-converged fits. The coefficient is
    # kept here too, but the flag is carried so the convergence map the paper
    # reports as Supplementary Figure S6 can be reproduced and so downstream
    # code can drop them if it wants to.
    return float(res.x[2]), rmse, bool(res.success)


def fit_one_vertex(y: np.ndarray,
                   x: np.ndarray = X_PERCENT) -> tuple[float, float, float, bool]:
    """Fit one intensity profile.

    Returns (c, c_free, rmse, converged), where c uses the reference's
    a >= min(y) bound and c_free leaves a unbounded below. A vertex whose
    samples are all equal carries no boundary at all and is returned as NaN
    rather than fit, since the optimiser would otherwise return the c start
    value unchanged and that would read downstream as a real measurement.
    """
    y_min = float(np.min(y))
    if not np.all(np.isfinite(y)) or float(np.max(y)) - y_min <= 0.0:
        return np.nan, np.nan, np.nan, False

    c, rmse, converged = _run_fit(y, x, y_min)
    c_free, _, _ = _run_fit(y, x, A_LOWER_FREE)
    return c, c_free, rmse, converged


def load_samples(samples_dir: Path, image_id: str, hemi: str,
                 n_vertices: int) -> np.ndarray:
    """Read the ten volume_object_evaluate outputs into a (10, n_vertices) array."""
    profiles = np.empty((len(SAMPLE_ORDER), n_vertices), dtype=np.float64)
    for i, frac in enumerate(SAMPLE_ORDER):
        path = samples_dir / f"{image_id}_{frac}_{hemi}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing sample file: {path}")
        vals = np.loadtxt(path, dtype=np.float64)
        if vals.size != n_vertices:
            raise ValueError(
                f"{path} has {vals.size} vertices, expected {n_vertices}"
            )
        profiles[i] = vals
    return profiles


def fit_hemisphere(profiles: np.ndarray) -> dict[str, np.ndarray]:
    """Fit every vertex of one hemisphere.

    profiles is (10, n_vertices), ordered as SAMPLE_ORDER.
    """
    n_vertices = profiles.shape[1]
    c = np.empty(n_vertices, dtype=np.float64)
    c_free = np.empty(n_vertices, dtype=np.float64)
    rmse = np.empty(n_vertices, dtype=np.float64)
    converged = np.zeros(n_vertices, dtype=bool)

    for v in range(n_vertices):
        c[v], c_free[v], rmse[v], converged[v] = fit_one_vertex(profiles[:, v])

    with np.errstate(invalid="ignore"):
        c_log = np.log(c + LOG_OFFSET)
        c_free_log = np.log(c_free + LOG_OFFSET)

    den = profiles[RATIO_DEN]
    ratio = np.divide(
        profiles[RATIO_NUM], den,
        out=np.full(n_vertices, np.nan), where=den != 0,
    )

    return {"model_c": c, "model_c_log": c_log,
            "model_c_free": c_free, "model_c_free_log": c_free_log,
            "model_ratio": ratio,
            "rmse": rmse, "converged": converged.astype(np.int8)}


def write_outputs(out_dir: Path, image_id: str, hemi: str,
                  results: dict[str, np.ndarray]) -> None:
    """Write one plain-text column per quantity, the format depth_potential and
    surface-resample expect."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in results.items():
        fmt = "%d" if arr.dtype == np.int8 else "%.10g"
        np.savetxt(out_dir / f"{image_id}_{name}_{hemi}.txt", arr, fmt=fmt)


def self_check() -> None:
    """Recover known growth rates from noiseless simulated profiles.

    Demonstrates the floor the reference's a >= min(y) bound imposes on blurred
    boundaries, and confirms the free variant does not have it.
    """
    print("Parameter recovery on noiseless profiles (a=30, amplitude=70, d=0)\n")
    print(f"{'true c':>8} | {'reference':>10} | {'free a':>10} | {'rmse':>9}")
    print("-" * 46)
    for c_true in (0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.9):
        y = sigmoid(np.array([30.0, np.log(70.0), c_true, 0.0]), X_PERCENT)
        c, c_free, rmse, _ = fit_one_vertex(y)
        print(f"{c_true:8.3f} | {c:10.5f} | {c_free:10.5f} | {rmse:9.2e}")
    print("\nThe reference estimator floors near 0.059: true rates of 0.01, "
          "0.02 and 0.05\nare not distinguishable from one another.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="run the parameter-recovery self check and exit")
    ap.add_argument("--samples_dir")
    ap.add_argument("--image_id")
    ap.add_argument("--hemi", choices=["left", "right"])
    ap.add_argument("--out_dir")
    ap.add_argument("--n_vertices", type=int, default=N_VERTICES_DEFAULT)
    args = ap.parse_args()

    if args.check:
        self_check()
        return

    missing = [f"--{n}" for n in ("samples_dir", "image_id", "hemi", "out_dir")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"missing required arguments: {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    marker = out_dir / f"{args.image_id}_model_c_{args.hemi}.txt"
    if marker.exists():
        print(f"[SKIP] {args.image_id} {args.hemi} already fit")
        return

    profiles = load_samples(Path(args.samples_dir), args.image_id, args.hemi,
                            args.n_vertices)
    results = fit_hemisphere(profiles)
    write_outputs(out_dir, args.image_id, args.hemi, results)

    n_conv = int(results["converged"].sum())
    n_valid = int(np.isfinite(results["model_c"]).sum())
    print(f"[OK] {args.image_id} {args.hemi}: "
          f"{n_valid}/{args.n_vertices} fit, {n_conv} converged, "
          f"median c={np.nanmedian(results['model_c']):.4f}")


if __name__ == "__main__":
    main()
