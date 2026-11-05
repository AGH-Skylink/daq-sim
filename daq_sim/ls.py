"""Algorithm A, step 2: source position from the 15 time differences (least squares).

1. Linear start (reference mic M1): for i = 2..6 with r_i = c * tau_i1,
   2 (m_i - m_1) . s + 2 r_i d_1 = |m_i|^2 - |m_1|^2 - r_i^2, unknowns s (3) and d_1 = |s - m_1|.
2. Gauss-Newton on all used pairs: minimise sum (tau_ij - (|s - m_i| - |s - m_j|) / c)^2.
3. Pair rejection: while the worst residual exceeds REJECT_US and at most MAX_REJECT pairs are dropped,
   drop it and refit (6 mics give 2 redundant equations).
4. The fit is ok when it is finite, its RMS residual is <= REJECT_US and the position is within MAX_RANGE_M.

Decision e (direction only when the fit is poor): if the position fit fails, `direction()` fits a plane
wave instead, tau_ij = (m_j - m_i) . v (v = u / c, 3 unknowns, linear least squares with the same pair
rejection); the direction is accepted when its RMS residual is <= REJECT_US and |v| c is within
SLOWNESS_TOL of 1. The result then has no distance.
"""
from dataclasses import dataclass, field

import numpy as np

from .geometry import PAIRS

REJECT_US = 25.0
MAX_REJECT = 2
GN_ITERATIONS = 10
SLOWNESS_TOL = 0.3
MAX_RANGE_M = 8.0         # a position farther than this is a diverged fit (plane-wave-like data), not a result


@dataclass
class LsResult:
    position_m: np.ndarray
    rms_residual_us: float
    used: list = field(default_factory=list)   # indices into PAIRS
    ok: bool = True


def _linear_start(g, tau):
    m = g.mics_m
    rows, rhs = [], []
    for i in range(1, 6):
        p = PAIRS.index((0, i))
        r = -g.c * tau[p]                   # tau_1i = t_1 - t_i  ->  r_i = d_i - d_1 = -c tau_1i
        rows.append(np.concatenate([2 * (m[i] - m[0]), [2 * r]]))
        rhs.append(m[i] @ m[i] - m[0] @ m[0] - r * r)
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return sol[:3]


def _gauss_newton(g, tau, used, s):
    m = g.mics_m
    for _ in range(GN_ITERATIONS):
        d = np.linalg.norm(s - m, axis=1)
        d = np.maximum(d, 1e-6)
        u = (s - m) / d[:, None]            # gradient of |s - m_k|
        res = np.array([tau[p] - (d[PAIRS[p][0]] - d[PAIRS[p][1]]) / g.c for p in used])
        jac = np.array([(u[PAIRS[p][0]] - u[PAIRS[p][1]]) / g.c for p in used])
        step, *_ = np.linalg.lstsq(jac, res, rcond=None)
        s = s + step
        if np.linalg.norm(step) < 1e-5:
            break
    d = np.linalg.norm(s - m, axis=1)
    res = np.array([tau[p] - (d[PAIRS[p][0]] - d[PAIRS[p][1]]) / g.c for p in used])
    return s, res


def locate(g, tau, start=None):
    """start: initial position (m); None = linear start from M1."""
    used = list(range(len(PAIRS)))
    if start is not None:
        s = np.asarray(start, dtype=float)
    else:
        try:
            s = _linear_start(g, tau)
        except np.linalg.LinAlgError:
            return LsResult(np.zeros(3), float("inf"), [], ok=False)
    for _ in range(MAX_REJECT + 1):
        s, res = _gauss_newton(g, tau, used, s)
        worst = int(np.argmax(np.abs(res)))
        if abs(res[worst]) * 1e6 <= REJECT_US or len(used) <= len(PAIRS) - MAX_REJECT:
            break
        used.pop(worst)
    rms = float(np.sqrt(np.mean(res ** 2)) * 1e6)
    ok = bool(np.all(np.isfinite(s))) and rms <= REJECT_US and float(np.linalg.norm(s)) <= MAX_RANGE_M
    return LsResult(s, rms, used, ok=ok)


def confidence(result, peak):
    """Mean GCC-PHAT peak of the used pairs x (1 - rms / REJECT_US), clipped to 0..1 (same as localize.c)."""
    if not result.used:
        return 0.0
    c = float(np.mean(np.asarray(peak)[result.used])) * (1.0 - result.rms_residual_us / REJECT_US)
    return min(max(c, 0.0), 1.0)


@dataclass
class DirResult:
    az_deg: float
    el_deg: float
    rms_residual_us: float
    used: list = field(default_factory=list)
    ok: bool = True


def direction(g, tau):
    """Plane-wave fit (far field): direction only. Same pair rejection as locate()."""
    m = g.mics_m
    rows = np.array([m[j] - m[i] for i, j in PAIRS])
    tau = np.asarray(tau, dtype=float)
    used = list(range(len(PAIRS)))
    for _ in range(MAX_REJECT + 1):
        v, *_ = np.linalg.lstsq(rows[used], tau[used], rcond=None)
        res = tau[used] - rows[used] @ v
        worst = int(np.argmax(np.abs(res)))
        if abs(res[worst]) * 1e6 <= REJECT_US or len(used) <= len(PAIRS) - MAX_REJECT:
            break
        used.pop(worst)
    rms = float(np.sqrt(np.mean(res ** 2)) * 1e6)
    speed_ratio = float(np.linalg.norm(v) * g.c)
    az = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    el = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    ok = rms <= REJECT_US and abs(speed_ratio - 1.0) <= SLOWNESS_TOL
    return DirResult(az, el, rms, used, ok)
