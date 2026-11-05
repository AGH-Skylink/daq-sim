"""Algorithm B: SRP-PHAT (steered response power) on the GCC-PHAT correlations of the 15 pairs.

Specification for the firmware port (daq-detector/main/localize.c), step by step:

1. Power of a point p: P(p) = sum over pairs of r_ij(tau_ij(p)), tau_ij(p) = (|p - m_i| - |p - m_j|) / c * fs
   in samples; r_ij is the GCC-PHAT correlation at integer lags -(L+1)..L+1 (tdoa.py), read with linear
   interpolation and clamped to the stored range.
2. Coarse grid = the BLE map grid: azimuth 0, 10, ..., 350 deg (36), elevation -85, -75, ..., 85 deg (18),
   distances DISTANCES_M (8). Map cell (az, el) = max over distances. The BLE map is this, scaled so that
   the maximum is 255 (negative values -> 0).
3. Peaks: local maxima of the coarse map (8 neighbours, azimuth wraps, no wrap in elevation), strongest first.
   Source 1 = strongest, accepted if P / 15 >= MIN_POWER. Source 2 = the strongest other local maximum at
   least MIN_SEPARATION_DEG away from source 1 with P >= SECOND_RATIO * P1 (and >= MIN_POWER * 15).
   Speaker count = number of accepted sources (0..2).
4. Refinement of each accepted source on the exact correlation r_ij(tau) = Re(sum_band W[k] e^{j 2 pi k tau / N}) / n_band:
   a. start distance = the best of DISTANCES_M at the coarse direction (exact power),
   b. pattern search with steps 5 deg (az), 5 deg (el), factor 1.25 (distance): evaluate the 26 neighbours
      ({-1, 0, +1} steps, order az-major, first one wins on ties); if one is better, move there and keep the steps,
      otherwise halve the angle steps and take the square root of the distance factor; stop when the angle step
      is below REFINE_MIN_STEP_DEG. Distance clamped to the DISTANCES_M range, elevation to +-90.
   The SRP power is very flat in distance, so the final position comes from step 5.
5. SRP-guided time differences + least squares: for every pair take the largest local value of r_ij within
   +-GUIDE_WINDOW samples of the delay predicted by the refined point (integer lags, parabolic refinement as in
   tdoa.py), then run the least-squares fit of algorithm A (ls.locate) with its linear start; if that fit is not ok,
   once more starting from the refined SRP point.
   Decision e: if the fit is ok the source is a position (kind P); otherwise it is the SRP direction only
   (distance 0, kind D).
6. Confidence = P / 15, clipped to 0..1.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import PAIRS, from_spherical, to_spherical
from .ls import confidence, locate
from .tdoa import FRAME_LEN, band_bins

AZ_STEP_DEG = 10.0
N_AZ = 36
N_EL = 18
DISTANCES_M = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIN_POWER = 0.15
SECOND_RATIO = 0.8
MIN_SEPARATION_DEG = 40.0
REFINE_MIN_STEP_DEG = 1.0
REFINE_MAX_ROUNDS = 40
GUIDE_WINDOW = 3          # samples (+-62 us)


def grid_el(j):
    return -85.0 + 10.0 * j


@dataclass
class SrpSource:
    az_deg: float
    el_deg: float
    dist_m: float          # 0 = unknown
    power: float           # P / 15
    confidence: float      # P / 15 for a direction; confidence of the least-squares fit for a position
    kind: str = "D"        # P = position, D = direction only


@dataclass
class SrpResult:
    sources: list          # SrpSource, strongest first
    map: np.ndarray        # (N_EL, N_AZ) uint8, row 0 = el -85


def angular_distance_deg(az1, el1, az2, el2):
    a1, e1, a2, e2 = np.radians([az1, el1, az2, el2])
    cosd = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return float(np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))))


class SrpPhat:
    def __init__(self, geometry, max_lags):
        self.g = geometry
        self.max_lags = max_lags
        k_lo, k_hi = band_bins(geometry.fs)
        self.k = np.arange(k_lo, k_hi + 1)
        az = np.arange(N_AZ) * AZ_STEP_DEG
        el = np.array([grid_el(j) for j in range(N_EL)])
        pts = np.array([[from_spherical(a, e, d) for d in DISTANCES_M] for e in el for a in az])   # (N_EL*N_AZ, ND, 3)
        self.coarse_delays = self._delays(pts.reshape(-1, 3)).reshape(N_EL, N_AZ, len(DISTANCES_M), len(PAIRS))

    def _delays(self, pts):
        """(n, 3) points -> (n, 15) delays in samples."""
        d = np.linalg.norm(pts[:, None, :] - self.g.mics_m[None, :, :], axis=2)
        return np.stack([(d[:, i] - d[:, j]) for i, j in PAIRS], axis=1) / self.g.c * self.g.fs

    def _power(self, corr, delays):
        """delays (..., 15) samples -> power (...)."""
        total = np.zeros(delays.shape[:-1])
        for p in range(len(PAIRS)):
            lag = self.max_lags[p]
            x = np.clip(delays[..., p] + lag + 1, 0, 2 * lag + 2)
            i0 = np.minimum(np.floor(x).astype(int), 2 * lag + 1)
            frac = x - i0
            r = corr[p]
            total += r[i0] * (1 - frac) + r[i0 + 1] * frac
        return total

    def power_at(self, cross, az, el, dist):
        """Exact power at one point: sum over pairs of Re(sum_k W[k] e^{j 2 pi k tau / N}) / n_band."""
        tau = self._delays(from_spherical(az, el, dist)[None, :])[0]
        total = 0.0
        for p in range(len(PAIRS)):
            total += float(np.real(np.sum(cross[p] * np.exp(2j * np.pi * self.k * tau[p] / FRAME_LEN)))) / len(self.k)
        return total

    def _refine(self, cross, az, el):
        lo, hi = DISTANCES_M[0], DISTANCES_M[-1]
        best, dist = max((self.power_at(cross, az, el, d), d) for d in DISTANCES_M)
        step_a, step_e, fac = 5.0, 5.0, 1.25
        for _ in range(REFINE_MAX_ROUNDS):
            cand = None
            for da in (-1, 0, 1):
                for de in (-1, 0, 1):
                    for dd in (-1, 0, 1):
                        if da == de == dd == 0:
                            continue
                        a = (az + da * step_a) % 360.0
                        e = min(max(el + de * step_e, -90.0), 90.0)
                        d = min(max(dist * fac ** dd, lo), hi)
                        p = self.power_at(cross, a, e, d)
                        if p > best:
                            best, cand = p, (a, e, d)
            if cand:
                az, el, dist = cand
            else:
                step_a, step_e, fac = step_a / 2, step_e / 2, fac ** 0.5
                if step_a < REFINE_MIN_STEP_DEG:
                    break
        return az, el, dist, best

    def guided_tdoa(self, corr, az, el, dist):
        """Time differences (s) and peaks: the best local value of each r_ij near the predicted delay."""
        pred = self._delays(from_spherical(az, el, dist)[None, :])[0]
        tau = np.zeros(len(PAIRS))
        peak = np.zeros(len(PAIRS))
        for p in range(len(PAIRS)):
            lag, r = self.max_lags[p], corr[p]
            c = int(round(pred[p]))
            lo = max(c - GUIDE_WINDOW, -lag) + lag + 1
            hi = min(c + GUIDE_WINDOW, lag) + lag + 1
            if lo > hi:
                lo = hi = min(max(c, -lag), lag) + lag + 1
            m = lo + int(np.argmax(r[lo:hi + 1]))
            y0, ym, yp = r[m], r[m - 1], r[m + 1]
            den = ym - 2 * y0 + yp
            delta = 0.5 * (ym - yp) / den if den != 0 else 0.0
            tau[p] = (m - lag - 1 + delta) / self.g.fs
            peak[p] = y0 - 0.25 * (ym - yp) * delta
        return tau, peak

    def process(self, corr, cross):
        power = self._power(corr, self.coarse_delays)          # (N_EL, N_AZ, ND)
        dmap = np.max(power, axis=2)                            # (N_EL, N_AZ)

        peaks = []
        for j in range(N_EL):
            for k in range(N_AZ):
                v = dmap[j, k]
                neigh = [dmap[jj, (k + dk) % N_AZ] for jj in (j - 1, j, j + 1) if 0 <= jj < N_EL
                         for dk in (-1, 0, 1) if (jj, dk) != (j, 0)]
                if all(v > n for n in neigh):
                    peaks.append((v, j, k))
        peaks.sort(key=lambda t: -t[0])

        accepted = []
        if peaks and peaks[0][0] / len(PAIRS) >= MIN_POWER:
            accepted.append(peaks[0])
            for cand in peaks[1:]:
                v, j, k = cand
                p1, j1, k1 = peaks[0]
                if (v >= SECOND_RATIO * p1 and v / len(PAIRS) >= MIN_POWER and
                        angular_distance_deg(k * AZ_STEP_DEG, grid_el(j), k1 * AZ_STEP_DEG, grid_el(j1)) >= MIN_SEPARATION_DEG):
                    accepted.append(cand)
                    break

        sources = []
        for v, j, k in accepted:
            az, el, dist, p = self._refine(cross, k * AZ_STEP_DEG, grid_el(j))
            pw = p / len(PAIRS)
            tau, peak = self.guided_tdoa(corr, az, el, dist)
            fit = locate(self.g, tau)                                   # linear start, as algorithm A
            if not fit.ok:
                fit = locate(self.g, tau, start=from_spherical(az, el, dist))
            if fit.ok:
                a2, e2, d2 = to_spherical(fit.position_m)
                sources.append(SrpSource(a2, e2, d2, pw, confidence(fit, peak), "P"))
            else:
                sources.append(SrpSource(az, el, 0.0, pw, min(max(pw, 0.0), 1.0), "D"))

        top = dmap.max()
        m = np.zeros_like(dmap) if top <= 0 else np.clip(dmap, 0, None) / top * 255.0
        return SrpResult(sources, np.floor(m + 0.5).astype(np.uint8))
