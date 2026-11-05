"""Algorithm C: broadband MUSIC on the small triangle M1-M3 (directions only, speaker count from MDL).

Specification for the firmware port (daq-detector/main/music.c). Decisions (02.11.2026): separate algorithm
(like A and B), azimuth only (the M1-M3 triangle is planar: +el and -el are identical and the elevation is weak),
covariance averaged over ~16 frames (0.5 s).

`Music.process(r)` is called for EVERY frame (r = tdoa.FrameResult; the state also decays on frames without voice)
and returns a result only for voice frames:

1. Bins KC_LO..KC_HI (300-2800 Hz, below the 2.9 kHz phase-ambiguity limit of the 60 mm triangle); N_BANDS
   narrow bands of ONE bin each, spread evenly: bin_b = KC_LO + floor((b (n - 1) + (N_BANDS - 1) / 2) / (N_BANDS - 1)),
   n = KC_HI - KC_LO + 1 (integer division). One bin per band: grouping neighbouring bins smears the steering vector
   (rank > 1) and made MDL report 2 sources for a single source in free field.
2. Snapshot of a bin: x = (X_1[k], X_2[k], X_3[k]) normalised to |x| = 1 (|x|^2 > 1e-20, otherwise skipped), so
   every bin and frame weighs the same (spectral whitening, as PHAT).
3. Covariance per band, exponential average: every frame R_b *= LAMBDA, w *= LAMBDA (LAMBDA = 1 - 1 / AVG_FRAMES);
   on a voice frame R_b += sum over the band of x x^H, w += 1.
4. Only on voice frames with w >= MIN_WEIGHT: eigenvalues l1 >= l2 >= l3 and eigenvectors of every R_b
   (Hermitian 3 x 3; scale does not matter).
5. Speaker count K in {1, 2} (at most M - 1 = 2; at least 1 because the frame has voice): K = argmin over k of the
   MDL summed over bands, MDL_b(k) = -N_b (M - k) ln(g / a) + MDL_PENALTY * 0.5 k (2M - k) ln N_b, g / a = geometric /
   arithmetic mean of the M - k smallest eigenvalues (each clamped to >= 1e-12 l1), N_b = w (snapshots: one bin per
   band, one per voice frame, exponentially weighted). MDL_PENALTY = 1 is the standard MDL; with only ~16 snapshots it
   tends to report 2 (see tools/evaluate.py for the trade-off).
6. Pseudospectrum on the azimuth grid 0, 1, ..., 359 deg (elevation 0): u = (cos az, sin az, 0),
   a_m = exp(j 2 pi f_b (u . p_m) / c), f_b = mean frequency of the band's bins (spectra use e^{-j}, so a mic closer
   to the source leads in phase), En = eigenvectors of the M - K smallest eigenvalues,
   d_b(az) = |En^H a|^2 / M in [0, 1], P(az) = mean over bands of 1 / max(d_b, 1e-6).
7. Peaks: local maxima of P on the circular grid (P[i] > P[i-1] and P[i] >= P[i+1]), the K largest, strongest first;
   parabolic refinement az = i + 0.5 (P[i-1] - P[i+1]) / (P[i-1] - 2 P[i] + P[i+1]) (mod 360).
8. Source: azimuth, elevation 0, distance 0 (kind D), confidence = 1 - mean over bands of d_b(az), clipped to 0..1.
"""
from dataclasses import dataclass

import numpy as np

from .tdoa import FRAME_LEN, band_bins

BAND_HZ = (300.0, 2800.0)
N_BANDS = 40
AVG_FRAMES = 16
LAMBDA = 1.0 - 1.0 / AVG_FRAMES
MIN_WEIGHT = 4.0
MDL_PENALTY = 1.0
MAX_SOURCES = 2
N_MICS = 3
AZ_GRID = np.arange(360, dtype=float)


@dataclass
class MusicSource:
    az_deg: float
    el_deg: float
    dist_m: float
    confidence: float
    kind: str = "D"


@dataclass
class MusicResult:
    sources: list            # MusicSource, strongest first
    count: int               # MDL estimate (0..2)
    spectrum: np.ndarray     # (360,) pseudospectrum


class Music:
    def __init__(self, geometry):
        self.g = geometry
        k_lo, _ = band_bins(geometry.fs)
        self.kc_lo = int(np.ceil(BAND_HZ[0] * FRAME_LEN / geometry.fs))
        self.kc_hi = int(np.floor(BAND_HZ[1] * FRAME_LEN / geometry.fs))
        n = self.kc_hi - self.kc_lo + 1
        self.bands = [np.array([self.kc_lo + (b * (n - 1) + (N_BANDS - 1) // 2) // (N_BANDS - 1)]) for b in range(N_BANDS)]
        self.offset = self.kc_lo - k_lo                           # index of KC_LO in the 300-3500 Hz band arrays
        self.freq = np.array([np.mean(b) * geometry.fs / FRAME_LEN for b in self.bands])
        self.R = np.zeros((N_BANDS, N_MICS, N_MICS), dtype=complex)
        self.w = 0.0
        p = geometry.mics_m[:N_MICS]                               # (3, 3)
        u = np.stack([np.cos(np.radians(AZ_GRID)), np.sin(np.radians(AZ_GRID)), np.zeros_like(AZ_GRID)], axis=1)
        self.proj = u @ p.T                                        # (360, 3) u . p_m in metres

    def _steer(self, b, proj):
        return np.exp(2j * np.pi * self.freq[b] * proj / self.g.c)   # (..., 3)

    def _den(self, En, b, proj):
        a = self._steer(b, proj)
        return np.sum(np.abs(a @ np.conj(En)) ** 2, axis=-1) / N_MICS

    def process(self, r):
        self.R *= LAMBDA
        self.w *= LAMBDA
        if not r.voice:
            return None
        for b, bins in enumerate(self.bands):
            x = r.spec[:N_MICS, bins - self.kc_lo + self.offset]     # (3, nb)
            e = np.sum(np.abs(x) ** 2, axis=0)
            keep = e > 1e-20
            x = x[:, keep] / np.sqrt(e[keep])
            self.R[b] += x @ np.conj(x.T)
        self.w += 1.0
        if self.w < MIN_WEIGHT:
            return MusicResult([], 0, np.zeros(len(AZ_GRID)))

        evals, evecs = np.linalg.eigh(self.R)                       # ascending
        mdl = np.full(MAX_SOURCES + 1, np.inf)
        mdl[1:] = 0.0
        for b, bins in enumerate(self.bands):
            lam = evals[b][::-1]                                    # descending
            lam = np.maximum(lam, 1e-12 * lam[0])
            nb = len(bins) * self.w
            for k in range(1, MAX_SOURCES + 1):
                tail = lam[k:]
                ratio = np.exp(np.mean(np.log(tail))) / np.mean(tail)
                mdl[k] += -nb * (N_MICS - k) * np.log(ratio) + MDL_PENALTY * 0.5 * k * (2 * N_MICS - k) * np.log(nb)
        count = int(np.argmin(mdl))           # first minimum wins on a tie (k = 1)

        En = evecs[:, :, :N_MICS - count]                           # smallest eigenvalues
        den = np.stack([self._den(En[b], b, self.proj) for b in range(N_BANDS)])   # (bands, 360)
        spec = np.mean(1.0 / np.maximum(den, 1e-6), axis=0)

        n = len(AZ_GRID)
        peaks = [i for i in range(n) if spec[i] > spec[i - 1] and spec[i] >= spec[(i + 1) % n]]
        peaks.sort(key=lambda i: -spec[i])
        sources = []
        p = self.g.mics_m[:N_MICS]
        for i in peaks[:count]:
            ym, y0, yp = spec[i - 1], spec[i], spec[(i + 1) % n]
            d = ym - 2 * y0 + yp
            delta = 0.5 * (ym - yp) / d if d != 0 else 0.0
            az = float((AZ_GRID[i] + delta) % 360.0)
            u = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), 0.0])
            dmean = float(np.mean([self._den(En[b], b, p @ u) for b in range(N_BANDS)]))
            sources.append(MusicSource(az, 0.0, 0.0, min(max(1.0 - dmean, 0.0), 1.0)))
        return MusicResult(sources, count, spec)
