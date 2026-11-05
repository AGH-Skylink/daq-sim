"""Reference front end: framing, voice activity detection and GCC-PHAT time differences.

This is the specification for `daq-detector/main/dsp.c`. Every step is defined so that the firmware
can reproduce it exactly (float32 on the device vs float64 here, expected difference << 1 us):

- frames of N = 2048 samples, hop 1536 (32 ms at 48 kHz), frame k starts at sample k * hop,
- samples scaled by 1/32768, periodic Hann window w[n] = 0.5 - 0.5 cos(2 pi n / N),
- spectra X_c[k] = sum_n x[n] w[n] e^{-j 2 pi k n / N}, only bins K_LO..K_HI (300-3500 Hz) are used,
- VAD: E = mean over channels of sum_{band} |X_c[k]|^2, energy_db = 10 log10(E + 1e-20);
  floor starts at the first frame's energy, then floor = min(energy_db, floor + 0.05);
  voice = energy_db > floor + 9 dB,
- GCC-PHAT per pair (i, j): W[k] = X_i[k] conj(X_j[k]) / (|X_i[k] X_j[k]| + 1e-20),
  r(n) = Re(sum_band W[k] e^{j 2 pi k n / N}) / n_band, for integer lags |n| <= L_ij + 1,
  L_ij = ceil(d_ij / c * fs) + 2; peak m = argmax over |n| <= L_ij;
  parabolic refinement delta = 0.5 (r[m-1] - r[m+1]) / (r[m-1] - 2 r[m] + r[m+1]),
  tau_ij = (m + delta) / fs (positive when the sound reaches mic i later than mic j),
  peak = r[m] - 0.25 (r[m-1] - r[m+1]) delta.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import PAIRS

FRAME_LEN = 2048
HOP = 1536
BAND_HZ = (300.0, 3500.0)
VAD_MARGIN_DB = 9.0
VAD_FLOOR_RISE_DB = 0.05


def band_bins(fs, n=FRAME_LEN):
    k_lo = int(np.ceil(BAND_HZ[0] * n / fs))
    k_hi = int(np.floor(BAND_HZ[1] * n / fs))
    return k_lo, k_hi


def hann_periodic(n=FRAME_LEN):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


@dataclass
class FrameResult:
    frame: int            # first sample of the frame (relative to the recording start)
    voice: bool
    energy_db: float
    tau_s: np.ndarray     # (15,) seconds, PAIRS order
    peak: np.ndarray      # (15,) GCC-PHAT peak value, 0..1
    corr: list = None     # per pair: r(n) for n = -(L+1) .. L+1 (SRP-PHAT coarse grid)
    cross: list = None    # per pair: PHAT-weighted cross spectrum W[k] over the band (SRP-PHAT refinement)
    spec: np.ndarray = None   # (6, n_band) band spectra X_c[k], K_LO..K_HI (MUSIC)


class TdoaEstimator:
    def __init__(self, geometry):
        self.g = geometry
        self.window = hann_periodic()
        self.k_lo, self.k_hi = band_bins(geometry.fs)
        self.k = np.arange(self.k_lo, self.k_hi + 1)
        self.max_lag = [geometry.max_lag_samples(i, j) for i, j in PAIRS]
        self.floor_db = None

    def spectra(self, frame):
        """frame: (N, 6) int16 -> band spectra (6, n_band) complex."""
        x = frame.astype(np.float64) / 32768.0 * self.window[:, None]
        return np.fft.rfft(x, axis=0)[self.k_lo:self.k_hi + 1].T

    def vad(self, spec):
        energy = float(np.mean(np.sum(np.abs(spec) ** 2, axis=1)))
        energy_db = 10.0 * np.log10(energy + 1e-20)
        self.floor_db = energy_db if self.floor_db is None else min(energy_db, self.floor_db + VAD_FLOOR_RISE_DB)
        return energy_db > self.floor_db + VAD_MARGIN_DB, energy_db

    def gcc_phat(self, xi, xj, max_lag):
        cross = xi * np.conj(xj)
        w = cross / (np.abs(cross) + 1e-20)
        lags = np.arange(-max_lag - 1, max_lag + 2)
        r = np.real(np.exp(2j * np.pi * np.outer(lags, self.k) / FRAME_LEN) @ w) / len(self.k)
        inner = r[1:-1]                       # lags -max_lag .. max_lag
        m = int(np.argmax(inner)) + 1         # index into r
        y0, ym, yp = r[m], r[m - 1], r[m + 1]
        den = ym - 2 * y0 + yp
        delta = 0.5 * (ym - yp) / den if den != 0 else 0.0
        return (lags[m] + delta) / self.g.fs, y0 - 0.25 * (ym - yp) * delta, r, w

    def process(self, frame, frame_start=0):
        spec = self.spectra(frame)
        voice, energy_db = self.vad(spec)
        tau = np.zeros(len(PAIRS))
        peak = np.zeros(len(PAIRS))
        corr, cross = [], []
        for p, (i, j) in enumerate(PAIRS):
            tau[p], peak[p], r, w = self.gcc_phat(spec[i], spec[j], self.max_lag[p])
            corr.append(r)
            cross.append(w)
        return FrameResult(frame_start, voice, energy_db, tau, peak, corr, cross, spec)

    def run(self, samples):
        """samples: (n, 6) int16 recording -> list of FrameResult, frames at k * HOP."""
        out = []
        for start in range(0, samples.shape[0] - FRAME_LEN + 1, HOP):
            out.append(self.process(samples[start:start + FRAME_LEN], start))
        return out
