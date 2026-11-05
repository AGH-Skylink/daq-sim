"""Synthetic 6-channel recordings: point sources in free field or in a shoebox room.

Free field: each source signal is delayed to every mic with an exact fractional delay (frequency domain)
and attenuated as 1/r. Room: image-source method (Allen & Berkley) with one reflection coefficient for
all walls, set from RT60 (Sabine); each mic gets an impulse response built from fractional-delay
windowed-sinc taps, convolved with the source. Independent white noise is added per mic.
Output: int16 (n, 6), like the device.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import from_spherical


def speech_like(n, fs, rng):
    """Stand-in for speech: noise shaped to 300-3500 Hz, 4 Hz syllable envelope, talk spurts
    (0.8 s talk / 0.5 s pause, first 0.5 s silent) so the VAD noise floor can follow the pauses."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 300) | (f > 3500)] = 0
    x = np.fft.irfft(spec, n)
    t = np.arange(n) / fs
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2
    talk = (t >= 0.5) & (((t - 0.5) % 1.3) < 0.8)
    return x * env * talk / np.max(np.abs(x))


@dataclass(frozen=True)
class Room:
    size_m: tuple            # (Lx, Ly, Lz)
    array_pos_m: tuple       # array origin (centre of M1-M3) in room coordinates
    rt60_s: float = 0.5
    max_order: int = 0       # 0 = automatic from RT60

    def beta(self):
        lx, ly, lz = self.size_m
        volume = lx * ly * lz
        surface = 2 * (lx * ly + lx * lz + ly * lz)
        alpha = min(0.161 * volume / (surface * self.rt60_s), 1.0)   # Sabine
        return float(np.sqrt(1.0 - alpha))


SINC_HALF = 16   # taps on each side of a fractional delay


def room_impulse_responses(geometry, room, source_room_m):
    """Impulse responses (6, L) from a source at source_room_m to the six mics, image-source method."""
    fs, c = geometry.fs, geometry.c
    size = np.asarray(room.size_m, dtype=float)
    mics = np.asarray(room.array_pos_m, dtype=float) + geometry.mics_m
    length = int(room.rt60_s * fs) + 2 * SINC_HALF + int(np.max(size) * 2 / c * fs)
    order = room.max_order or int(min(np.ceil(c * room.rt60_s / np.min(size)), 12))
    beta = room.beta()

    n = np.arange(-order, order + 1)
    q = np.array([0, 1])
    nx, ny, nz, qx, qy, qz = np.meshgrid(n, n, n, q, q, q, indexing="ij")
    nn = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    qq = np.stack([qx.ravel(), qy.ravel(), qz.ravel()], axis=1)
    images = (1 - 2 * qq) * np.asarray(source_room_m) + 2 * nn * size
    reflections = np.sum(np.abs(nn - qq) + np.abs(nn), axis=1)
    keep = reflections <= order
    images, gain = images[keep], beta ** reflections[keep]

    h = np.zeros((6, length))
    taps = np.arange(-SINC_HALF, SINC_HALF + 1)
    window = 0.5 + 0.5 * np.cos(np.pi * taps / (SINC_HALF + 1))
    for m in range(6):
        d = np.linalg.norm(images - mics[m], axis=1)
        t = d / c * fs
        ok = t < length - SINC_HALF - 1
        t, amp = t[ok], gain[ok] / d[ok]
        base = np.floor(t).astype(int)
        frac = t - base
        idx = base[:, None] + taps[None, :]
        vals = amp[:, None] * np.sinc(taps[None, :] - frac[:, None]) * window[None, :]
        np.add.at(h[m], idx.ravel(), vals.ravel())
    return h


def _convolve(x, h, n):
    size = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:n]


def render(geometry, sources, seconds, snr_db=30.0, level_dbfs=-20.0, seed=1, room=None):
    """sources: list of (az_deg, el_deg, dist_m) or (az, el, dist, signal) in array coordinates.
    room: None = free field, or a Room. Returns int16 (n, 6)."""
    fs = geometry.fs
    n = int(seconds * fs)
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 6))
    f = np.fft.rfftfreq(n, 1 / fs)
    for src in sources:
        az, el, dist = src[:3]
        sig = src[3] if len(src) > 3 else speech_like(n, fs, rng)
        pos = from_spherical(az, el, dist)
        if room is not None:
            pos_room = np.asarray(room.array_pos_m) + pos
            if np.any(pos_room <= 0) or np.any(pos_room >= np.asarray(room.size_m)):
                raise ValueError(f"source {src[:3]} is outside the room")
            h = room_impulse_responses(geometry, room, pos_room)
            for c in range(6):
                out[:, c] += _convolve(sig[:n], h[c], n)
            continue
        d = np.linalg.norm(geometry.mics_m - pos, axis=1)
        spec = np.fft.rfft(sig[:n], n)
        for c in range(6):
            delay = d[c] / geometry.c
            out[:, c] += np.fft.irfft(spec * np.exp(-2j * np.pi * f * delay), n) * (1.0 / d[c])
    peak = np.max(np.abs(out))
    out *= 10 ** (level_dbfs / 20) / peak
    sig_rms = np.sqrt(np.mean(out ** 2))
    out += rng.standard_normal(out.shape) * sig_rms * 10 ** (-snr_db / 20)
    return np.clip(np.round(out * 32767), -32768, 32767).astype(np.int16)
