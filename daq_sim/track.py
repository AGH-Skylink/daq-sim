"""Speaker tracking and counting over time (decision 1c: direction histogram + tracker).

Specification for the firmware port (daq-detector/main/track.c). `Tracker.update()` is called for every frame
(hop 32 ms), with the sources of that frame (empty list for frames without voice or without a result).

Three association modes (selectable, to be compared on real recordings):
  AZ      (1, default) azimuth only: 1-D histogram of 36 azimuth bins, distance = |wrapped azimuth difference|,
          merge below MERGE_DEG_AZ. Robust to unstable elevation in reverberant rooms; speakers closer than
          MERGE_DEG_AZ in azimuth count as one.
  AZEL    (2) azimuth x elevation: 2-D histogram (the 36 x 18 grid of the SRP map), distance = great-circle angle.
  ELLIPSE (3) 2-D histogram, distance = GATE_DEG * max(|d_az| / ELLIPSE_AZ_DEG, |d_el| / ELLIPSE_EL_DEG)
          (an elliptical gate: tight in azimuth, loose in elevation).

1. Histogram decays by 0.5 ** (HOP_S / HIST_HALF_LIFE_S) per frame; every source adds 1 (unit weight, the
   confidences of positions and of directions are on different scales) times a kernel at its cell:
   AZ: centre 1, azimuth neighbours 0.5; AZEL / ELLIPSE: centre 1, 4 neighbours 0.5, diagonals 0.25.
   Azimuth wraps, elevation is clipped. Cell: k = floor(az / 10 + 0.5) mod 36, j = clip(floor((el + 85) / 10 + 0.5), 0, 17).
2. Association, sources in order of decreasing confidence: the nearest track (mode distance) not yet updated in
   this frame within GATE_DEG is updated (first one wins on ties); otherwise, if the histogram at the source cell
   >= BIRTH_HIST and there are fewer than MAX_TRACKS tracks, a tentative track is born (ids 1..255, then wrap,
   0 is never used).
   Update with ALPHA: direction = normalised mix of unit vectors, distance = mix of the known distances (a source with
   distance 0 does not change it; the first known distance is taken as is), confidence = mix, hits + 1.
   A track with CONFIRM_HITS hits is confirmed.
3. A tentative track not updated for TENTATIVE_TIMEOUT_S, or a confirmed one for CONFIRMED_TIMEOUT_S, is removed.
4. Two tracks closer than the merge distance (mode distance; MERGE_DEG_AZ in AZ mode, MERGE_DEG otherwise):
   the younger one is removed.
5. Output: confirmed tracks by decreasing confidence; speaker count = their number.
"""
from dataclasses import dataclass

import numpy as np

from .srp import N_AZ, N_EL, angular_distance_deg

MODES = {"AZ": 1, "AZEL": 2, "ELLIPSE": 3}
MODE_NAMES = {v: k for k, v in MODES.items()}
DEFAULT_MODE = MODES["AZ"]

HOP_S = 1536 / 48000
HIST_HALF_LIFE_S = 1.0
BIRTH_HIST = 1.5
GATE_DEG = 20.0
ELLIPSE_AZ_DEG = 15.0
ELLIPSE_EL_DEG = 45.0
ALPHA = 0.3
CONFIRM_HITS = 4
TENTATIVE_TIMEOUT_S = 0.3
CONFIRMED_TIMEOUT_S = 1.5
MERGE_DEG = 10.0
MERGE_DEG_AZ = 30.0
MAX_TRACKS = 4


@dataclass
class Track:
    id: int
    az_deg: float
    el_deg: float
    dist_m: float          # 0 = unknown
    confidence: float
    hits: int
    born: int              # frame (first sample) of birth
    last: int              # frame of the last update
    confirmed: bool = False


def _cell(az, el):
    k = int(np.floor(az / 10.0 + 0.5)) % N_AZ
    j = min(max(int(np.floor((el + 85.0) / 10.0 + 0.5)), 0), N_EL - 1)
    return j, k


def _unit(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _az_diff(a1, a2):
    return abs((a1 - a2 + 180.0) % 360.0 - 180.0)


class Tracker:
    def __init__(self, fs=48000, mode=DEFAULT_MODE):
        if mode not in MODE_NAMES:
            raise ValueError(f"tracker mode {mode}")
        self.fs = fs
        self.mode = mode
        self.hist = np.zeros(N_AZ) if mode == MODES["AZ"] else np.zeros((N_EL, N_AZ))
        self.decay = 0.5 ** (HOP_S / HIST_HALF_LIFE_S)
        self.merge = MERGE_DEG_AZ if mode == MODES["AZ"] else MERGE_DEG
        self.tracks = []
        self.next_id = 1

    def distance(self, az1, el1, az2, el2):
        if self.mode == MODES["AZ"]:
            return _az_diff(az1, az2)
        if self.mode == MODES["AZEL"]:
            return angular_distance_deg(az1, el1, az2, el2)
        return GATE_DEG * max(_az_diff(az1, az2) / ELLIPSE_AZ_DEG, abs(el1 - el2) / ELLIPSE_EL_DEG)

    def _add(self, j, k):
        if self.mode == MODES["AZ"]:
            for dk, kern in ((-1, 0.5), (0, 1.0), (1, 0.5)):
                self.hist[(k + dk) % N_AZ] += kern
            return
        for dj in (-1, 0, 1):
            jj = j + dj
            if not 0 <= jj < N_EL:
                continue
            for dk in (-1, 0, 1):
                kern = 1.0 if dj == dk == 0 else (0.5 if dj == 0 or dk == 0 else 0.25)
                self.hist[jj, (k + dk) % N_AZ] += kern

    def _hist_at(self, j, k):
        return self.hist[k] if self.mode == MODES["AZ"] else self.hist[j, k]

    def _new_id(self):
        tid = self.next_id
        self.next_id = self.next_id + 1 if self.next_id < 255 else 1
        return tid

    def update(self, frame, sources):
        """frame: first sample of the frame; sources: list of (az_deg, el_deg, dist_m, confidence)."""
        self.hist *= self.decay
        for az, el, _, _ in sources:
            self._add(*_cell(az, el))

        updated = set()
        for az, el, dist, conf in sorted(sources, key=lambda s: -s[3]):
            best, best_d = None, GATE_DEG
            for t in self.tracks:
                if t.id in updated:
                    continue
                d = self.distance(az, el, t.az_deg, t.el_deg)
                if d <= best_d:
                    best, best_d = t, d
            if best is not None:
                v = (1 - ALPHA) * _unit(best.az_deg, best.el_deg) + ALPHA * _unit(az, el)
                v /= np.linalg.norm(v)
                best.az_deg = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
                best.el_deg = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
                if dist > 0:
                    best.dist_m = dist if best.dist_m == 0 else (1 - ALPHA) * best.dist_m + ALPHA * dist
                best.confidence = (1 - ALPHA) * best.confidence + ALPHA * conf
                best.hits += 1
                best.last = frame
                best.confirmed = best.confirmed or best.hits >= CONFIRM_HITS
                updated.add(best.id)
            elif self._hist_at(*_cell(az, el)) >= BIRTH_HIST and len(self.tracks) < MAX_TRACKS:
                t = Track(self._new_id(), az, el, dist, conf, 1, frame, frame)
                self.tracks.append(t)
                updated.add(t.id)

        def alive(t):
            timeout = CONFIRMED_TIMEOUT_S if t.confirmed else TENTATIVE_TIMEOUT_S
            return frame - t.last <= timeout * self.fs
        self.tracks = [t for t in self.tracks if alive(t)]

        keep = []
        for t in self.tracks:   # creation order: an older track wins
            if all(self.distance(t.az_deg, t.el_deg, o.az_deg, o.el_deg) > self.merge for o in keep):
                keep.append(t)
        self.tracks = keep

        return sorted((t for t in self.tracks if t.confirmed), key=lambda t: -t.confidence)
