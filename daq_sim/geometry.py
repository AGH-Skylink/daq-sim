"""Array geometry shared by the simulation, the reference algorithms and the firmware.

Source of truth: `daq-detector/main/config.h` (DAQ_MIC_XYZ_MM, DAQ_SPEED_OF_SOUND_MPS), which follows
pinout DAQ-PIN-01. The constants below mirror it; `load_geometry(config_h)` reads the file directly,
and `parse_csv_header()` reads the geometry the device actually used (NVS calibration) from its CSV log.

Coordinates in metres: origin = centre of the M1-M3 triangle, x = azimuth 0, y = azimuth 90, z = up.
"""
import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 48000
SPEED_OF_SOUND_MPS = 343.0

# M1..M6 in mm (pinout DAQ-PIN-01): body 90/210/330 deg, arms 150 (M4) / 270 (M5) / 30 (M6) deg, 161 mm lower
MIC_XYZ_MM = (
    (0.0, 34.6, 0.0),
    (-30.0, -17.3, 0.0),
    (30.0, -17.3, 0.0),
    (-303.1, 175.0, -161.0),
    (0.0, -350.0, -161.0),
    (303.1, 175.0, -161.0),
)

# The 15 pairs (i, j), i < j, in the order used by the firmware and the CSV columns (t12, t13, ..., t56)
PAIRS = tuple(itertools.combinations(range(6), 2))


@dataclass(frozen=True)
class Geometry:
    mics_m: np.ndarray            # (6, 3) metres
    c: float = SPEED_OF_SOUND_MPS
    fs: int = SAMPLE_RATE_HZ

    def pair_distance(self, i, j):
        return float(np.linalg.norm(self.mics_m[i] - self.mics_m[j]))

    def max_lag_samples(self, i, j):
        """Search range of GCC-PHAT for a pair: physical maximum + 2 samples (same rule as dsp.c)."""
        return int(np.ceil(self.pair_distance(i, j) / self.c * self.fs)) + 2

    def tdoa(self, source_m):
        """Expected tau_ij = (|s - m_i| - |s - m_j|) / c for all PAIRS, in seconds."""
        d = np.linalg.norm(self.mics_m - np.asarray(source_m, dtype=float), axis=1)
        return np.array([(d[i] - d[j]) / self.c for i, j in PAIRS])


def default_geometry():
    return Geometry(np.array(MIC_XYZ_MM) / 1000.0)


def load_geometry(config_h=None):
    """Geometry from config.h (DAQ_MIC_XYZ_MM, DAQ_SPEED_OF_SOUND_MPS), or the defaults above."""
    if config_h is None:
        return default_geometry()
    text = Path(config_h).read_text(encoding="utf-8")
    block = re.search(r"#define\s+DAQ_MIC_XYZ_MM\s*\{(.*?)\n\}", text, re.S)
    if not block:
        raise ValueError(f"{config_h}: DAQ_MIC_XYZ_MM not found")
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", block.group(1).replace("\\", ""))]
    if len(nums) != 18:
        raise ValueError(f"{config_h}: expected 18 coordinates, got {len(nums)}")
    c = re.search(r"#define\s+DAQ_SPEED_OF_SOUND_MPS\s+(\d+(?:\.\d+)?)", text)
    return Geometry(np.array(nums).reshape(6, 3) / 1000.0, float(c.group(1)) if c else SPEED_OF_SOUND_MPS)


def parse_csv_header(line):
    """Geometry from a device CSV header line: '# ... fs=48000 c=343.0 mics=x,y,z;...' (mm)."""
    fields = dict(kv.split("=", 1) for kv in line.lstrip("#").split() if "=" in kv)
    mics = np.array([[float(v) for v in m.split(",")] for m in fields["mics"].split(";")]) / 1000.0
    return Geometry(mics, float(fields.get("c", SPEED_OF_SOUND_MPS)), int(fields.get("fs", SAMPLE_RATE_HZ))), fields


def to_spherical(p):
    """Position (m) -> azimuth deg [0, 360), elevation deg, distance m."""
    x, y, z = p
    az = np.degrees(np.arctan2(y, x)) % 360.0
    el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    return float(az), float(el), float(np.linalg.norm(p))


def from_spherical(az_deg, el_deg, dist_m):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return dist_m * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
