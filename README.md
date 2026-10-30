# daq-sim

Simulation and offline algorithm development for the **Audio Detection System** (6-microphone speaker localization on ESP32-S3).

Algorithms are developed and compared here in Python on simulated signals and on raw 6-channel recordings from the device SD card. The best candidate is then ported to C in `daq-detector`.

## Scope

- Array geometry and room simulation (direct path + reverberation)
- Shared front end: voice activity detection (300–3500 Hz energy), STFT (2048 samples, Hann)
- Compared algorithms:
  - **A** — GCC-PHAT + least squares (one 3D position per frame)
  - **B** — SRP-PHAT (power map, multiple speakers)
  - **C** — MUSIC on the M1–M3 triangle (directions + number of sources)
- Evaluation: azimuth/elevation error, distance error, speaker-count accuracy

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Recordings and generated results are not tracked in git (see `.gitignore`). Keep them in `data/` locally.

## Related

- `daq-detector` — ESP32-S3 firmware.
