# daq-sim

PC side of the **Audio Detection System** (6-microphone speaker localization on ESP32-S3):
- tools for receiving and loading recordings from `daq-detector`,
- simulation and offline development of the localization algorithms in Python (reference implementations that the firmware must match).

The best algorithm is later ported to C in `daq-detector`.

## Setup

Requires Python 3.10+.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Files

### `requirements.txt`
| Package | Used for |
|---|---|
| `numpy` | signal processing, least squares, sample arrays |
| `pyserial` | reading the native USB port of the ESP32-S3 |

### `tools/usb_record.py`
Receives the 6-channel stream from `daq-detector` over native USB and saves it as a WAV file.

```
python tools/usb_record.py --port COM7 --out data/usb_test.wav --seconds 30
```

| Argument | Default | Meaning |
|---|---|---|
| `--port` | required | native USB port of the ESP32-S3 (Windows `COMx`, Linux `/dev/ttyACM0`); not the CP2102N UART port |
| `--out` | required | output WAV path |
| `--seconds` | 10 | recording length |

What it does:
1. Opens the port and sends `S` (start streaming).
2. Parses packets: finds the `DAQ1` magic, reads the 24-byte header, checks it and extracts the payload (header format in `daq-detector/README.md`).
3. Checks `seq` continuity. Lost packets are counted and replaced with silence, so the timeline stays aligned (important for time-difference measurements).
4. Writes the samples to a WAV file (6 channels, rate taken from the stream, 16-bit).
5. Every second prints the throughput in kB/s, the RMS level of M1..M6 in dBFS, lost packets and resyncs.
6. On exit sends `X` (stop streaming) and closes the file.

Output line example:
```
   576 kB/s | M1: -52.3 M2: -51.8 M3: -53.0 M4: -50.9 M5: -52.1 M6: -51.5 | lost=0 resync=0
```
The full stream is ~576 kB/s. A lower value with growing `lost` means native USB does not keep up.

### `tools/simulate_wav.py`
Makes a synthetic 6-channel WAV with one or more point sources, in free field or in a shoebox room with reverberation.

```
python tools/simulate_wav.py data/sim.wav --source 60 20 2.0 --seconds 5
python tools/simulate_wav.py data/sim2.wav --source 60 20 2 --source 200 -5 3.5 --snr 20
python tools/simulate_wav.py data/room.wav --source 60 20 2 --room 6 5 3 --array-pos 2.5 2 0.661 --rt60 0.5
```

| Argument | Default | Meaning |
|---|---|---|
| `--source AZ EL DIST` | required, repeatable | azimuth / elevation in degrees, distance in metres (array coordinates) |
| `--seconds` | 5 | length |
| `--snr` | 30 | signal-to-noise ratio per mic, dB |
| `--config` | — | take the geometry from `daq-detector/main/config.h` instead of the built-in copy |
| `--room LX LY LZ` | — (free field) | shoebox room size in m |
| `--array-pos X Y Z` | room centre, 0.661 m high | array origin (centre of M1–M3) in room coordinates, m |
| `--rt60` | 0.5 | reverberation time in s (sets one reflection coefficient for all walls, Sabine) |

The source signal is speech-like noise (300–3500 Hz, 4 Hz syllable envelope, 0.8 s talk / 0.5 s pause, first 0.5 s silent),
delayed to each mic with an exact fractional delay and attenuated as 1/r. The file can be copied to the SD card
and processed by the device in replay mode.

### `tools/localize_wav.py`
Reference localization — **algorithm A** (GCC-PHAT + least squares), **B** (SRP-PHAT) or **C** (MUSIC on M1–M3) — on a 6-channel WAV
(from the SD card, `usb_record.py` or `simulate_wav.py`).

```
python tools/localize_wav.py data/REC00003.WAV
python tools/localize_wav.py data/REC00003.WAV --alg B
python tools/localize_wav.py data/REC00003.WAV --alg B --track AZEL
python tools/localize_wav.py data/REC00003.WAV --alg C
python tools/localize_wav.py data/REC00003.WAV --compare data/PLY00003.CSV
```

| Argument | Meaning |
|---|---|
| `--csv` | output CSV (default `<wav>_pyA.csv` / `_pyB.csv`) |
| `--alg A\|B\|C` | algorithm (default A) |
| `--track AZ\|AZEL\|ELLIPSE` | tracker association mode (default AZ, see `daq_sim/track.py`) |
| `--config` | geometry from `daq-detector/main/config.h` |
| `--compare` | device CSV from replay mode: geometry, speed of sound, algorithm and tracker mode are read from its header; frames with voice on both sides are compared: time differences (target ≤ 1 µs), kind and number of sources, directions (≤ 0.05°) and distances (≤ 5 mm); tracks in every frame (count, ids, directions, distances) |

CSV: first line `# ... first_frame=… fs=… c=… alg=… trk=… mics=x,y,z;…` (mm), then one row per frame:
`frame, vad, energy_db, t12…t56` (τ in µs, 15 pairs), `p12…p56` (GCC-PHAT peak 0..1), and for voice frames
`alg, kind, speakers, az_deg, el_deg, dist_m, conf, res_us, pairs, az2_deg, el2_deg, dist2_m, conf2`
(`kind`: P position, D direction only with `dist_m` 0, `-` none; `res_us`/`pairs` for A only; second source for B and C; C: azimuth only),
then on every row the tracker: `tracks, t1_id, t1_az_deg, t1_el_deg, t1_dist_m, t1_conf, t2_…` (confirmed tracks = speakers, two most confident).
The firmware (0.8.0) writes the same columns. The script prints how many frames gave a position, a direction or nothing, and the
number of tracked speakers per frame.

Results (source at 60° / 20° / 2 m, SNR 30 dB, 4 s):

| Conditions | Algorithm | Frames with a result | Azimuth (median, p10–p90) | Elevation | Distance (P) |
|---|---|---|---|---|---|
| free field | A or B | 78 / 78 (all P) | 60.0° (59.9–60.1) | 20.0° (19.6–20.5) | 2.0 m (1.9–2.1) |
| room 6 × 5 × 3 m, RT60 0.5 s | A | 15 / 89 (8 P, 7 D) | 59.5° (56.8–64.8) | 22° (−4…28) | 0.9 m (0.6–2.5) |
| same room | B | 80 / 89 (33 P, 47 D) | 60.0° (58.6–80.0) | 21° (−5…25) | 1.8 m (0.6–3.7) |
| same room, second source at 220° / 0° / 3 m | B | 55 / 89 | both directions found, but 2 sources in only 28 frames | | |

### `daq_sim/__init__.py`
Marks `daq_sim` as a Python package (empty).

### `daq_sim/geometry.py`
Shared array geometry. Source of truth: `daq-detector/main/config.h` (pinout DAQ-PIN-01).
- `MIC_XYZ_MM`, `SPEED_OF_SOUND_MPS`, `SAMPLE_RATE_HZ`: copies of the firmware defaults.
- `PAIRS`: the 15 mic pairs in the firmware / CSV order (12, 13, …, 56).
- `Geometry`: mic positions (m), speed of sound, sample rate; `tdoa(source)` = expected τ for all pairs, `max_lag_samples(i, j)`.
- `load_geometry(config_h)`: reads `DAQ_MIC_XYZ_MM` and `DAQ_SPEED_OF_SOUND_MPS` from `config.h`.
- `parse_csv_header(line)`: geometry the device actually used (NVS calibration), from its CSV log.
- `to_spherical()`, `from_spherical()`: position ↔ azimuth, elevation, distance.

Coordinates: origin = centre of the M1–M3 triangle, x = azimuth 0°, y = azimuth 90°, z = up.

### `daq_sim/tdoa.py`
Reference front end, **the specification for `daq-detector/main/dsp.c`** (the docstring lists every step and constant):
frames of 2048 samples with hop 1536 (32 ms), periodic Hann window, band 300–3500 Hz,
VAD on band energy with a noise floor (floor rises at most 0.05 dB per frame, voice = 9 dB above the floor),
GCC-PHAT for 15 pairs evaluated at integer lags within the physical range + 2 samples, parabolic refinement.
`TdoaEstimator.run(samples)` → list of `FrameResult(frame, voice, energy_db, tau_s[15], peak[15])`.

### `daq_sim/ls.py`
Algorithm A, step 2: position from the 15 time differences. Linear start (reference M1, unknowns position and distance to M1)
or a given start point, Gauss–Newton on all pairs, rejection of the worst pair while its residual exceeds 25 µs (at most 2 pairs);
ok if finite, RMS ≤ 25 µs and within 8 m. `direction(geometry, tau_s)`: plane-wave fit for decision e (direction only).
`locate(geometry, tau_s)` → `LsResult(position_m, rms_residual_us, used, ok)`; `confidence(result, peak)` = mean peak of the used pairs × (1 − RMS / 25 µs).
Ported to `daq-detector/main/localize.c` (firmware 0.5.0); on the simulated recording both agree within 0.005° and 0.5 mm.

### `daq_sim/srp.py`
**Algorithm B (SRP-PHAT)**, the specification for the firmware (docstring lists every step and constant):
coarse power grid = BLE map grid (36 × 18 directions) × 8 distances on the GCC-PHAT correlations, up to 2 local maxima,
pattern-search refinement on the exact correlation, SRP-guided time differences + least squares (position) or the SRP direction only.
`SrpPhat(geometry, max_lags).process(corr, cross)` → `SrpResult(sources, map)`; the map is the 36 × 18 u8 map sent over BLE.

### `daq_sim/music.py`
**Algorithm C (MUSIC on M1–M3)**, specification of `daq-detector/main/music.c` (docstring lists every step and constant).
`Music(geometry).process(frame)` is called for every frame (the covariance decays also without voice): 40 single bins in
300–2800 Hz, whitened M1–M3 snapshots, 3 × 3 covariance averaged over ~16 frames, MDL count K = 1 or 2 (`MDL_PENALTY`),
pseudospectrum on a 1° azimuth grid, K largest peaks. Azimuth only (elevation 0, distance 0, kind D): the triangle is planar.

### `tools/evaluate.py`
Evaluation of A, B, C and the tracker on simulated versions of the test plan (project documentation 8.5):
T1 one speaker (azimuths, distances 1–4 m, heights 1.2 / 1.6 m), T2 two speakers (20–180° apart), T3 three-person conversation,
T4 small office (RT60 0.3 s) and lecture hall (1.0 s), T5 ventilation, music, footsteps. The array is 1.2 m above the floor.

```
python tools/evaluate.py                 # quick set, 14 scenarios (~6 min)
python tools/evaluate.py --full          # T1 8 azimuths x 4 distances x 2 heights, T2 4 angles (~30 min)
python tools/evaluate.py --only T2 T5 --alg B
```

Metrics: azimuth / elevation / distance error (mean, p90) of the sources within 20° of an active speaker, false sources,
detection (voice frames with a matching source), counting accuracy of the tracker (per frame and per 1 s window, against the
speakers active in the last 1.5 s). The threshold sweep (tracker mode × `BIRTH_HIST` × `CONFIRM_HITS`, MUSIC `MDL_PENALTY`)
runs on cached per-frame results and does not change any default. Output: `output/eval/report.md` and `results.json`.
Results of 02.11.2026 and the proposals derived from them: `daq-detector/docs/firmware_plan.md`, section "Evaluation 02.11.2026".

### `daq_sim/track.py`
Speaker tracking and counting over time (decision 1c), specification of `daq-detector/main/track.c`. `Tracker(fs, mode).update(frame, sources)`
is called for every frame; a direction histogram (1 s half-life, unit weight per source) decides where tracks are born, a tracker
(gate 20°, confirmed after 4 hits, timeouts 0.3 s tentative / 1.5 s confirmed, up to 4 tracks) follows them; speakers = confirmed tracks.
Association modes: `AZ` (1, default, azimuth only, merge < 30°), `AZEL` (2, azimuth × elevation), `ELLIPSE` (3, gate ±15° azimuth / ±45° elevation).
In the reverberant simulation only AZ counts the two speakers (60° / 220°) correctly and has no ghost track with one speaker; AZEL can
separate speakers at different heights in the same direction, if the elevation is reliable. Choose on real recordings.

### `daq_sim/simulate.py`
`render(geometry, sources, seconds, snr_db, room=None)` → int16 (n, 6) recording; `Room(size_m, array_pos_m, rt60_s, max_order)` switches on
the image-source method (reflection order chosen from RT60, at most 12; fractional-delay windowed-sinc impulse responses); `speech_like()` test signal.

### `daq_sim/recording.py`
`load_recording(path) -> (samples, sample_rate)` loads a WAV recording from the SD card or from `usb_record.py`.
- `samples`: `int16` NumPy array of shape `(frames, channels)`, columns M1..M6.
- Works on files cut by a power loss (`data_size = 0` in the header): the data length is taken from the file size and a trailing partial frame is dropped.
- Accepts only 16-bit PCM.

```python
from daq_sim.recording import load_recording

samples, fs = load_recording("data/REC00001.WAV")
m1 = samples[:, 0]
```

### `.gitignore`
Excludes virtual environments, caches and data: `data/`, `recordings/`, `output/`, `*.wav`, `*.raw`, `*.npy`, `*.npz`, `*.h5`. Keep recordings locally in `data/`.

## Scope

| Status | Item |
|---|---|
| done | shared geometry, free-field simulation, front end (VAD, GCC-PHAT), algorithm A (GCC-PHAT + least squares) |
| done | reverberation (image-source shoebox room) |
| done | algorithm A with direction-only fallback, algorithm B (SRP-PHAT, 2 sources, map) — ported to firmware 0.6.0 |
| done | speaker tracking and counting (histogram + tracker, modes AZ / AZEL / ELLIPSE) — ported to firmware 0.7.0 |
| done | algorithm C (MUSIC on M1–M3, azimuth only) — ported to firmware 0.8.0 |
| done | simulated evaluation T1–T5 with threshold sweep (`tools/evaluate.py`) |
| next | the same evaluation on real recordings, choice of the tracker mode and thresholds |
| later | evaluation on recordings T1–T5 (azimuth/elevation/distance error, counting accuracy) |

Workflow (decision D3): an algorithm is written and validated here first, then ported to C in `daq-detector`.
The device replay mode (D4) runs a WAV from the SD card through the firmware pipeline and writes `PLYnnnnn.CSV`;
`localize_wav.py --compare` checks it against this reference.

## Related

- `daq-detector`: ESP32-S3 firmware (acquisition, SD recording, USB stream).
