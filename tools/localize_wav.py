"""Reference localization (algorithm A, B or C) on a 6-channel WAV recording.

    python tools/localize_wav.py data/REC00003.WAV --csv data/REC00003_py.csv
    python tools/localize_wav.py data/REC00003.WAV --alg B
    python tools/localize_wav.py data/REC00003.WAV --alg B --track AZEL
    python tools/localize_wav.py data/REC00003.WAV --alg C
    python tools/localize_wav.py data/REC00003.WAV --compare data/PLY00003.CSV

Writes one CSV row per frame (same columns as the firmware log).
--compare checks the result against a CSV made by the device in replay mode: the geometry, speed of sound
algorithm and tracker mode are then taken from the device CSV header; frames with voice on both sides are compared
(targets: time differences <= 1 us, directions <= 0.05 deg, distances <= 5 mm).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daq_sim.geometry import PAIRS, load_geometry, parse_csv_header, to_spherical  # noqa: E402
from daq_sim.ls import confidence, direction, locate  # noqa: E402
from daq_sim.music import Music  # noqa: E402
from daq_sim.recording import load_recording  # noqa: E402
from daq_sim.srp import SrpPhat  # noqa: E402
from daq_sim.tdoa import TdoaEstimator  # noqa: E402
from daq_sim.track import DEFAULT_MODE, MODE_NAMES, MODES, Tracker  # noqa: E402

PAIR_NAMES = [f"{i + 1}{j + 1}" for i, j in PAIRS]
TDOA_COLUMNS = ["frame", "vad", "energy_db"] + [f"t{p}" for p in PAIR_NAMES] + [f"p{p}" for p in PAIR_NAMES]
# kind: P = position, D = direction only (distance unknown, dist_m = 0), - = no result
LOC_COLUMNS = ["alg", "kind", "speakers", "az_deg", "el_deg", "dist_m", "conf", "res_us", "pairs",
               "az2_deg", "el2_deg", "dist2_m", "conf2"]
# tracker output (every frame): number of confirmed tracks and the two most confident ones
TRACK_COLUMNS = ["tracks", "t1_id", "t1_az_deg", "t1_el_deg", "t1_dist_m", "t1_conf",
                 "t2_id", "t2_az_deg", "t2_el_deg", "t2_dist_m", "t2_conf"]


def localize_a(geometry, r):
    """Algorithm A with decision e: position if the fit is good, otherwise direction only.
    Returns (CSV columns, sources as (az, el, dist, confidence))."""
    res = locate(geometry, r.tau_s)
    if res.ok:
        az, el, dist = to_spherical(res.position_m)
        conf = confidence(res, r.peak)
        return (["A", "P", 1, f"{az:.2f}", f"{el:.2f}", f"{dist:.3f}", f"{conf:.3f}",
                 f"{res.rms_residual_us:.2f}", len(res.used)] + [""] * 4, [(az, el, dist, conf)])
    d = direction(geometry, r.tau_s)
    if d.ok:
        conf = confidence(d, r.peak)
        return (["A", "D", 1, f"{d.az_deg:.2f}", f"{d.el_deg:.2f}", "0.000", f"{conf:.3f}",
                 f"{d.rms_residual_us:.2f}", len(d.used)] + [""] * 4, [(d.az_deg, d.el_deg, 0.0, conf)])
    return ["A", "-", 0] + [""] * 10, []


def localize_b(srp, r):
    out = srp.process(r.corr, r.cross)
    row = ["B", out.sources[0].kind if out.sources else "-", len(out.sources)]
    for k in range(2):
        if k < len(out.sources):
            s = out.sources[k]
            vals = [f"{s.az_deg:.2f}", f"{s.el_deg:.2f}", f"{s.dist_m:.3f}", f"{s.confidence:.3f}"]
        else:
            vals = ["", "", "", ""]
        row += vals
        if k == 0:
            row += ["", ""]   # res_us, pairs: algorithm A only
    return row, [(s.az_deg, s.el_deg, s.dist_m, s.confidence) for s in out.sources]


def localize_c(music, r):
    """Algorithm C; must be called for every frame (the covariance also decays without voice)."""
    out = music.process(r)
    if out is None:
        return None, []
    row = ["C", "D" if out.sources else "-", len(out.sources)]
    for k in range(2):
        if k < len(out.sources):
            s = out.sources[k]
            row += [f"{s.az_deg:.2f}", f"{s.el_deg:.2f}", f"{s.dist_m:.3f}", f"{s.confidence:.3f}"]
        else:
            row += ["", "", "", ""]
        if k == 0:
            row += ["", ""]
    return row, [(s.az_deg, s.el_deg, s.dist_m, s.confidence) for s in out.sources]


def track_columns(tracks):
    row = [len(tracks)]
    for k in range(2):
        if k < len(tracks):
            t = tracks[k]
            row += [t.id, f"{t.az_deg:.2f}", f"{t.el_deg:.2f}", f"{t.dist_m:.3f}", f"{t.confidence:.3f}"]
        else:
            row += [""] * 5
    return row


def read_device_csv(path):
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        rows = list(csv.DictReader(f))
    geometry, fields = parse_csv_header(header)
    first = int(fields.get("first_frame", 0))
    return geometry, {int(r["frame"]) - first: r for r in rows}, fields.get("alg"), fields.get("trk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--csv", help="output CSV (default: <wav>_py.csv)")
    ap.add_argument("--config", help="daq-detector/main/config.h to take the geometry from")
    ap.add_argument("--compare", help="device CSV from replay mode (PLYnnnnn.CSV)")
    ap.add_argument("--alg", choices=["A", "B", "C"], default="A",
                    help="A = GCC-PHAT + least squares, B = SRP-PHAT, C = MUSIC on M1-M3 (azimuth only)")
    ap.add_argument("--track", choices=list(MODES), default=MODE_NAMES[DEFAULT_MODE],
                    help="tracker association: AZ = azimuth only, AZEL = azimuth x elevation, ELLIPSE = elliptical gate")
    args = ap.parse_args()

    samples, fs = load_recording(args.wav)
    device = None
    alg, trk = args.alg, args.track
    if args.compare:
        geometry, device, dev_alg, dev_trk = read_device_csv(args.compare)
        alg = dev_alg if dev_alg in ("A", "B", "C") else alg
        trk = dev_trk if dev_trk in MODES else trk
    else:
        geometry = load_geometry(args.config)
    if fs != geometry.fs:
        sys.exit(f"sample rate {fs} != geometry fs {geometry.fs}")

    est = TdoaEstimator(geometry)
    srp = SrpPhat(geometry, est.max_lag) if alg == "B" else None
    music = Music(geometry) if alg == "C" else None
    frames = est.run(samples)
    out = Path(args.csv) if args.csv else Path(args.wav).with_name(Path(args.wav).stem + f"_py{alg}.csv")
    n_voice = 0
    results = {}
    tracked = {}
    with open(out, "w", newline="", encoding="utf-8") as f:
        mics = ";".join(",".join(f"{v * 1000:.1f}" for v in m) for m in geometry.mics_m)
        f.write(f"# daq-sim reference wav={Path(args.wav).name} first_frame=0 fs={geometry.fs} c={geometry.c:.1f} "
                f"alg={alg} trk={trk} mics={mics}\n")
        w = csv.writer(f)
        w.writerow(TDOA_COLUMNS + LOC_COLUMNS + TRACK_COLUMNS)
        tracker = Tracker(geometry.fs, MODES[trk])
        for r in frames:
            row = [r.frame, int(r.voice), f"{r.energy_db:.2f}"]
            row += [f"{t * 1e6:.3f}" for t in r.tau_s] + [f"{p:.4f}" for p in r.peak]
            sources, loc = [], None
            if alg == "C":
                loc, sources = localize_c(music, r)
            elif r.voice:
                loc, sources = localize_a(geometry, r) if alg == "A" else localize_b(srp, r)
            if r.voice:
                n_voice += 1
                results[r.frame] = dict(zip(LOC_COLUMNS, [str(v) for v in loc]))
                row += loc
            else:
                row += [""] * len(LOC_COLUMNS)
            trow = track_columns(tracker.update(r.frame, sources))
            tracked[r.frame] = dict(zip(TRACK_COLUMNS, [str(v) for v in trow]))
            row += trow
            w.writerow(row)
    print(f"{len(frames)} frames, {n_voice} with voice, algorithm {alg}, tracker {trk} -> {out}")
    kinds = [v["kind"] for v in results.values()]
    print(f"results: {kinds.count('P')} positions, {kinds.count('D')} directions only, {kinds.count('-')} none")
    counts = [int(v["tracks"]) for v in tracked.values()]
    print("tracked speakers per frame: " + ", ".join(f"{n}: {counts.count(n)}" for n in sorted(set(counts))))

    if device is not None:
        diffs = []
        for r in frames:
            d = device.get(r.frame)
            if d is None or not r.voice or d["vad"] != "1":
                continue
            dev = np.array([float(d[f"t{p}"]) for p in PAIR_NAMES])
            diffs.append(np.abs(dev - r.tau_s * 1e6))
        if not diffs:
            sys.exit("no common voice frames to compare")
        diffs = np.array(diffs)
        print(f"compared {len(diffs)} frames x 15 pairs: max |dt| = {diffs.max():.3f} us, "
              f"mean = {diffs.mean():.4f} us -> {'OK' if diffs.max() <= 1.0 else 'DIFFERENT'}")

        # localization results: same kind, directions and distances
        pos, kind_diff = [], 0
        for frame, py in results.items():
            d = device.get(frame)
            if d is None or d.get("vad") != "1" or "kind" not in d:
                continue
            if d["kind"] != py["kind"] or d.get("speakers") != py["speakers"]:
                kind_diff += 1
                continue
            for a, e, dist in (("az_deg", "el_deg", "dist_m"), ("az2_deg", "el2_deg", "dist2_m")):
                if py.get(a):
                    daz = (float(d[a]) - float(py[a]) + 180.0) % 360.0 - 180.0
                    pos.append((abs(daz), abs(float(d[e]) - float(py[e])), abs(float(d[dist]) - float(py[dist]))))
        if pos or kind_diff:
            p = np.array(pos) if pos else np.zeros((1, 3))
            ok = kind_diff == 0 and p[:, 0].max() <= 0.05 and p[:, 1].max() <= 0.05 and p[:, 2].max() <= 0.005
            print(f"compared {len(pos)} sources ({kind_diff} frames with different kind/speakers): "
                  f"max |d az| = {p[:, 0].max():.3f} deg, |d el| = {p[:, 1].max():.3f} deg, "
                  f"|d dist| = {p[:, 2].max() * 1000:.1f} mm -> {'OK' if ok else 'DIFFERENT'}")
        # tracker: same number of tracks and ids, same directions / distances
        tdiff, tpos = 0, []
        for frame, py in tracked.items():
            d = device.get(frame)
            if d is None or "tracks" not in d:
                continue
            if d["tracks"] != py["tracks"] or d["t1_id"] != py["t1_id"] or d["t2_id"] != py["t2_id"]:
                tdiff += 1
                continue
            for k in ("t1", "t2"):
                if py[f"{k}_id"]:
                    daz = (float(d[f"{k}_az_deg"]) - float(py[f"{k}_az_deg"]) + 180.0) % 360.0 - 180.0
                    tpos.append((abs(daz), abs(float(d[f"{k}_el_deg"]) - float(py[f"{k}_el_deg"])),
                                 abs(float(d[f"{k}_dist_m"]) - float(py[f"{k}_dist_m"]))))
        if tracked and any("tracks" in d for d in device.values()):
            p = np.array(tpos) if tpos else np.zeros((1, 3))
            ok = tdiff == 0 and p[:, 0].max() <= 0.05 and p[:, 1].max() <= 0.05 and p[:, 2].max() <= 0.005
            print(f"compared tracks in {len(tracked)} frames ({tdiff} frames with different tracks/ids): "
                  f"max |d az| = {p[:, 0].max():.3f} deg, |d el| = {p[:, 1].max():.3f} deg, "
                  f"|d dist| = {p[:, 2].max() * 1000:.1f} mm -> {'OK' if ok else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
