"""Evaluation of algorithms A / B / C and the tracker on simulated test scenarios T1-T5 (project documentation, 8.5).

    python tools/evaluate.py                 # quick set (14 scenarios), report in output/eval/
    python tools/evaluate.py --full          # full set (T1: 8 azimuths x 4 distances x 2 heights, T2: 4 angles, ...)
    python tools/evaluate.py --only T2 T5    # selected tests
    python tools/evaluate.py --no-sweep      # metrics with the current thresholds only

Scenarios are rendered with daq_sim.simulate (image-source room), each algorithm runs on the same frames, then:
- direction / distance errors (mean and 90th percentile, degrees / metres) of the per-frame sources that match an
  active speaker (nearest one within 20 deg; azimuth only for C); distance only for positions (kind P),
- false sources: share of per-frame sources farther than 20 deg from every active speaker (reflections, ghosts,
  interferers),
- detection: share of voice frames with a source within 20 deg of an active speaker,
- counting: share of frames where the number of confirmed tracks equals the number of speakers active in the last
  1.5 s (the tracker's hold time), from 1 s after the first speech onward; also per 1 s window (mode of the tracked
  count vs the true count at the window end).
Threshold sweep (cheap, on cached per-frame results): tracker mode x BIRTH_HIST x CONFIRM_HITS for every algorithm,
and MDL_PENALTY of algorithm C. Nothing is changed in the code: the report lists the best settings per test so the
team can decide. Simulation only; repeat on real recordings (T1-T5) before changing defaults.
"""
import argparse
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import daq_sim.music as music_mod  # noqa: E402
import daq_sim.track as track_mod  # noqa: E402
from daq_sim.geometry import default_geometry  # noqa: E402
from daq_sim.music import Music  # noqa: E402
from daq_sim.simulate import Room, render  # noqa: E402
from daq_sim.srp import SrpPhat, angular_distance_deg  # noqa: E402
from daq_sim.tdoa import HOP, TdoaEstimator  # noqa: E402
from localize_wav import localize_a, localize_b, localize_c  # noqa: E402

SECONDS = 6.0
ARRAY_HEIGHT_M = 1.2
LAB = dict(size=(10.0, 10.0, 3.0), rt60=0.3)            # T1-T3, T5: room large enough for 4 m in every direction
OFFICE = dict(size=(4.0, 3.5, 2.7), rt60=0.3)            # T4 small office
HALL = dict(size=(12.0, 9.0, 4.0), rt60=1.0)             # T4 lecture hall
HIT_DEG = 20.0
HOLD_S = track_mod.CONFIRMED_TIMEOUT_S


@dataclass
class Speaker:
    az: float
    el: float
    dist: float
    offset_s: float = 0.0      # shift of the talk-spurt schedule (conversation)


@dataclass
class Scenario:
    name: str
    test: str
    speakers: list
    room: dict
    interferers: list = field(default_factory=list)   # (kind, az, el, dist, level rel. to speech)
    array_xy: tuple = None                            # array position in the room (default: centre)


# ---------------------------------------------------------------- signals

def talk_mask(n, fs, offset_s):
    t = np.arange(n) / fs - offset_s
    return (t >= 0.5) & (((t - 0.5) % 1.3) < 0.8)


def speech(n, fs, rng, offset_s):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < 300) | (f > 3500)] = 0
    x = np.fft.irfft(spec, n)
    t = np.arange(n) / fs
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t + rng.uniform(0, 2 * np.pi)) ** 2
    return x * env * talk_mask(n, fs, offset_s) / np.max(np.abs(x))


def interferer(kind, n, fs, rng):
    t = np.arange(n) / fs
    f = np.fft.rfftfreq(n, 1 / fs)
    if kind == "ventilation":                       # continuous low-frequency noise, 1/f above 50 Hz
        spec = np.fft.rfft(rng.standard_normal(n))
        spec[f < 50] = 0
        spec[f >= 50] /= np.sqrt(f[f >= 50] / 50.0)
        spec[f > 4000] = 0
        x = np.fft.irfft(spec, n)
    elif kind == "music":                           # chords changing every 0.5 s, harmonics up to 4 kHz
        x = np.zeros(n)
        notes = 220.0 * 2 ** (np.array([0, 4, 7, 12, 5, 9, 2, 7]) / 12)
        for i, start in enumerate(range(0, n, int(0.5 * fs))):
            seg = slice(start, min(n, start + int(0.5 * fs)))
            for root in (notes[i % 8], notes[(i + 2) % 8]):
                for h in range(1, 12):
                    if root * h < 4000:
                        x[seg] += np.sin(2 * np.pi * root * h * t[seg]) / h
    elif kind == "steps":                           # broadband clicks every 0.6 s, 30 ms decay
        x = np.zeros(n)
        for start in range(int(0.3 * fs), n, int(0.6 * fs)):
            m = min(n - start, int(0.1 * fs))
            x[start:start + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (0.03 * fs))
    else:
        raise ValueError(kind)
    return x / np.max(np.abs(x))


# ---------------------------------------------------------------- scenarios

def scenarios(full):
    out = []
    azs = range(0, 360, 45) if full else (0, 135, 270)
    dists = (1.0, 2.0, 3.0, 4.0) if full else (1.0, 3.0)
    heights = (1.2, 1.6) if full else (1.6,)
    for az, d, h in itertools.product(azs, dists, heights):
        el = float(np.degrees(np.arctan2(h - ARRAY_HEIGHT_M, d)))
        out.append(Scenario(f"T1 az{az} d{d:g} h{h:g}", "T1", [Speaker(az, el, d)], LAB))
    for delta in ((20, 45, 90, 180) if full else (45, 180)):
        out.append(Scenario(f"T2 {delta}deg", "T2", [Speaker(60, 0, 2.0), Speaker(60 + delta, 0, 2.0)], LAB))
    out.append(Scenario("T3 conversation", "T3",
                        [Speaker(30, 5, 2.0, 0.0), Speaker(150, 0, 2.5, 0.45), Speaker(270, 10, 1.5, 0.9)], LAB))
    out.append(Scenario("T4 office RT60 0.3", "T4", [Speaker(60, 10, 1.5)], OFFICE))
    out.append(Scenario("T4 hall RT60 1.0", "T4", [Speaker(60, 5, 3.0)], HALL))
    for kind, pos, level in (("ventilation", (0, 60, 1.6), 0.3), ("music", (300, 0, 3.0), 0.25), ("steps", (200, -40, 1.8), 0.6)):
        out.append(Scenario(f"T5 {kind}", "T5", [Speaker(60, 5, 2.0)], LAB, [(kind, *pos, level)]))
    return out


def render_scenario(g, sc, seed):
    fs = g.fs
    n = int(SECONDS * fs)
    rng = np.random.default_rng(seed)
    size = sc.room["size"]
    pos = (size[0] / 2, size[1] / 2, ARRAY_HEIGHT_M) if sc.array_xy is None else (*sc.array_xy, ARRAY_HEIGHT_M)
    room = Room(tuple(size), pos, sc.room["rt60"])
    srcs = [(s.az, s.el, s.dist, speech(n, fs, rng, s.offset_s)) for s in sc.speakers]
    srcs += [(az, el, d, interferer(kind, n, fs, rng) * lvl) for kind, az, el, d, lvl in sc.interferers]
    samples = render(g, srcs, SECONDS, room=room, seed=seed)
    active = np.stack([talk_mask(n, fs, s.offset_s) for s in sc.speakers])   # (speakers, n)
    return samples, active


# ---------------------------------------------------------------- evaluation

def run_algorithms(g, frames, algs):
    """Per-frame sources [(az, el, dist, conf, kind)] for each algorithm (empty list = no result / no voice)."""
    est_lags = [g.max_lag_samples(i, j) for i, j in itertools.combinations(range(6), 2)]
    srp = SrpPhat(g, est_lags) if "B" in algs else None
    out = {}
    for alg in algs:
        music = Music(g) if alg == "C" else None
        per = []
        for r in frames:
            if alg == "C":
                row, src = localize_c(music, r)
            elif r.voice:
                row, src = localize_a(g, r) if alg == "A" else localize_b(srp, r)
            else:
                row, src = None, []
            kind = row[1] if row else "-"
            per.append([(a, e, d, c, kind if d > 0 else "D") for a, e, d, c in src])
        out[alg] = per
    return out


def truth_at(active, frame_start, fs, hold_s=0.0):
    """Speakers active at the frame centre (hold_s = 0) or within the last hold_s seconds."""
    c = frame_start + 1024
    lo = max(0, int(c - hold_s * fs))
    return [k for k in range(active.shape[0]) if active[k, lo:c + 1].any()]


def direction_errors(sc, frames, sources, active, fs, alg):
    az_err, el_err, dist_err, hits, voiced, n_src, n_false = [], [], [], 0, 0, 0, 0
    for r, src in zip(frames, sources):
        act = truth_at(active, r.frame, fs)
        if not r.voice or not act:
            continue
        voiced += 1
        hit = False
        for a, e, d, _, kind in src:
            def dist_to(k):
                s = sc.speakers[k]
                return abs((a - s.az + 180) % 360 - 180) if alg == "C" else angular_distance_deg(a, e, s.az, s.el)
            k = min(act, key=dist_to)
            s = sc.speakers[k]
            n_src += 1
            if dist_to(k) > HIT_DEG:
                n_false += 1
                continue
            hit = True
            az_err.append(abs((a - s.az + 180) % 360 - 180))
            if alg != "C":
                el_err.append(abs(e - s.el))
            if kind == "P" and d > 0:
                dist_err.append(abs(d - s.dist))
        hits += hit
    stat = lambda v: (float(np.mean(v)), float(np.percentile(v, 90))) if v else (None, None)   # noqa: E731
    return dict(az=stat(az_err), el=stat(el_err), dist=stat(dist_err),
                detect=hits / voiced if voiced else None, false=n_false / n_src if n_src else None, n_sources=n_src)


def counting(speakers, frames, sources, active, fs, mode, birth, confirm):
    track_mod.BIRTH_HIST, track_mod.CONFIRM_HITS = birth, confirm
    tr = track_mod.Tracker(fs, mode)
    onset = int(np.argmax(active.any(axis=0)))
    ok = total = 0
    win, wins_ok, wins = [], 0, 0
    win_len = int(round(fs / HOP))
    tracked_err = []
    for r, src in zip(frames, sources):
        tracks = tr.update(r.frame, [(a, e, d, c) for a, e, d, c, _ in src])
        if r.frame < onset + fs:
            continue
        truth = truth_at(active, r.frame, fs, HOLD_S)
        total += 1
        ok += len(tracks) == len(truth)
        win.append(len(tracks))
        if len(win) == win_len:
            wins += 1
            vals, cnt = np.unique(win, return_counts=True)
            wins_ok += vals[np.argmax(cnt)] == len(truth)
            win = []
        for t in tracks:
            if truth:
                tracked_err.append(min(abs((t.az_deg - speakers[k].az + 180) % 360 - 180) for k in truth))
    return dict(frames=ok / total if total else None, windows=wins_ok / wins if wins else None,
                track_az=float(np.mean(tracked_err)) if tracked_err else None)


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full scenario set (slower)")
    ap.add_argument("--only", nargs="+", help="tests to run, e.g. T2 T5")
    ap.add_argument("--alg", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--out", default="output/eval")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    g = default_geometry()
    scs = [s for s in scenarios(args.full) if not args.only or s.test in args.only]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    modes = list(track_mod.MODES.items())
    births = (1.0, 1.5, 2.5)
    confirms = (2, 4, 6)
    penalties = (0.5, 1.0, 1.5, 2.0, 3.0)
    base = dict(birth=track_mod.BIRTH_HIST, confirm=track_mod.CONFIRM_HITS, penalty=music_mod.MDL_PENALTY)

    results = []
    for i, sc in enumerate(scs):
        samples, active = render_scenario(g, sc, args.seed + i)
        frames = TdoaEstimator(g).run(samples)
        srcs = run_algorithms(g, frames, args.alg)
        entry = dict(name=sc.name, test=sc.test, n_speakers=len(sc.speakers), algs={})
        for alg in args.alg:
            e = dict(direction=direction_errors(sc, frames, srcs[alg], active, g.fs, alg))
            e["count"] = {name: counting(sc.speakers, frames, srcs[alg], active, g.fs, code, base["birth"], base["confirm"])
                          for name, code in modes}
            if not args.no_sweep:
                e["sweep"] = [dict(mode=name, birth=b, confirm=c, **counting(sc.speakers, frames, srcs[alg], active, g.fs, code, b, c))
                              for (name, code), b, c in itertools.product(modes, births, confirms)]
            entry["algs"][alg] = e
        if "C" in args.alg and not args.no_sweep:
            sweep = []
            for pen in penalties:
                music_mod.MDL_PENALTY = pen
                m = Music(g)
                counts = [o.count for o in (m.process(r) for r in frames) if o is not None and o.count]
                per = run_algorithms(g, frames, ["C"])["C"]
                d = direction_errors(sc, frames, per, active, g.fs, "C")
                cnt = counting(sc.speakers, frames, per, active, g.fs, track_mod.MODES["AZ"], base["birth"], base["confirm"])
                sweep.append(dict(penalty=pen, mdl_two=float(np.mean(np.array(counts) == 2)) if counts else None,
                                  az_mean=d["az"][0], false=d["false"], detect=d["detect"], count_frames=cnt["frames"]))
            music_mod.MDL_PENALTY = base["penalty"]
            entry["music_sweep"] = sweep
        track_mod.BIRTH_HIST, track_mod.CONFIRM_HITS = base["birth"], base["confirm"]
        results.append(entry)
        a = entry["algs"]
        print(f"[{i + 1}/{len(scs)}] {sc.name}: " + " | ".join(
            f"{alg} az {fmt(a[alg]['direction']['az'][0])}° det {fmt(a[alg]['direction']['detect'], 2)} "
            f"false {fmt(a[alg]['direction']['false'], 2)} "
            f"cnt(AZ) {fmt(a[alg]['count']['AZ']['frames'], 2)}" for alg in args.alg), flush=True)

    (outdir / "results.json").write_text(json.dumps(results, indent=1))
    write_report(results, args, base, outdir / "report.md")
    print(f"report: {outdir / 'report.md'}")


def write_report(results, args, base, path):
    L = ["# Evaluation on simulated scenarios", "",
         f"Set: {'full' if args.full else 'quick'}, {SECONDS:g} s per scenario, array at {ARRAY_HEIGHT_M} m, seed {args.seed}.",
         f"Current thresholds: tracker BIRTH_HIST {base['birth']}, CONFIRM_HITS {base['confirm']}, MUSIC MDL_PENALTY {base['penalty']}.",
         "Errors: mean / p90 of the sources within 20° of an active speaker. False: share of sources farther than 20° "
         "from every active speaker. Detection: voice frames with a source within 20° of an active speaker. "
         "Counting: frames (and 1 s windows) where confirmed tracks = speakers active in the last 1.5 s.", ""]
    for alg in args.alg:
        L += [f"## Algorithm {alg}", "",
              "| Scenario | Azimuth [°] | Elevation [°] | Distance [m] | Detection | False | Count AZ | Count AZEL | Count ELLIPSE | Count AZ (1 s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for e in results:
            d, c = e["algs"][alg]["direction"], e["algs"][alg]["count"]
            L.append(f"| {e['name']} | {fmt(d['az'][0])} / {fmt(d['az'][1])} | {fmt(d['el'][0])} / {fmt(d['el'][1])} | "
                     f"{fmt(d['dist'][0], 2)} / {fmt(d['dist'][1], 2)} | {fmt(d['detect'], 2)} | {fmt(d['false'], 2)} | {fmt(c['AZ']['frames'], 2)} | "
                     f"{fmt(c['AZEL']['frames'], 2)} | {fmt(c['ELLIPSE']['frames'], 2)} | {fmt(c['AZ']['windows'], 2)} |")
        L.append("")
        if not args.no_sweep:
            L += [f"Tracker sweep (algorithm {alg}): mean counting accuracy per test, best settings", "",
                  "| Test | Current (AZ) | Best | Mode | BIRTH_HIST | CONFIRM_HITS |", "|---|---|---|---|---|---|"]
            for test in sorted({e["test"] for e in results}):
                es = [e for e in results if e["test"] == test]
                cur = np.mean([x["algs"][alg]["count"]["AZ"]["frames"] or 0 for x in es])
                combos = {}
                for x in es:
                    for s in x["algs"][alg]["sweep"]:
                        combos.setdefault((s["mode"], s["birth"], s["confirm"]), []).append(s["frames"] or 0)
                (m, b, c), v = max(combos.items(), key=lambda kv: np.mean(kv[1]))
                L.append(f"| {test} | {cur:.2f} | {np.mean(v):.2f} | {m} | {b} | {c} |")
            combos = {}
            for x in results:
                for s in x["algs"][alg]["sweep"]:
                    combos.setdefault((s["mode"], s["birth"], s["confirm"]), []).append(s["frames"] or 0)
            ranked = sorted(combos.items(), key=lambda kv: -np.mean(kv[1]))[:5]
            L += ["", "Best over all tests: " + "; ".join(f"{m} / {b} / {c}: {np.mean(v):.2f}" for (m, b, c), v in ranked), ""]
    if "C" in args.alg and not args.no_sweep:
        L += ["## MUSIC: MDL_PENALTY", "", "Per penalty: share of results with K = 2 / azimuth error of matched sources / false sources / counting (AZ).", "",
              "| Scenario | " + " | ".join(f"pen {s['penalty']}" for s in results[0]["music_sweep"]) + " |",
              "|---|" + "---|" * len(results[0]["music_sweep"])]
        for e in results:
            L.append(f"| {e['name']} ({e['n_speakers']} spk) | " + " | ".join(
                f"{fmt(s['mdl_two'], 2)} / {fmt(s['az_mean'])} / {fmt(s['false'], 2)} / {fmt(s['count_frames'], 2)}"
                for s in e["music_sweep"]) + " |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
