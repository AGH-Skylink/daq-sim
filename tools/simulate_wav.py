"""Make a synthetic 6-channel WAV (free field or shoebox room) with one or more sources.

    python tools/simulate_wav.py data/sim.wav --source 60 20 2.0 --seconds 5
    python tools/simulate_wav.py data/sim2.wav --source 60 20 2 --source 200 -5 3.5 --snr 20
    python tools/simulate_wav.py data/room.wav --source 60 20 2 --room 6 5 3 --array-pos 2.5 2 0.66 --rt60 0.5

--source AZ EL DIST: azimuth and elevation in degrees, distance in metres (array coordinates).
The file can be processed with tools/localize_wav.py or copied to the SD card for the device replay mode.
"""
import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daq_sim.geometry import load_geometry  # noqa: E402
from daq_sim.simulate import Room, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--source", nargs=3, type=float, action="append", metavar=("AZ", "EL", "DIST"), required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--snr", type=float, default=30.0, help="SNR per mic in dB")
    ap.add_argument("--config", help="daq-detector/main/config.h to take the geometry from")
    ap.add_argument("--room", nargs=3, type=float, metavar=("LX", "LY", "LZ"), help="shoebox room size in m (default: free field)")
    ap.add_argument("--array-pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="array origin in room coordinates, m (default: room centre, 0.66 m above the floor)")
    ap.add_argument("--rt60", type=float, default=0.5, help="reverberation time in s (with --room)")
    args = ap.parse_args()

    g = load_geometry(args.config)
    room = None
    if args.room:
        pos = args.array_pos or [args.room[0] / 2, args.room[1] / 2, 0.661]
        room = Room(tuple(args.room), tuple(pos), args.rt60)
    samples = render(g, [tuple(s) for s in args.source], args.seconds, snr_db=args.snr, room=room)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(6)
        w.setsampwidth(2)
        w.setframerate(g.fs)
        w.writeframes(samples.tobytes())
    print(f"{args.out}: {samples.shape[0]} frames, sources {args.source}")


if __name__ == "__main__":
    main()
