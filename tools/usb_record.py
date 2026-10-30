"""Receive the 6-channel stream from daq-detector over native USB and save it as WAV.

Usage:
    python tools/usb_record.py --port COM7 --out data/usb_test.wav --seconds 30

The device starts streaming after receiving 'S' and stops after 'X'.
Lost packets (gaps in `seq`) are replaced with silence so the timeline stays aligned.
"""
import argparse
import struct
import time
import wave

import numpy as np
import serial

MAGIC = 0x31514144  # "DAQ1"
HEADER = struct.Struct("<IIQIHH")  # magic, seq, first_frame, sample_rate, channels, frames
MAGIC_BYTES = struct.pack("<I", MAGIC)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="native USB port of the ESP32-S3 (e.g. COM7, /dev/ttyACM0)")
    p.add_argument("--out", required=True, help="output WAV path")
    p.add_argument("--seconds", type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    ser = serial.Serial(args.port, timeout=0.1)  # baud rate is ignored by USB CDC
    ser.reset_input_buffer()
    ser.write(b"S")

    buf = bytearray()
    wav = None
    expected_seq = None
    stats = {"packets": 0, "lost": 0, "resyncs": 0, "bytes": 0}
    sum_sq = None
    level_frames = 0
    t_start = time.monotonic()
    t_report = t_start

    try:
        while time.monotonic() - t_start < args.seconds:
            buf += ser.read(ser.in_waiting or 1)

            while True:
                idx = buf.find(MAGIC_BYTES)
                if idx < 0:
                    del buf[:-3]  # keep a possible partial magic
                    break
                if idx > 0:
                    stats["resyncs"] += 1
                    del buf[:idx]
                if len(buf) < HEADER.size:
                    break
                _, seq, _, rate, channels, frames = HEADER.unpack_from(buf)
                payload_len = frames * channels * 2
                if channels == 0 or channels > 8 or frames == 0 or frames > 4096:
                    del buf[:4]  # false magic match
                    stats["resyncs"] += 1
                    continue
                if len(buf) < HEADER.size + payload_len:
                    break
                payload = bytes(buf[HEADER.size:HEADER.size + payload_len])
                del buf[:HEADER.size + payload_len]

                if wav is None:
                    wav = wave.open(args.out, "wb")
                    wav.setnchannels(channels)
                    wav.setsampwidth(2)
                    wav.setframerate(rate)
                    sum_sq = np.zeros(channels)
                if expected_seq is not None and seq != expected_seq:
                    gap = (seq - expected_seq) & 0xFFFFFFFF
                    stats["lost"] += gap
                    wav.writeframes(bytes(gap * payload_len))  # silence keeps the timeline aligned
                expected_seq = (seq + 1) & 0xFFFFFFFF

                wav.writeframes(payload)
                stats["packets"] += 1
                stats["bytes"] += payload_len
                block = np.frombuffer(payload, dtype="<i2").reshape(frames, channels).astype(np.float64)
                sum_sq += (block ** 2).sum(axis=0)
                level_frames += frames

            now = time.monotonic()
            if now - t_report >= 1.0 and level_frames:
                dbfs = 20 * np.log10(np.sqrt(sum_sq / level_frames) / 32768 + 1e-12)
                rate_kbs = stats["bytes"] / (now - t_start) / 1e3
                levels = " ".join(f"M{i + 1}:{v:6.1f}" for i, v in enumerate(dbfs))
                print(f"{rate_kbs:6.0f} kB/s | {levels} | lost={stats['lost']} resync={stats['resyncs']}")
                sum_sq[:] = 0
                level_frames = 0
                t_report = now
    finally:
        ser.write(b"X")
        ser.close()
        if wav is not None:
            wav.close()

    print(f"done: {stats['packets']} packets, lost {stats['lost']}, resyncs {stats['resyncs']} -> {args.out}")


if __name__ == "__main__":
    main()
