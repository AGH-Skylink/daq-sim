"""Loading of multichannel WAV recordings made by daq-detector (SD card or USB)."""
import struct
from pathlib import Path

import numpy as np


def load_recording(path):
    """Return (samples, sample_rate); samples is an int16 array of shape (frames, channels).

    Files from the SD card keep data_size = 0 in the header until they are closed
    (e.g. after a power loss), so the data length falls back to the file size.
    """
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")

    channels = sample_rate = None
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id, chunk_size = struct.unpack_from("<4sI", raw, pos)
        body = pos + 8
        if chunk_id == b"fmt ":
            _, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", raw, body)
            if bits != 16:
                raise ValueError(f"{path}: expected 16-bit PCM, got {bits}")
        elif chunk_id == b"data":
            if channels is None:
                raise ValueError(f"{path}: data chunk before fmt chunk")
            available = len(raw) - body
            size = chunk_size if 0 < chunk_size <= available else available
            size -= size % (2 * channels)  # drop a trailing partial frame
            samples = np.frombuffer(raw, dtype="<i2", count=size // 2, offset=body)
            return samples.reshape(-1, channels), sample_rate
        pos = body + chunk_size + (chunk_size & 1)

    raise ValueError(f"{path}: no data chunk")
