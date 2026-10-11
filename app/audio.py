"""PCM helpers: raw 24 kHz mono PCM16 as Open WebUI's bridge sends it."""

import io
import struct
import wave

RATE = 24000
BYTES_PER_SAMPLE = 2
SAMPLE_WIDTH = 2
CHANNELS = 1


def pcm_ms(payload: bytes) -> float:
    return len(payload) / BYTES_PER_SAMPLE / RATE * 1000.0


def duration_ms_to_bytes(ms: int) -> int:
    return int(RATE * ms / 1000.0) * BYTES_PER_SAMPLE


def rms(payload: bytes) -> float:
    if len(payload) < 2:
        return 0.0
    n = len(payload) // 2
    acc = 0
    for (v,) in struct.iter_unpack("<h", payload[: n * 2]):
        acc += v * v
    return (acc / n) ** 0.5


def pcm_to_wav(payload: bytes, rate: int = RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(rate)
        w.writeframes(payload)
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != SAMPLE_WIDTH:
            raise ValueError("unsupported sample width")
        rate = w.getframerate()
        channels = w.getnchannels()
        pcm = w.readframes(w.getnframes())
    if channels != 1:
        # de-interleave the first channel
        step = channels * SAMPLE_WIDTH
        pcm = b"".join(pcm[i : i + SAMPLE_WIDTH] for i in range(0, len(pcm), step))
    return pcm, rate


def resample(payload: bytes, rate: int, target: int) -> bytes:
    """Linear-interpolation resample between any two rates."""
    if rate == target:
        return payload
    src = struct.unpack("<%dh" % (len(payload) // 2), payload[: len(payload) // 2 * 2])
    if not src:
        return b""
    ratio = target / rate
    n_out = int(len(src) * ratio)
    out = bytearray()
    for i in range(n_out):
        pos = i / ratio
        i0 = int(pos)
        i1 = min(i0 + 1, len(src) - 1)
        frac = pos - i0
        v = int(src[i0] + (src[i1] - src[i0]) * frac)
        out += struct.pack("<h", max(-32768, min(32767, v)))
    return bytes(out)


def resample16(payload: bytes, rate: int = RATE) -> bytes:
    return resample(payload, rate, 16000)
