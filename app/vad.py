"""Streaming energy-based VAD over 24 kHz PCM16.

Open WebUI's bridge sets turn_detection=server_vad on the provider, so the
provider owns the speech boundaries. A light RMS detector with an adaptive
noise floor and hangover is enough for browser-captured voice (the browser
already runs echo cancellation and noise suppression); it keeps the app free
of model downloads and GPU-adjacent dependencies.
"""

from audio import rms

FRAME_MS = 20
BYTES_PER_MS = 48  # 24000 Hz * 2 bytes / 1000


class EnergyVad:
    def __init__(
        self,
        min_rms: float = 140.0,
        noise_mult: float = 3.0,
        start_ms: int = 60,
        hangover_ms: int = 520,
        min_speech_ms: int = 220,
        max_segment_ms: int = 30000,
        preroll_ms: int = 240,
    ) -> None:
        self.min_rms = min_rms
        self.noise_mult = noise_mult
        self.start_frames = max(1, start_ms // FRAME_MS)
        self.hangover_frames = max(1, hangover_ms // FRAME_MS)
        self.min_speech_ms = min_speech_ms
        self.max_segment_bytes = max_segment_ms * BYTES_PER_MS
        self.preroll_bytes = preroll_ms * BYTES_PER_MS
        self.frame_bytes = FRAME_MS * BYTES_PER_MS

        self.t_ms = 0.0  # absolute end-of-processed-audio time
        self._partial = b""
        self._preroll = b""
        self._noise = 0.0
        self._speaking = False
        self._started = False
        self._above = 0
        self._below = 0
        self._segment = b""
        self._seg_start_ms = 0.0
        self._silence_start_ms = 0.0

    def _threshold(self) -> float:
        return max(self.min_rms, self._noise * self.noise_mult)

    def feed(self, pcm: bytes) -> list[dict]:
        events: list[dict] = []
        data = self._partial + pcm
        off = 0
        while off + self.frame_bytes <= len(data):
            frame = data[off : off + self.frame_bytes]
            off += self.frame_bytes
            events.extend(self._frame(frame))
        self._partial = data[off:]
        return events

    def flush(self) -> list[dict]:
        events = self._frame(self._partial, final=True) if self._partial else []
        self._partial = b""
        if self._started:
            events.append(self._end(max(self.t_ms, self._seg_start_ms + self.min_speech_ms)))
        return events

    def _frame(self, frame: bytes, final: bool = False) -> list[dict]:
        events: list[dict] = []
        level = rms(frame)
        speech = level > self._threshold()

        if not self._started:
            # track the noise floor only while idle
            self._noise = level if self._noise == 0.0 else 0.985 * self._noise + 0.015 * level
            if speech:
                self._above += 1
            else:
                self._above = 0
            if self._above >= self.start_frames:
                self._started = True
                self._speaking = True
                self._segment = self._preroll + frame
                self._seg_start_ms = self.t_ms - self._above * FRAME_MS
                events.append({"type": "start", "ms": self._seg_start_ms})
            else:
                self._preroll = (self._preroll + frame)[-self.preroll_bytes :]
        else:
            self._segment += frame
            if speech:
                self._below = 0
                self._speaking = True
            else:
                if self._speaking:
                    self._silence_start_ms = self.t_ms
                self._below += 1
                self._speaking = False

            if self._below >= self.hangover_frames or len(self._segment) >= self.max_segment_bytes or final:
                end_ms = self._silence_start_ms if self._below >= self.hangover_frames else self.t_ms
                seg = self._segment
                # trim the trailing hangover from the segment audio
                trim = max(0, int((self.t_ms - end_ms) * BYTES_PER_MS))
                if trim:
                    seg = seg[: max(0, len(seg) - trim)]
                events.append(self._end(end_ms, segment=seg))

        self.t_ms += FRAME_MS
        return events

    def _end(self, end_ms: float, segment: bytes | None = None) -> dict:
        seg = segment if segment is not None else self._segment
        duration = len(seg) / BYTES_PER_MS
        self._reset()
        if duration < self.min_speech_ms:
            return {"type": "drop"}
        return {"type": "end", "ms": end_ms, "pcm": seg, "duration_ms": duration}

    def _reset(self) -> None:
        self._started = False
        self._speaking = False
        self._above = 0
        self._below = 0
        self._segment = b""
        self._preroll = b""
        self._silence_start_ms = 0.0
