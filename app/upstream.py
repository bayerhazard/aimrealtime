"""OpenAI-compatible clients for the AIM speech and chat apps.

All three upstreams are called over the Olares appid routes with the same
bearer-free in-cluster access the Router uses. Credentials are not needed;
the services are reachable pod-to-pod.
"""

import json
import logging

import aiohttp

from audio import resample, pcm_to_wav, wav_to_pcm

log = logging.getLogger("aimrealtime.upstream")


class UpstreamError(RuntimeError):
    pass


class Upstreams:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=110)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise UpstreamError("client session not started")
        return self._session

    async def transcribe(self, pcm24k: bytes, model: str) -> str:
        wav = pcm_to_wav(resample(pcm24k, 24000, 16000), rate=16000)
        form = aiohttp.FormData()
        form.add_field("model", model or self.cfg.stt_model)
        form.add_field("file", wav, filename="segment.wav", content_type="audio/wav")
        try:
            async with self.session.post(f"{self.cfg.stt_url}/audio/transcriptions", data=form) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise UpstreamError(f"stt {resp.status}: {body[:200]}")
                data = json.loads(body)
                return (data.get("text") or "").strip()
        except aiohttp.ClientError as exc:
            raise UpstreamError(f"stt transport: {exc}") from exc

    async def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        payload: dict = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 700,
        }
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t}
                for t in tools
                if isinstance(t, dict) and t.get("name")
            ]
            payload["tool_choice"] = "auto"
        try:
            async with self.session.post(f"{self.cfg.llm_url}/chat/completions", json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning(
                        "llm error %s roles=%s tools=%d body=%s",
                        resp.status,
                        [m.get("role") for m in messages],
                        len(payload.get("tools") or []),
                        body[:300],
                    )
                    raise UpstreamError(f"llm {resp.status}: {body[:200]}")
                data = json.loads(body)
                return data["choices"][0]["message"] or {}
        except aiohttp.ClientError as exc:
            raise UpstreamError(f"llm transport: {exc}") from exc

    async def tts(self, text: str, voice: str) -> bytes:
        payload = {"model": self.cfg.tts_model, "input": text, "voice": voice, "response_format": "wav"}
        try:
            async with self.session.post(f"{self.cfg.tts_url}/audio/speech", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise UpstreamError(f"tts {resp.status}: {body[:200]}")
                data = await resp.read()
                ctype = resp.headers.get("content-type", "")
        except aiohttp.ClientError as exc:
            raise UpstreamError(f"tts transport: {exc}") from exc
        if "wav" not in ctype and not data[:4] == b"RIFF":
            raise UpstreamError(f"tts: unexpected content type {ctype!r}")
        pcm, rate = wav_to_pcm(data)
        return resample(pcm, rate, 24000)
