"""OpenAI-compatible clients for the AIM speech and chat apps.

All three upstreams are called over the Olares appid routes with the same
bearer-free in-cluster access the Router uses. Credentials are not needed;
the services are reachable pod-to-pod.
"""

import json
import logging
import uuid

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
        # Qwen3.8 on this vLLM stack is fragile around tool calling: with
        # enable_thinking=false it degenerates into repetition loops once the
        # prefix cache has seen the prompt, and non-zero sampling loops inside
        # the thinking block. Verified stable: thinking on (default), greedy,
        # a generous token budget, and a unique prefix nonce per request.
        if messages and messages[0].get("role") == "system":
            messages = list(messages)
            messages[0] = dict(messages[0], content=f"[call {uuid.uuid4().hex[:10]}] " + str(messages[0].get("content") or ""))
        payload: dict = {
            "model": self.cfg.llm_model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.0,
        }
        cleaned = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = function.get("name")
            if not name:
                continue
            cleaned.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(name),
                        "description": str(function.get("description") or ""),
                        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        if cleaned:
            payload["tools"] = cleaned
            payload["tool_choice"] = "auto"
        log.debug("llm tools=%d temp=%s msgs=%d", len(cleaned), payload.get("temperature"), len(messages))
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
