"""OpenAI-compatible clients for speech and chat.

Upstreams are addressed through the Router of this Olares, which resolves
the default categories (default-stt / default-chat / default-tts) to whatever
models this machine actually has. The Router reads the caller from the
platform edge, so no credential is involved. UPSTREAM_*_URL still overrides
that, but an override that does not answer here is dropped at startup.
"""

import json
import logging
import uuid

import aiohttp

from audio import resample, pcm_to_wav, wav_to_pcm
from router import RouterEndpoint

log = logging.getLogger("aimrealtime.upstream")

_ATTR = {"stt": "stt_url", "tts": "tts_url", "llm": "llm_url"}


class UpstreamError(RuntimeError):
    pass


def _fail(kind: str, status: int, body: str) -> UpstreamError:
    text = body[:200]
    if "no_default_model" in text:
        text = f"this Olares has no {kind} model behind its Router category ({text})"
    elif "model_not_ready" in text:
        text = f"the {kind} model is still loading (retry shortly)"
    return UpstreamError(f"{kind} {status}: {text}")


class Upstreams:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.router = RouterEndpoint()
        self.request_host = ""
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=120, connect=10, sock_read=110)
        self._session = aiohttp.ClientSession(timeout=timeout)
        for kind, attr in _ATTR.items():
            url = getattr(self.cfg, attr)
            if url and not await self.router.probe(url, {}):
                log.warning("%s upstream %s does not answer here, using the Router", kind, url)
                setattr(self.cfg, attr, "")
        await self.router.get(self.request_host)

    def fixed(self, kind: str) -> bool:
        """True when this kind points at an explicit upstream URL, False when the Router answers."""
        return bool(getattr(self.cfg, _ATTR[kind]))

    async def target(self, kind: str) -> tuple[str, dict[str, str]]:
        url = getattr(self.cfg, _ATTR[kind])
        if url:
            return url, {}
        base, headers = await self.router.get(self.request_host)
        if not base:
            raise UpstreamError(f"no reachable Router on this Olares (used for {kind})")
        return base, headers

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
        base, headers = await self.target("stt")
        form = aiohttp.FormData()
        form.add_field("model", model or self.cfg.stt_model)
        form.add_field("file", wav, filename="segment.wav", content_type="audio/wav")
        try:
            async with self.session.post(f"{base}/audio/transcriptions", data=form, headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise _fail("stt", resp.status, body)
                data = json.loads(body)
                return (data.get("text") or "").strip()
        except aiohttp.ClientError as exc:
            self.router.invalidate()
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
        base, headers = await self.target("llm")
        try:
            async with self.session.post(f"{base}/chat/completions", json=payload, headers=headers) as resp:
                body = await resp.text()
                if resp.status != 200:
                    log.warning(
                        "llm error %s roles=%s tools=%d body=%s",
                        resp.status,
                        [m.get("role") for m in messages],
                        len(payload.get("tools") or []),
                        body[:300],
                    )
                    raise _fail("llm", resp.status, body)
                data = json.loads(body)
                return data["choices"][0]["message"] or {}
        except aiohttp.ClientError as exc:
            self.router.invalidate()
            raise UpstreamError(f"llm transport: {exc}") from exc

    async def tts(self, text: str, voice: str) -> bytes:
        payload = {"model": self.cfg.tts_model, "input": text, "voice": voice, "response_format": "wav"}
        base, headers = await self.target("tts")
        try:
            async with self.session.post(f"{base}/audio/speech", json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise _fail("tts", resp.status, body)
                data = await resp.read()
                ctype = resp.headers.get("content-type", "")
        except aiohttp.ClientError as exc:
            self.router.invalidate()
            raise UpstreamError(f"tts transport: {exc}") from exc
        if "wav" not in ctype and not data[:4] == b"RIFF":
            raise UpstreamError(f"tts: unexpected content type {ctype!r}")
        pcm, rate = wav_to_pcm(data)
        return resample(pcm, rate, 24000)
