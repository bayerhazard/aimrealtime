"""AIM Realtime Voice — provider endpoint for Open WebUI 0.12 "Realtime" calls.

Speaks the subset of the OpenAI Realtime WebSocket protocol that Open WebUI's
bridge (backend/open_webui/routers/audio/realtime.py, v0.12.0) drives:

  inbound   session.update, input_audio_buffer.{append,commit,clear},
            response.create, response.cancel, conversation.item.truncate,
            conversation.item.create, conversation.item.delete
  outbound  session.created, session.updated, input_audio_buffer.speech_started,
            input_audio_buffer.speech_stopped,
            conversation.item.input_audio_transcription.completed,
            response.created, response.output_audio.delta, response.done, error

Phase 1: session handshake and a canned binaural-free beep response so the
WebSocket upgrade can be proven through the Olares ingress. STT/LLM/TTS
orchestration lands in Phase 2 (see docs/anleitung-aim-realtime.md).
"""

import asyncio
import base64
import json
import logging
import os
import struct
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

APP_VERSION = "26.10.1"

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


class Config:
    def __init__(self) -> None:
        self.advertised_model = env("ADVERTISED_MODEL", "aimrealtime")
        self.api_key = env("API_KEY")
        self.stt_url = env("UPSTREAM_STT_URL", "").rstrip("/")
        self.tts_url = env("UPSTREAM_TTS_URL", "").rstrip("/")
        self.llm_url = env("UPSTREAM_LLM_URL", "").rstrip("/")
        self.stt_model = env("STT_MODEL", "Qwen/Qwen3-ASR-1.7B")
        self.tts_model = env("TTS_MODEL", "omnivoice")
        self.llm_model = env("LLM_MODEL", "qwen3.8-27b")
        self.default_voice = env("DEFAULT_VOICE", "female")
        self.vad_threshold = float(env("VAD_THRESHOLD", "0.5"))
        self.max_calls = int(env("MAX_CALLS", "2"))
        self.log_level = env("LOG_LEVEL", "info").upper()

    def masked(self) -> dict:
        return {
            "advertised_model": self.advertised_model,
            "stt": self.stt_url or "(unset)",
            "tts": self.tts_url or "(unset)",
            "llm": self.llm_url or "(unset)",
            "stt_model": self.stt_model,
            "tts_model": self.tts_model,
            "llm_model": self.llm_model,
            "default_voice": self.default_voice,
            "api_key_set": bool(self.api_key),
            "max_calls": self.max_calls,
        }


CFG = Config()
logging.basicConfig(level=getattr(logging, CFG.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("aimrealtime")

app = FastAPI(title="AIM Realtime Voice", version=APP_VERSION)
ACTIVE_CALLS = 0


def beep_pcm(ms: int = 220, freq: float = 440.0, amplitude: float = 0.18) -> bytes:
    n = int(SAMPLE_RATE * ms / 1000)
    out = bytearray()
    for i in range(n):
        # linear fade in/out to keep the tone click-free
        fade = min(1.0, i / (SAMPLE_RATE * 0.01), (n - i) / (SAMPLE_RATE * 0.01))
        v = amplitude * fade * (1.0 if int(freq * i / SAMPLE_RATE) % 2 == 0 else -1.0)
        out += struct.pack("<h", int(v * 32767))
    return bytes(out)


def pcm_ms(payload: bytes) -> float:
    return len(payload) / BYTES_PER_SAMPLE / SAMPLE_RATE * 1000.0


class Call:
    """One OWUI voice call: protocol state for a single WebSocket."""

    def __init__(self, ws: WebSocket, model: str) -> None:
        self.ws = ws
        self.model = model
        self.session_id = "sess_" + uuid.uuid4().hex[:16]
        self.voice = CFG.default_voice
        self.instructions = ""
        self.transcription_model = CFG.stt_model
        self.tools = []
        self.tool_choice = "auto"
        self.responses: dict[str, dict] = {}
        self.pcm_in = 0
        self.item_counter = 0

    async def send(self, event: dict) -> None:
        await self.ws.send_text(json.dumps(event, separators=(",", ":")))

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(
            {
                "type": "session.created",
                "session": {
                    "object": "realtime.session",
                    "id": self.session_id,
                    "model": self.model or CFG.advertised_model,
                    "output_modalities": ["audio"],
                },
            }
        )
        log.info("call open session=%s model=%s", self.session_id, self.model)
        while True:
            raw = await self.ws.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self.error("invalid_event", "Malformed JSON event")
                continue
            kind = event.get("type", "")
            handler = getattr(self, f"on_{kind.replace('.', '_')}", None)
            if handler is None:
                log.debug("ignored event type=%s", kind)
                continue
            await handler(event)

    async def error(self, code: str, message: str) -> None:
        await self.send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": message}})

    # -- inbound events ----------------------------------------------------

    async def on_session_update(self, event: dict) -> None:
        session = event.get("session") or {}
        if isinstance(session.get("instructions"), str):
            self.instructions = session["instructions"]
        audio = session.get("audio") or {}
        out_voice = ((audio.get("output") or {}).get("voice"))
        if isinstance(out_voice, str) and out_voice.strip():
            self.voice = out_voice.strip()
        tr = (((audio.get("input") or {}).get("transcription")) or {})
        if isinstance(tr.get("model"), str) and tr["model"].strip():
            self.transcription_model = tr["model"].strip()
        if isinstance(session.get("tools"), list):
            self.tools = session["tools"]
        if isinstance(session.get("tool_choice"), str):
            self.tool_choice = session["tool_choice"]
        await self.send(
            {
                "type": "session.updated",
                "session": {
                    "object": "realtime.session",
                    "id": self.session_id,
                    "model": self.model or CFG.advertised_model,
                    "instructions": self.instructions,
                    "voice": self.voice,
                    "transcription_model": self.transcription_model,
                    "tools": self.tools,
                },
            }
        )

    async def on_input_audio_buffer_append(self, event: dict) -> None:
        try:
            pcm = base64.b64decode(event.get("audio", ""), validate=True)
        except Exception:
            await self.error("invalid_audio", "audio is not valid base64")
            return
        self.pcm_in += len(pcm)

    async def on_input_audio_buffer_commit(self, event: dict) -> None:
        # Phase 2 replaces this with VAD + upstream transcription.
        item_id = self.next_item("in")
        self.pcm_in = 0
        await self.send({"type": "input_audio_buffer.speech_started", "item_id": item_id, "audio_start_ms": 0})
        await self.send({"type": "input_audio_buffer.speech_stopped", "item_id": item_id, "audio_end_ms": 0})
        await self.send(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": "",
            }
        )

    async def on_input_audio_buffer_clear(self, event: dict) -> None:
        self.pcm_in = 0

    async def on_conversation_item_create(self, event: dict) -> None:
        item = event.get("item") or {}
        log.info("call item_create session=%s type=%s role=%s", self.session_id, item.get("type"), item.get("role"))

    async def on_conversation_item_delete(self, event: dict) -> None:
        log.info("call item_delete session=%s item=%s", self.session_id, event.get("item_id"))

    async def on_conversation_item_truncate(self, event: dict) -> None:
        log.info(
            "call truncate session=%s item=%s content=%s end_ms=%s",
            self.session_id,
            event.get("item_id"),
            event.get("content_index"),
            event.get("audio_end_ms"),
        )

    async def on_response_cancel(self, event: dict) -> None:
        rid = event.get("response_id")
        state = self.responses.get(rid)
        if state is None:
            await self.error("response_cancel_not_active", "No such active response")
            return
        state["cancelled"] = True

    async def on_response_create(self, event: dict) -> None:
        request = event.get("response") or {}
        rid = "resp_" + uuid.uuid4().hex[:16]
        state = {"cancelled": False, "metadata": request.get("metadata") or {}}
        self.responses[rid] = state
        await self.send(
            {
                "type": "response.created",
                "response": {"id": rid, "object": "realtime.response", "status": "in_progress", "metadata": state["metadata"]},
            }
        )
        item_id = self.next_item("out")
        text = "AIM Realtime Voice is connected."
        await self.send(
            {
                "type": "response.output_audio_transcript.delta",
                "response_id": rid,
                "item_id": item_id,
                "content_index": 0,
                "delta": text,
            }
        )
        pcm = beep_pcm()
        chunk = 4800  # 100 ms of PCM16 @ 24 kHz
        for off in range(0, len(pcm), chunk):
            if state["cancelled"]:
                break
            await self.send(
                {
                    "type": "response.output_audio.delta",
                    "response_id": rid,
                    "item_id": item_id,
                    "content_index": 0,
                    "delta": base64.b64encode(pcm[off : off + chunk]).decode(),
                }
            )
            await asyncio.sleep(0.01)
        await self.send(
            {
                "type": "response.output_audio_transcript.done",
                "response_id": rid,
                "item_id": item_id,
                "content_index": 0,
                "transcript": text,
            }
        )
        await self.send(
            {
                "type": "response.done",
                "response": {
                    "id": rid,
                    "object": "realtime.response",
                    "status": "cancelled" if state["cancelled"] else "completed",
                    "metadata": state["metadata"],
                },
            }
        )
        self.responses.pop(rid, None)

    def next_item(self, side: str) -> str:
        self.item_counter += 1
        return f"item_{side}_{self.item_counter}_{int(time.time() * 1000) % 100000}"


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": APP_VERSION, "active_calls": ACTIVE_CALLS, "config": CFG.masked()})


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    global ACTIVE_CALLS
    if CFG.api_key:
        auth = ws.headers.get("authorization", "")
        if auth != f"Bearer {CFG.api_key}":
            await ws.close(code=4401)
            return
    if ACTIVE_CALLS >= CFG.max_calls:
        await ws.close(code=4429)
        return
    ACTIVE_CALLS += 1
    try:
        await Call(ws, ws.query_params.get("model", "")).run()
    except WebSocketDisconnect:
        pass
    finally:
        ACTIVE_CALLS -= 1
        log.info("call closed active=%d", ACTIVE_CALLS)
